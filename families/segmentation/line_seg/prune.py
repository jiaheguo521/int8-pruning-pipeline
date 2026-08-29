#!/usr/bin/env python3
"""Lane-following segmentation pruning (line_seg_*).

Structured (channel-level) pruning of the three upstream line-seg models
(`line_seg_base128` / `line_seg_w96` / `line_seg_link_r34`), with recovery
fine-tuning on the upstream self-collected dataset. Phase-1 worker for the
fourth prunable family in this repo, per
`<upstream>/docs/Lane_Seg_Pruning_Handoff.md`.

⚠ `pruned<P>pct` HERE MEANS CHANNEL RATIO, NOT PARAMETER RATIO:
Every other family in this repo (classification / detection / re-id) reads
`pruned50pct` as "50% of the PARAMETERS removed" — their workers iterate a
pruner until a parameter target is hit. This family follows the handoff's
measured byte table instead, which pins `MetaPruner(pruning_ratio=r).step()`
in ONE shot: torch_pruning's `pruning_ratio` is a per-layer CHANNEL ratio, and
params fall roughly as r^2. Verified against handoff §E.1:

    w96  30% channels -> 4,219,138 / 8,633,857  = 0.489 ~ 0.7^2
    r34  60% channels -> 4,531,181 / 28,507,649 = 0.159 ~ 0.4^2

Re-labelling those as `pruned51pct` / `pruned84pct` would break both the
"ratio must be a multiple of 10%" delivery constraint and every lookup into
the handoff's already-measured on-chip/off-chip table. So the channel-ratio
convention wins here, and every log records BOTH `channel_ratio_pct` and the
resulting `param_reduction_pct` so the two conventions never get confused.

`--ratio_semantics param` is the DEFAULT: `--checkpoints` is a reduction in
remaining PARAMETERS, reached over 400 incremental steps, so this family sits on
the same x-axis as the DSD 2026 paper and as every other family here. It writes
`..._pruned<P>pct_<imp>_param.pt`.

`--ratio_semantics channel` is the retired round-1 convention, where the number
was a per-layer CHANNEL ratio and parameters fell roughly as r^2 -- `pruned60pct`
on link_r34 was an 84.1% parameter cut. That ladder was withdrawn on 2026-08-22
(see results/_deleted/); the flag is kept only to reproduce it. Ranking stays
local and the fine-tune recipe is unchanged in both modes;
`prune_to_param_target` documents the measurements behind those two deviations.

Why a separate worker:
  - The output is a 4-D `[1,1,H/2,W/2]` mask-logit map, not a `[1,D]` embedding
    (re-id/CLIP) and not a multi-output head pyramid (detection). None of this
    repo's eval paths apply; accuracy is scored upstream.
  - Recovery FT needs the upstream self-collected dataset — lane-following
    segmentation has no public substitute — plus its held-out protocol
    (contiguous-block split, scoring at native ROI resolution, track-only IoU,
    floor-only misfire rate). All of that is imported verbatim from the upstream
    `lane_seg_data` module; nothing is re-implemented here.
  - The output head is a 1x1 conv with `out_channels == 1`. That 1 IS the task
    definition (single-channel mask logits), so it goes into `ignored_layers`;
    pruning it destroys the output contract.

Only the ~20-line training loop is local: upstream `train_lane_seg.py`'s `run()`
is welded to model construction (it always builds a fresh model from
`VARIANTS[name]["build"]()`), so there is no "continue training this module"
entry point to call. The recipe it implements is the upstream `mix` stage,
parameter for parameter: 60/40 real/synthetic, dice+BCE with pos_weight=12,
AdamW lr=8e-4 + cosine, 30 epochs.

Outputs (handoff naming; no dataset segment — the dataset is implicit)
    outputs/pytorch_pruned/<variant>_pruned<P>pct_<imp>.pt
    outputs/pruning_logs/<variant>_<imp>_<P>pct.json

Usage:
    python families/segmentation/line_seg/prune.py --variant line_seg_w96 \
        --upstream ../MicroROS-Pi5_Coral_TPU/training --checkpoints 10 30
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

ALL_IMPORTANCES = ["magnitude_l1", "magnitude_l2", "fpgm", "random", "lamp"]
# Upstream `train_lane_seg.py` mix stage, verbatim: lr/epochs/pos_weight are what the baselines
# themselves were trained with, and drifting makes the pruned rows incomparable to the baseline rows.
RECIPE_FT = {"optimizer": "adamw", "lr": 8e-4, "weight_decay": 1e-4,
             "scheduler": "cosine", "epochs": 30, "batch_size": 32,
             "samples_per_epoch": 3000, "pos_weight": 12.0}
# handoff §E: uniform per-layer channel ratio, one-shot step, output head pinned.
PRUNING_PROTOCOL = {"global_pruning": False, "one_shot": True,
                    "ratio_semantics": "per-layer channel ratio (NOT param ratio)"}
# `--ratio_semantics param` overrides the two ratio keys above so the sweep lands on the paper's
# {10..90}% PARAMETER grid. Ranking stays local and the fine-tune recipe is untouched.
PRUNING_PROTOCOL_PARAM = {"global_pruning": False, "iterative_steps": 400,
                          "max_pruning_ratio": 0.9,
                          "ratio_semantics": "parameter ratio (paper-conformant)"}


def load_upstream(upstream: Path):
    """Import the upstream data/eval module and variant table.

    Also gates on `families/segmentation/line_seg/lane_seg_models.py` being byte-identical to the
    upstream original: the local copy exists so delivered .pt files unpickle
    without the upstream repo on disk (handoff §B), and two copies that silently
    diverge would produce models that load into the wrong architecture.
    """
    if not (upstream / "lane_seg_data.py").is_file():
        raise SystemExit(f"--upstream does not look like the upstream training/ dir: {upstream}")

    def _sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    local_mod = Path(__file__).parent / "lane_seg_models.py"
    up_mod = upstream / "lane_seg_models.py"
    if _sha(local_mod) != _sha(up_mod):
        raise SystemExit(
            f"families/segmentation/line_seg/lane_seg_models.py has drifted from {up_mod}.\n"
            f"  local:    {_sha(local_mod)}\n  upstream: {_sha(up_mod)}\n"
            f"Re-run families/segmentation/line_seg/setup.sh to refresh the copy.")

    sys.path.insert(0, str(upstream))
    import lane_seg_data as D
    from lane_seg_models import VARIANTS
    return D, VARIANTS


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_importance(name):
    import torch_pruning as tp
    return {"magnitude_l1": lambda: tp.importance.MagnitudeImportance(p=1),
            "magnitude_l2": lambda: tp.importance.MagnitudeImportance(p=2),
            "fpgm": tp.importance.FPGMImportance,
            "random": tp.importance.RandomImportance,
            "lamp": lambda: tp.importance.LAMPImportance(p=2)}[name]()


def prune_one_shot(model, ratio, imp_name, size):
    """handoff §E, verbatim: one-shot uniform per-layer channel pruning with the
    single-channel output head pinned."""
    import torch_pruning as tp

    head = [m for m in model.modules() if isinstance(m, nn.Conv2d) and m.out_channels == 1]
    if len(head) != 1:
        raise RuntimeError(
            f"expected exactly one out_channels==1 conv (the mask-logit head); "
            f"found {len(head)}. Refusing to prune — the output contract is at risk.")
    tp.pruner.MetaPruner(
        model=model, example_inputs=torch.randn(1, 3, size, size),
        importance=get_importance(imp_name), pruning_ratio=ratio / 100.0,
        ignored_layers=head, global_pruning=PRUNING_PROTOCOL["global_pruning"],
    ).step()
    return model


def prune_to_param_target(model, target_pct, imp_name, size):
    """Incremental local pruning until `target_pct` of PARAMETERS are gone.

    The paper expresses its target as a reduction in remaining parameters and
    reaches it over 400 incremental steps, so this is the mode that puts this
    family on the same x-axis as the paper and as the detection families.

    Two of the paper's axes are deliberately NOT adopted, both measured:

    * Ranking stays local. Under `global_pruning=True` at a matched 51%
      parameter cut, w96's stem `e1.c` falls 96 -> 20 (20.8%) and link_r34's
      `enc.l3.3.conv1` falls 256 -> 34 (13.3%). The same concentration pattern
      on EfficientDet costs 30 points of relative mAP that recovery
      fine-tuning does not win back.
    * `RECIPE_FT` stays AdamW with `pos_weight`. Lane masks are a few thin
      tape pixels against everything else; the paper's CIFAR-100 cross-entropy
      recipe has no equivalent term.

    Returns (n_steps, realized_pct).
    """
    import torch_pruning as tp

    head = [m for m in model.modules() if isinstance(m, nn.Conv2d) and m.out_channels == 1]
    if len(head) != 1:
        raise RuntimeError(
            f"expected exactly one out_channels==1 conv (the mask-logit head); "
            f"found {len(head)}. Refusing to prune — the output contract is at risk.")
    pruner = tp.pruner.MetaPruner(
        model=model, example_inputs=torch.randn(1, 3, size, size),
        importance=get_importance(imp_name), pruning_ratio=1.0,
        iterative_steps=PRUNING_PROTOCOL_PARAM["iterative_steps"],
        global_pruning=PRUNING_PROTOCOL_PARAM["global_pruning"],
        max_pruning_ratio=PRUNING_PROTOCOL_PARAM["max_pruning_ratio"],
        ignored_layers=head)
    initial = count_params(model)
    target = initial * (1 - target_pct / 100)
    steps = 0
    while count_params(model) > target:
        if pruner.current_step >= pruner.iterative_steps:
            print(f"  [WARN] step budget exhausted at "
                  f"{100*(1-count_params(model)/initial):.1f}% < target {target_pct}%",
                  flush=True)
            break
        pruner.step(); steps += 1
    return steps, 100 * (1 - count_params(model) / initial)


def finetune(model, D, split, v, epochs, batch, samples, workers, device, seed):
    """Upstream `train_lane_seg.py` mix stage. Returns the per-epoch history."""
    if epochs <= 0:
        return []
    size, mask = v["input"], v["mask"]
    mean, std = v["mean255"], v["std255"]

    real = D.RealSet(split["track"]["train"] + split["floor"]["train"],
                     size, mask, mean, std, aug=True)
    synth = D.SynthSet(samples, 1, size, mask, mean, std)
    dl = torch.utils.data.DataLoader(
        D.MixSet(real, synth, samples), batch_size=batch, shuffle=True,
        num_workers=workers, drop_last=True, pin_memory=(device.type == "cuda"),
        persistent_workers=workers > 0)

    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE_FT["lr"],
                            weight_decay=RECIPE_FT["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pw = torch.tensor(RECIPE_FT["pos_weight"], device=device)

    history = []
    for ep in range(epochs):
        model.train()
        t0 = time.time(); losses = []
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = D.dice_bce(model(x), y, pw)
            loss.backward(); opt.step()
            losses.append(float(loss))
        sched.step()
        row = {"epoch": ep + 1, "loss": round(float(np.mean(losses)), 4),
               "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            row.update({k: round(x, 4) for k, x
                        in D.evaluate(model, size, mean, std, split).items()})
        history.append(row)
        print("    " + json.dumps(row), flush=True)
    return history


def run_one(variant, ratio, imp_name, args, D, VARIANTS, device, split):
    v = VARIANTS[variant]
    size, mask = v["input"], v["mask"]
    out_dir = Path(args.out_dir)
    log_dir = Path(__file__).parents[3] / "outputs" / "pruning_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # `_param` keeps the two ladders from colliding: the same `pruned50pct` means 50% of CHANNELS in
    # the default mode and 50% of PARAMETERS here. It still matches family.yaml's
    # `_pruned\d+pct(?:_\w+)?` because `\w` covers the underscore.
    param_mode = getattr(args, "ratio_semantics", "channel") == "param"
    sfx = "_param" if param_mode else ""
    out_name = (f"{variant}_pruned{ratio}pct_{imp_name}{sfx}" if ratio > 0
                else f"{variant}_pruned0pct")
    out_path = out_dir / f"{out_name}.pt"
    log_path = log_dir / f"{variant}_{imp_name}_{ratio}pct{sfx}.json"
    if out_path.exists() and not args.force:
        print(f"  [{out_name}] present, skip.")
        return None

    epochs = args.epochs if not args.no_finetune else 0
    print(f"\n{'━'*70}\n  {variant.upper()} — {imp_name.upper()} — {ratio}% "
          f"{'PARAMETERS' if param_mode else 'channels'}"
          f"{'  (NO finetune)' if args.no_finetune else ''}\n{'━'*70}", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    baseline = args.baseline_path or (
        Path(__file__).parents[3] / "outputs" / "models" / f"{variant}.pt")
    model = torch.load(str(baseline), map_location="cpu", weights_only=False)
    model.eval()
    pre_params = count_params(model)

    t0 = time.time()
    n_steps, realized_pct = 0, 0.0
    if ratio > 0:
        if param_mode:
            n_steps, realized_pct = prune_to_param_target(model, ratio, imp_name, size)
        else:
            prune_one_shot(model, ratio, imp_name, size)
    post_params = count_params(model)
    # Output-contract gate: the pinned head must still emit [1,1,mask,mask].
    with torch.no_grad():
        out_shape = tuple(model(torch.randn(1, 3, size, size)).shape)
    if out_shape != (1, 1, mask, mask):
        raise RuntimeError(f"output contract broken: got {out_shape}, expected (1,1,{mask},{mask})")
    param_red = 100 * (1 - post_params / pre_params)
    print(f"  [Pruning] {time.time()-t0:.0f}s — {post_params:,} params "
          f"({pre_params/max(1,post_params):.2f}x) — "
          f"{'param_target' if param_mode else 'channel_ratio'}={ratio}% "
          f"=> param_reduction={param_red:.1f}%, out={out_shape}", flush=True)

    model = model.to(device)
    if args.no_finetune:
        history, met = [], None
    else:
        print(f"  [Recovery FT] {epochs} ep, AdamW lr={RECIPE_FT['lr']:g} "
              f"wd={RECIPE_FT['weight_decay']:g} cosine, dice+BCE pos_weight="
              f"{RECIPE_FT['pos_weight']:g}, {args.samples} samples/ep bs={args.batch}",
              flush=True)
        t1 = time.time()
        history = finetune(model, D, split, v, epochs, args.batch, args.samples,
                           args.workers, device, args.seed)
        t_ft = time.time() - t1
        met = D.evaluate(model, size, v["mean255"], v["std255"], split)
        print(f"  HELD-OUT: val_iou={met['val_iou']:.4f} floor_fire={met['floor_fire']:.4f} "
              f"(n={met['n_track']}/{met['n_floor']})", flush=True)

    model.eval().to("cpu")   # the delivered .pt must be device-agnostic (upstream does the same)
    torch.save(model, str(out_path))

    result = {
        "model": variant, "importance": imp_name,
        "prune_mode": ("structured_incremental_param" if param_mode
                       else "structured_one_shot_channel"),
        "ratio_semantics": "param" if param_mode else "channel",
        # BOTH numbers, always. In channel mode `pct` in the filename is the channel ratio; in param mode
        # it is the parameter target, and `realized_param_pct` is what was actually reached.
        "channel_ratio_pct": None if param_mode else ratio,
        "param_target_pct": ratio if param_mode else None,
        "realized_param_pct": round(realized_pct, 2) if param_mode else None,
        "n_pruning_steps": n_steps if param_mode else 1,
        "pre_params": pre_params, "post_params": post_params,
        "param_reduction_pct": param_red,
        "compression_x": pre_params / max(1, post_params),
        "input_size": size, "mask_size": mask,
        "mean255": v["mean255"], "std255": v["std255"],
        "finetuned": not args.no_finetune,
        "recipe_ft": None if args.no_finetune else RECIPE_FT,
        "epochs": epochs, "samples_per_epoch": args.samples, "batch_size": args.batch,
        "duration_s": {"finetune": None if args.no_finetune else round(t_ft, 1)},
        "pruning_protocol": PRUNING_PROTOCOL_PARAM if param_mode else PRUNING_PROTOCOL,
        "seed": args.seed,
        "output_file": str(out_path), "ft_history": history,
    }
    if met is not None:
        result.update({k: (round(x, 6) if isinstance(x, float) else x) for k, x in met.items()})
        result["oracle_ceiling"] = round(D.oracle_ceiling(mask, split), 4)
    log_path.write_text(json.dumps(result, indent=2, default=str))
    tail = "" if met is None else (f"  val_iou={met['val_iou']:.4f} "
                                   f"floor_fire={met['floor_fire']:.4f}")
    print(f"  ╔══ DONE  params -{param_red:.1f}% "
          f"({result['compression_x']:.2f}x){tail}  → {out_path.name}", flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", required=True,
                   help="line_seg_base128 | line_seg_w96 | line_seg_link_r34")
    p.add_argument("--upstream", type=Path, required=True,
                   help="the upstream MicroROS-Pi5_Coral_TPU training/ directory "
                        "(supplies lane_seg_data + the self-collected dataset)")
    p.add_argument("--baseline_path", type=Path, default=None,
                   help="baseline .pt (default: outputs/models/<variant>.pt)")
    p.add_argument("--checkpoints", nargs="+", type=int, required=True,
                   help="CHANNEL ratios in percent (handoff §E.1 semantics), e.g. 10 30")
    p.add_argument("--importance", nargs="+", default=["magnitude_l2"],
                   choices=ALL_IMPORTANCES,
                   help="handoff §E pins magnitude_l2; the others are for sweeps")
    p.add_argument("--out_dir", type=Path,
                   default=Path(__file__).parents[3] / "outputs" / "pytorch_pruned")
    p.add_argument("--no_finetune", action="store_true",
                   help="prune and save without recovery FT. Bytes and latency depend "
                        "only on the STRUCTURE, so this is the cheap way to map the "
                        "on-chip/off-chip crossover — the models are NOT deliverable.")
    p.add_argument("--epochs", type=int, default=RECIPE_FT["epochs"])
    p.add_argument("--batch", type=int, default=RECIPE_FT["batch_size"])
    p.add_argument("--samples", type=int, default=RECIPE_FT["samples_per_epoch"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--ratio_semantics", choices=["channel", "param"], default="param",
                   help="'param' (default) sweeps the DSD 2026 paper's target: "
                        "--checkpoints is a per-layer CHANNEL ratio and params fall "
                        "roughly as r^2. 'param' sweeps the paper's target instead -- "
                        "a reduction in remaining PARAMETERS, reached incrementally -- "
                        "and writes to a '_param' suffixed stem so the two ladders "
                        "never share a filename. Ranking and the FT recipe are "
                        "identical in both modes.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    D, VARIANTS = load_upstream(args.upstream.resolve())
    if args.variant not in VARIANTS:
        raise SystemExit(f"--variant {args.variant} not in upstream VARIANTS: {list(VARIANTS)}")
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = D.build_split()
    print("=" * 70)
    print(f"PRUNING {args.variant} (lane-following segmentation) — upstream mix-stage recovery")
    print(f"  device={device}  upstream={args.upstream}")
    print(f"  channel ratios={sorted(args.checkpoints)}%  importances={args.importance}")
    print(f"  train track/floor={len(split['track']['train'])}/{len(split['floor']['train'])}"
          f"  val={len(split['track']['val'])}/{len(split['floor']['val'])}")
    print("=" * 70, flush=True)

    results = []
    for imp in args.importance:
        for r in sorted(args.checkpoints):
            try:
                out = run_one(args.variant, r, imp, args, D, VARIANTS, device, split)
                if out:
                    results.append(out)
            except Exception as e:
                import traceback
                print(f"\n  [{imp} @{r}%] FAILED {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
    print(f"\n{'='*70}\nDONE — {len(results)} runs", flush=True)


if __name__ == "__main__":
    main()
