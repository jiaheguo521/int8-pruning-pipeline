#!/usr/bin/env python3
"""Youtu Re-ID baseline_lite pruning (reid_youtu_lite).

Structured (channel-level) pruning of the four-source YouReID `baseline_lite`
embedder, with **feature-distillation** recovery — the Phase-1 worker for the
Re-ID ratio sweep ("pruning ratio vs accuracy vs latency vs on-chip residency").

Structured, not sparse: the Edge TPU does NOT exploit sparsity, so unstructured
pruning leaves the weight byte count — and therefore the streaming cost and the
latency — completely unchanged. Only real channel removal shrinks the model.

Why a separate worker (not families/classification/clip_rn50/prune.py / families/classification/imagenet_backbones/prune.py):
  - Input is 256x128 (H x W), not square. The shared `build_transforms` helper is
    square-only (Resize + RandomResizedCrop) and unusable here: Re-ID uses a
    DIRECT resize because the pedestrian crops are already tight.
  - The output dim is pinned by a 1x1 **Conv2d** (`embedding_layer`), not by a
    Linear head, so the CLIP worker's `get_ignored_layers` finds nothing.
  - No labels are used: Duke/MSMT17/CUHK03 are unobtainable (Duke withdrawn,
    MSMT17 licensed), so CrossEntropy/triplet recovery on the ORIGINAL four-source
    label space is impossible. Distillation sidesteps it — the teacher IS the
    four-source model, so its multi-source knowledge transfers without labels.

Pruning mechanics (verified at bring-up):
MetaPruner with `embedding_layer` in `ignored_layers`. DepGraph traces the
`cat([gap(x), gmp(x)], 1)` (two branches off ONE tensor into a 1x1 conv) and
resizes `embedding_layer.in_channels` (4096 -> 2*C) automatically as the resnet
body shrinks, while its out_channels — the 768-d output — stay pinned. Measured:
  0% -> 26.66M (1.00x) | 20% -> 21.3M (1.25x) | 50% -> 13.3M (2.00x)
 70% ->  7.97M (3.35x) | 80% ->  5.26M (5.07x) | 90% ->  2.60M (10.25x)

**The pinned head dominates at high ratios**: embedding_layer is 11.8% of params
at 0% but 48.5% at 90% (1.26M of 2.60M) — so `--embed_dim 256` exists to prune it
down first and hand the budget back to the backbone (0.95M vs 0.41M of backbone
at the 95% capacity-matched point).

Distillation losses:
  cosine  `1 - cos(student(x), teacher(x))`. Same dim as the teacher, so it pins
          the embedding directly. Default for the 768-d mainline.
  simmat  `MSE(cos_sim(student_batch), cos_sim(teacher_batch))`. Dimension-AGNOSTIC
          (required when --embed_dim != 768) and constrains cosine structure in the
          student's OWN space, which is what downstream ranking actually uses.
          NB: a learnable projector + cosine would be wrong here — preserving
          cosine AFTER a linear map does not preserve cosine in the student space.

Outputs (naming per the repo convention; recovery distillation runs on Market crops
-> _market1501, even though the baseline weights are four-source and unsuffixed)
    outputs/pytorch_pruned/reid_youtu_lite[_e256]_market1501_pruned<P>pct_<imp>.pt
    outputs/pruning_logs/reid_youtu_lite[_e256]_market1501_<imp>_<P>pct.json

Usage:
    python families/re-identification/reid/prune.py \
        --market_dir data/datasets/market1501 \
        --checkpoints 20 50 70 80 90 --importance magnitude_l2
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms

from int8_pruning.prune.recover import DISTILL_LOSSES, distill_finetune

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent))

ALL_IMPORTANCES = ["magnitude_l1", "magnitude_l2", "fpgm", "random", "lamp"]
REID_H, REID_W = 256, 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BASE_EMBED_DIM = 768

# global_pruning is FALSE here, unlike the other workers, and that is measured rather
# than preferred. At 90%, cos(student, teacher) after an identical 1 epoch of
# distillation, with stem channels surviving of 64:
#
#     protocol                 post-prune   after 1 ep   stem    head%   backbone
#     global + magnitude_l2      0.088        0.390       1/64    64.7%   0.93M
#     global + lamp              0.109        0.688      50/64     9.2%   2.41M
#     local  + magnitude_l2      0.117        0.766      17/64    32.8%   1.78M
#     local  + lamp              0.117        0.766      17/64    32.8%   1.78M
#
# Global magnitude ranks RAW filter norms across layers, and resnet50's deep filters
# out-norm the stem's, so it deletes the early layers wholesale and even loses to global
# RANDOM (0.100). The two `local` rows tying to 4 decimals is the lesson: once a uniform
# per-layer ratio fixes the ALLOCATION, the within-layer criterion barely moves the
# number. `global + lamp` has MORE backbone (2.41M) and still loses, its taper
# (stem 78% -> layer4 8%) gutting the layer where Re-ID semantics live.
PRUNING_PROTOCOL = {"iterative_steps": 400, "global_pruning": False,
                    "max_pruning_ratio": 0.97}  # PER-LAYER cap; 90% global forces layers past 0.9
# Distillation recovery FT, 30 ep not the CLIP worker's 10: Market's train split is 12936 images.
RECIPE_FT = {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 1e-4,
             "scheduler": "cosine", "epochs": 30, "batch_size": 64}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_importance(name):
    import torch_pruning as tp
    return {"magnitude_l1": lambda: tp.importance.MagnitudeImportance(p=1),
            "magnitude_l2": lambda: tp.importance.MagnitudeImportance(p=2),
            "fpgm": tp.importance.FPGMImportance,
            "random": tp.importance.RandomImportance,
            "lamp": lambda: tp.importance.LAMPImportance(p=2)}[name]()


def progressive_pruning_to_target(model, pruner, target_pct, initial_params):
    target = initial_params * (1 - target_pct / 100)
    n = 0
    while count_params(model) > target:
        if pruner.current_step >= pruner.iterative_steps:
            print(f"  [WARN] iterative_steps cap hit before target {target_pct}%", flush=True)
            break
        pruner.step(); n += 1
    actual_pct = 100 * (1 - count_params(model) / initial_params)
    return n, actual_pct, count_params(model)


def shrink_embedding_head(model, new_dim, example_inputs):
    """Prune `embedding_layer`'s OUT channels 768 -> new_dim, keeping the highest
    L2-norm filters (and shrinking the following BN with them, via DepGraph).

    Runs BEFORE the body pruning so the freed budget goes to the backbone: at the
    95% point the head falls from 69% of the model (0.91M, backbone 0.41M) to 27%
    (0.36M, backbone 0.95M).
    """
    import torch_pruning as tp

    conv = model.embedding_layer
    keep = new_dim
    if keep >= conv.out_channels:
        return model
    w = conv.weight.detach()
    norms = w.flatten(1).norm(p=2, dim=1)
    drop = torch.argsort(norms)[: conv.out_channels - keep].tolist()

    dg = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
    group = dg.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=drop)
    group.prune()
    return model


# Data — unlabeled pedestrian crops
class _CropFolder(torch.utils.data.Dataset):
    """Unlabeled crops. Direct resize to 256x128 (no RandomResizedCrop — the
    pedestrian boxes are already tight), hflip, ImageNet norm. Same preprocessing
    contract as _calib_market1501 in int8_pruning.convert.tflite."""

    def __init__(self, paths):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize((REID_H, REID_W), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def collect_distill_images(market_dir, extra_dirs):
    """Market bounding_box_train + any --extra_image_dirs, unlabeled.

    Extra dirs matter: distillation only pins the teacher WHERE WE SAMPLE. The
    teacher is four-source, but a student distilled on Market crops alone is only
    held to the teacher on Market-like inputs — and at high pruning ratios capacity
    is scarce enough that it will sacrifice unsampled domains. Dropping unlabeled
    crops from the deployment camera in here costs nothing (the teacher supplies
    the targets) and is the highest-leverage knob for a non-Market viewpoint.
    """
    train_dirs = sorted(Path(market_dir).glob("**/bounding_box_train"))
    if not train_dirs:
        raise FileNotFoundError(f"No bounding_box_train under {market_dir}")
    paths = sorted(p for p in train_dirs[0].glob("*.jpg")
                   if not p.name.startswith(("-1", "0000")))
    n_market = len(paths)
    for d in extra_dirs or []:
        extra = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png")
                       for p in Path(d).rglob(ext))
        if not extra:
            raise FileNotFoundError(f"--extra_image_dirs {d} matched no images")
        paths += extra
    print(f"  [data] {len(paths)} distill images ({n_market} Market"
          f"{f' + {len(paths)-n_market} extra' if len(paths) > n_market else ''})")
    return paths


def build_distill_loader(paths, batch_size, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        _CropFolder(paths), batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True, generator=g)


# Distillation recovery FT
@torch.no_grad()
def teacher_fidelity(student, teacher, loader, device, max_batches=8):
    """mean cos(student, teacher) on held-out-ish batches. Reported as a
    first-class metric: it measures how much of the four-source teacher survived,
    needs no labels, and (unlike Market rank-1) can be computed on ANY domain —
    including the deployment camera, via --extra_image_dirs."""
    if student.embedding_layer.out_channels != teacher.embedding_layer.out_channels:
        return None  # dims differ (--embed_dim); cosine to the teacher undefined
    student.eval()
    tot, n = 0.0, 0
    for i, images in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        se = F.normalize(student(images).float(), dim=-1)
        te = F.normalize(teacher(images).float(), dim=-1)
        tot += (se * te).sum(-1).sum().item(); n += images.size(0)
    return tot / max(1, n)


def run_one(imp_name, pct, args, device, models_dir, pruned_dir, log_dir, paths):
    import torch_pruning as tp

    protocol = dict(PRUNING_PROTOCOL, global_pruning=bool(args.global_pruning))
    stem = args.baseline_name  # full output stem, dataset included (repo convention)
    out_name = f"{stem}_pruned{pct}pct_{imp_name}" if pct > 0 else f"{stem}_pruned0pct"
    out_path = pruned_dir / f"{out_name}.pt"
    log_path = log_dir / f"{stem}_{imp_name}_{pct}pct.json"
    if out_path.exists() and not args.force:
        print(f"  [{out_name}] present, skip."); return None

    print(f"\n{'━'*70}\n  REID_YOUTU_LITE — {imp_name.upper()} — {pct}%  "
          f"(embed_dim={args.embed_dim}, loss={args.distill_loss})\n{'━'*70}", flush=True)
    baseline_path = args.baseline_path or (models_dir / f"{args.baseline_name}.pt")
    seed_everything(args.seed)

    student = torch.load(str(baseline_path), map_location="cpu", weights_only=False).to(device)
    teacher = copy.deepcopy(student).to(device)
    loader = build_distill_loader(paths, args.batch_size, args.seed)

    ex = torch.randn(1, 3, REID_H, REID_W).to(device)
    pre_params = count_params(student)

    if args.embed_dim != BASE_EMBED_DIM:
        shrink_embedding_head(student, args.embed_dim, ex)
        print(f"  [head] embedding_layer out_channels {BASE_EMBED_DIM} -> "
              f"{student.embedding_layer.out_channels}  "
              f"({count_params(student)/1e6:.3f}M params)", flush=True)

    pruner = tp.pruner.MetaPruner(
        model=student, example_inputs=ex, importance=get_importance(imp_name),
        pruning_ratio=1.0, iterative_steps=protocol["iterative_steps"],
        global_pruning=protocol["global_pruning"],
        max_pruning_ratio=protocol["max_pruning_ratio"],
        ignored_layers=[student.embedding_layer])  # pins the embedding dim

    t0 = time.time()
    # Target is relative to the ORIGINAL 26.66M, so an --embed_dim run is budget-comparable at the same pct.
    n_steps, actual_pct, post_params = progressive_pruning_to_target(
        student, pruner, pct, pre_params)
    with torch.no_grad():
        out_dim = int(student(ex).shape[1])
    head = student.embedding_layer
    head_params = head.in_channels * head.out_channels
    print(f"  [Pruning] {n_steps} steps in {time.time()-t0:.0f}s — {post_params:,} params "
          f"({actual_pct:.1f}% pruned, {pre_params/max(1,post_params):.2f}x), out={out_dim}-d",
          flush=True)
    print(f"  [Pruning] embedding_layer {head.in_channels}->{head.out_channels} = "
          f"{head_params/1e6:.2f}M ({100*head_params/post_params:.1f}% of the model), "
          f"backbone {(post_params-head_params)/1e6:.2f}M", flush=True)

    pre_fid = teacher_fidelity(student, teacher, loader, device)
    if pre_fid is not None:
        print(f"  Post-prune (no FT) teacher_cos: {pre_fid:.4f}", flush=True)

    recipe = dict(RECIPE_FT)
    if args.lr is not None: recipe["lr"] = args.lr
    if args.weight_decay is not None: recipe["weight_decay"] = args.weight_decay
    epochs = args.final_epochs if args.final_epochs is not None else recipe["epochs"]
    print(f"  [Distill FT] {epochs} ep, AdamW lr={recipe['lr']:g} "
          f"wd={recipe['weight_decay']:g}, cosine sched — loss={args.distill_loss}", flush=True)
    t1 = time.time()
    history = distill_finetune(
        student, teacher, loader, device, epochs, recipe,
        loss_fn=DISTILL_LOSSES[args.distill_loss],
        metric_fn=lambda: (teacher_fidelity(student, teacher, loader, device), ""),
        metric_name="teacher_cos", loss_name=args.distill_loss)
    t_ft = time.time() - t1
    final_fid = teacher_fidelity(student, teacher, loader, device)

    student.eval()
    torch.save(student, str(out_path))

    result = {
        "model": stem, "importance": imp_name,
        "checkpoint_pct": pct, "prune_mode": "structured_distill",
        "embed_dim": out_dim, "distill_loss": args.distill_loss,
        "pre_params": pre_params, "post_params": post_params,
        "param_reduction_pct": 100 * (1 - post_params / pre_params),
        "compression_x": pre_params / max(1, post_params),
        "head_params": head_params, "head_frac_of_model": head_params / post_params,
        "backbone_params": post_params - head_params,
        "actual_pct": actual_pct, "n_pruning_steps": n_steps,
        "post_prune_teacher_cos": pre_fid, "final_teacher_cos": final_fid,
        "n_distill_images": len(paths), "extra_image_dirs": args.extra_image_dirs,
        "duration_s": {"finetune": t_ft}, "output_file": str(out_path),
        "recipe_ft": recipe, "pruning_protocol": protocol,
        "seed": args.seed, "input_hw": [REID_H, REID_W],
        "ft_history": history,
    }
    log_path.write_text(json.dumps(result, indent=2, default=str))
    fstr = f"{final_fid:.4f}" if final_fid is not None else "n/a (dim differs)"
    print(f"  ╔══ DONE  teacher_cos {pre_fid if pre_fid else float('nan'):.4f}→{fstr}  "
          f"params -{result['param_reduction_pct']:.1f}% "
          f"({result['compression_x']:.2f}x)  → {out_path.name}", flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--market_dir", required=True,
                   help="Market-1501 root (bounding_box_train supplies the distill images)")
    p.add_argument("--extra_image_dirs", nargs="+", default=None,
                   help="extra UNLABELED crop dirs mixed into distillation (e.g. the "
                        "deployment camera's own view — the teacher supplies targets, "
                        "so no labels are needed)")
    p.add_argument("--baseline_path", type=Path, default=None,
                   help="baseline .pt (default: MODELS_DIR/reid_youtu_lite.pt)")
    p.add_argument("--baseline_name", default="reid_youtu_lite_market1501",
                   help="FULL output stem, dataset included (repo convention). The "
                        "baseline .pt itself is unsuffixed — its weights are four-source; "
                        "the _market1501 here is the distillation set")
    p.add_argument("--embed_dim", type=int, default=BASE_EMBED_DIM,
                   help="768 = pin the released head (drop-in). Smaller = prune the head "
                        "first and give the budget to the backbone; needs --distill_loss simmat")
    p.add_argument("--distill_loss", default="cosine", choices=["cosine", "simmat"])
    p.add_argument("--global_pruning", type=int, default=0, choices=[0, 1],
                   help="0 (default) = uniform per-layer ratio, 1 = compare importance "
                        "across layers. Load-bearing: a GLOBAL magnitude compare deletes "
                        "resnet50's stem (1/64 channels survive at 90%%) and halves the "
                        "recovered fidelity (0.39 vs 0.77). See PRUNING_PROTOCOL.")
    p.add_argument("--batch_size", type=int, default=RECIPE_FT["batch_size"])
    p.add_argument("--checkpoints", nargs="+", type=int, required=True)
    p.add_argument("--importance", nargs="+", default=None, choices=ALL_IMPORTANCES)
    p.add_argument("--final_epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    if args.embed_dim != BASE_EMBED_DIM and args.distill_loss == "cosine":
        raise SystemExit(
            f"--embed_dim {args.embed_dim} with --distill_loss cosine is invalid: the "
            f"student's {args.embed_dim}-d embedding has no cosine against the teacher's "
            f"{BASE_EMBED_DIM}-d one. Use --distill_loss simmat.")

    # parents[3] = repo root; see the same note in families/detection/effdet/prune.py.
    outputs = Path(__file__).resolve().parents[3] / "outputs"
    models_dir, pruned_dir, log_dir = (outputs / "models",
                                       outputs / "pytorch_pruned",
                                       outputs / "pruning_logs")
    pruned_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    importances = args.importance or ALL_IMPORTANCES
    checkpoints = sorted(args.checkpoints)
    print("=" * 70)
    print("PRUNING reid_youtu_lite (Youtu Re-ID baseline_lite) — distillation recovery")
    print(f"  device={device}  market_dir={args.market_dir}")
    print(f"  embed_dim={args.embed_dim}  distill_loss={args.distill_loss}")
    print(f"  importances={importances}  checkpoints={checkpoints}%")
    print("=" * 70, flush=True)

    paths = collect_distill_images(args.market_dir, args.extra_image_dirs)

    results = []
    for imp in importances:
        for cp in checkpoints:
            try:
                r = run_one(imp, cp, args, device, models_dir, pruned_dir, log_dir, paths)
                if r:
                    results.append(r)
            except Exception as e:
                import traceback
                print(f"\n  [{imp} @{cp}%] FAILED {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
    print(f"\n{'='*70}\nDONE — {len(results)} runs", flush=True)


if __name__ == "__main__":
    main()
