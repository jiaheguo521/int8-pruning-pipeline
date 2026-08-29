#!/usr/bin/env python3
"""relu-clip efflite4 student pruning (open-vocabulary).

Structured pruning of the published relu-clip student -- `tf_efficientnet_lite4`
distilled from CLIP ViT-L/14 into a 768-d embedding -- recovered by **cosine
feature distillation against the model as it was before pruning**.

Why this family exists next to `clip_rn50`:
Same objective, different starting point, and the difference is the point. The
`clip_rn50` baseline is CLIP's own RN50 image tower: 37.31 MiB int8, 16.09 MiB of
it streamed off-chip every frame, 51.51 ms. This baseline is a distilled CNN that
was *designed* against int8 and the device: 13.58 MiB, 7.50 MiB off-chip,
26.49 ms, and an int8 drop of 0.10 points where the RN50 tower's ReLU-trunk peers
lose 8 to 59. Pruning a model that already fits the accelerator's shape is a
different question from pruning one that does not, and the on-chip cap is what
makes it interesting: 7.60 MiB on-chip + 7.50 off-chip means the streamed half
disappears somewhere around a 45-50% parameter cut, and the repo's own
weight-transfer law (`results/latency_law/latency_law_fit.json`) says that is
where the latency return stops.

Recovery:
There is no dataset-method alternative here. Zero-shot classification is
image-embedding x text-embedding cosine, so there is no label space to fine-tune
a head against; the recovery objective is
`1 - cos(student(x), teacher(x))` against a FROZEN deepcopy of the unpruned
student, on UNLABELED images. See `int8_pruning.prune.recover` for the two methods and
which families can use which.

Allocation (local scope, deviating from clip_rn50):
`PRUNING_PROTOCOL["global_pruning"]` is **False** here, where `clip_rn50` has it
True. That is not a style choice. On the RN50 tower, global ranking x
mean-normalised magnitude at only -30% params left `stem.conv1` at 3 of 32
channels and `stages.0.0.conv2_kxk` at 5 of 64, and zero-shot went 99.0 -> 0.0 ->
70.75 after recovery (`results/clip_rn50/global_allocation.json`). EfficientNet-
Lite4's early projections are narrower still, and the Edge TPU's systolic array
is 64 wide -- channels cut below it cost accuracy and buy no time
(docs/PRUNING_HAZARDS.md section 2). `--global_pruning` flips it back if you want
to measure the fork rather than assume it.

Outputs (naming per the repo convention; recovery FT runs on ImageNet -> _imagenet)
    outputs/pytorch_pruned/relu_clip_imagenet_pruned<P>pct_<imp>.pt
    outputs/pruning_logs/relu_clip_imagenet_<imp>_<P>pct.json

Baseline: `python families/classification/relu_clip/download_baseline.py` (once).

Usage (smoke on the local ImageNet subset, no zero-shot):
    python families/classification/relu_clip/prune.py \
        --data_root data/datasets/Imagenet_1k \
        --checkpoints 30 --importance magnitude_l2 --final_epochs 1 --force
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent))
from int8_pruning.prune.core import (
    ALL_IMPORTANCES, count_macs, count_params, extract_layer_structure,
    get_importance, progressive_pruning_to_target, seed_everything,
)
from int8_pruning.prune.ladder import mode_suffix, resume_point
from int8_pruning.prune.recover import cosine_loss, distill_finetune
from student import INPUT_SIZE, student_transform  # sibling: families/classification/relu_clip/student.py

# Mirrors clip_rn50's budget and per-layer cap so the two CLIP-space families stay comparable.
PRUNING_PROTOCOL = {"iterative_steps": 200, "global_pruning": False,
                    "max_pruning_ratio": 0.9}
# Distillation recovery FT, same recipe as clip_rn50: realigning embeddings, not relearning a head.
RECIPE_FT = {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 1e-4,
             "scheduler": "cosine", "epochs": 10, "batch_size": 64}


def get_ignored_layers(model):
    """Protect the embedding head.

    `CnnStudent.head` is the single Linear that projects pooled features to the
    teacher's 768 dims. Pinning it fixes the output width -- the conv body still
    prunes, and Torch-Pruning resizes `head.in_features` to follow the backbone's
    surviving channels. Output stays [B, 768], so the pruned .pt flows through
    Phase 2 (per-channel int8) and Phase 3 (edgetpu) unchanged.
    """
    head = getattr(model, "head", None)
    if not isinstance(head, nn.Linear):
        raise RuntimeError(
            f"expected model.head to be nn.Linear, got {type(head).__name__}; "
            f"this worker only prunes families/classification/relu_clip/student.py:CnnStudent")
    return [head]


# Data
def build_distill_loader(train_dir, image_size, batch_size, seed):
    """Unlabeled image loader (labels ignored) for distillation. ImageFolder over
    <data_root>/train, with the published student transform."""
    ds = ImageFolder(str(train_dir), transform=student_transform(image_size))
    g = torch.Generator(); g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                       num_workers=4, pin_memory=True,
                                       drop_last=True, generator=g)


@torch.no_grad()
def teacher_cosine(student, teacher, loader, device, max_batches=8):
    """mean cos(student, teacher) over a few batches.

    The recovery metric that is always available: no labels, no text embeddings,
    and it measures exactly what the objective optimises. Upstream reports
    cos = 0.997 between the dense student's int8 and fp32 embeddings, so this is
    also the scale on which an int8-safe pruned rung has to land.
    """
    student.eval()
    tot, n = 0.0, 0
    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        se = F.normalize(student(images).float(), dim=-1)
        te = F.normalize(teacher(images).float(), dim=-1)
        tot += (se * te).sum(-1).sum().item(); n += images.size(0)
    return tot / max(1, n)


def load_text_emb(path):
    """(embs [C, D] float32, labels [C] str) from the published .npz.

    Accepts the two key spellings upstream ships: `embs`/`labels` for the
    ImageNet-1k matrix and `text_emb`/`labels` for the demo prompts.
    """
    z = np.load(path, allow_pickle=False)
    key = "embs" if "embs" in z.files else "text_emb"
    embs = z[key].astype(np.float32)
    labels = [str(x) for x in z["labels"]]
    return embs, labels


def _folder_to_row(val_dir, labels):
    """Map each val subfolder to a text-embedding row.

    Two alignments, because both trees exist in practice and guessing between
    them silently mis-scores every image:

    * by NAME, when every folder name appears in the .npz labels;
    * by POSITION, when the folder count equals the label count -- sorted wnids
      are ImageNet class-index order, which is the order the published matrix is
      built in.

    Anything else (a wnid subset against a 1000-row matrix, say) is refused
    rather than half-matched.
    """
    subs = sorted(p for p in Path(val_dir).iterdir() if p.is_dir())
    names = [p.name for p in subs]
    name_to_row = {n: i for i, n in enumerate(labels)}
    if all(n in name_to_row for n in names):
        return [(p, name_to_row[p.name]) for p in subs]
    if len(subs) == len(labels):
        return list(zip(subs, range(len(subs))))
    raise ValueError(
        f"cannot align {len(subs)} folders under {val_dir} with {len(labels)} "
        f"text-embedding rows: folder names are not label names "
        f"(e.g. {names[:2]} vs {labels[:2]}) and the counts differ. Point "
        f"--val_images_dir at the full val tree, or pass a matching --text_emb.")


@torch.no_grad()
def zeroshot_top1(model, val_dir, embs, labels, device, image_size, num_images=0):
    """Zero-shot top-1 of `model` on val_dir against `embs` (cosine)."""
    tf = student_transform(image_size)
    text = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    text_t = torch.from_numpy(text).to(device)
    from PIL import Image

    model.eval()
    c = n = 0
    for sub, row in _folder_to_row(val_dir, labels):
        files = []
        for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.png"):
            files += sorted(sub.glob(ext))
        for f in files:
            with Image.open(f) as im:
                x = tf(im.convert("RGB")).unsqueeze(0).to(device)
            emb = model(x).float()
            emb = emb / (emb.norm() + 1e-12)
            c += int(int((text_t @ emb.squeeze(0)).argmax()) == row); n += 1
            if num_images and n >= num_images:
                break
        if num_images and n >= num_images:
            break
    return (100.0 * c / n if n else 0.0), n


def run_one(imp_name, pct, args, device, models_dir, pruned_dir, log_dir):
    import torch_pruning as tp
    protocol = dict(PRUNING_PROTOCOL, global_pruning=bool(args.global_pruning))
    mode_sfx = mode_suffix(args.prune_mode)   # "" here -- see prune.ladder
    out_name = f"{args.baseline_name}_pruned{pct}pct_{imp_name}{mode_sfx}"
    out_path = pruned_dir / f"{out_name}.pt"
    log_path = log_dir / f"{args.baseline_name}_{imp_name}_{pct}pct{mode_sfx}.json"
    if out_path.exists() and not args.force:
        print(f"  [{out_name}] present, skip."); return None

    print(f"\n{'━'*70}\n  RELU_CLIP — {imp_name.upper()} — {pct}%\n{'━'*70}", flush=True)
    baseline_path = args.baseline_path or (models_dir / "relu_clip.pt")
    seed_everything(args.seed)

    student = torch.load(str(baseline_path), map_location="cpu", weights_only=False).to(device)
    teacher = copy.deepcopy(student).to(device)

    loader = build_distill_loader(Path(args.data_root) / "train", args.image_size,
                                  args.batch_size, args.seed)

    embs = labels = None
    if args.text_emb and args.val_images_dir:
        embs, labels = load_text_emb(args.text_emb)
    eval_fn = (lambda: zeroshot_top1(student, args.val_images_dir, embs, labels,
                                     device, args.image_size, args.eval_images)) \
        if embs is not None else None

    pre_params = count_params(student)
    pre_macs = count_macs(student, args.image_size)
    ref_zs = eval_fn() if eval_fn else (None, 0)
    if ref_zs[0] is not None:
        print(f"  Dense zero-shot top-1: {ref_zs[0]:.2f}% (n={ref_zs[1]})", flush=True)

    ignored = get_ignored_layers(student)
    ex = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    pruner = tp.pruner.MetaPruner(
        model=student, example_inputs=ex, importance=get_importance(imp_name),
        pruning_ratio=1.0, iterative_steps=protocol["iterative_steps"],
        global_pruning=protocol["global_pruning"],
        max_pruning_ratio=protocol["max_pruning_ratio"],
        round_to=args.round_to or None, ignored_layers=ignored)

    t0 = time.time()
    n_steps, actual_pct, post_params = progressive_pruning_to_target(student, pruner, pct)
    post_macs = count_macs(student, args.image_size)
    print(f"  [Pruning] {n_steps} steps in {time.time()-t0:.0f}s — {post_params:,} params "
          f"({actual_pct:.1f}% pruned), {post_macs:,.0f} MACs "
          f"(global={protocol['global_pruning']}, round_to={args.round_to or None})",
          flush=True)
    post_zs = eval_fn() if eval_fn else (None, 0)
    if post_zs[0] is not None:
        print(f"  Post-prune (no FT) zero-shot top-1: {post_zs[0]:.2f}%", flush=True)
    post_cos = teacher_cosine(student, teacher, loader, device)
    print(f"  Post-prune (no FT) cos(student, teacher): {post_cos:.4f}", flush=True)

    recipe = dict(RECIPE_FT)
    if args.lr is not None: recipe["lr"] = args.lr
    if args.weight_decay is not None: recipe["weight_decay"] = args.weight_decay
    epochs = args.final_epochs if args.final_epochs is not None else recipe["epochs"]
    print(f"  [Distill FT] {epochs} ep, AdamW lr={recipe['lr']:g} "
          f"wd={recipe['weight_decay']:g}, cosine — loss=1-cos(student,teacher)",
          flush=True)
    t1 = time.time()
    history = distill_finetune(
        student, teacher, loader, device, epochs, recipe,
        loss_fn=cosine_loss,
        metric_fn=lambda: (teacher_cosine(student, teacher, loader, device), ""),
        metric_name="teacher_cos")
    t_ft = time.time() - t1

    final_zs = eval_fn() if eval_fn else (None, 0)
    final_cos = teacher_cosine(student, teacher, loader, device)
    torch.save(student, str(out_path))

    result = {
        "model": args.baseline_name, "importance": imp_name, "checkpoint_pct": pct,
        "prune_mode": "independent_distill", "recovery": "distill",
        "dense_zeroshot_top1": ref_zs[0], "post_prune_zeroshot_top1": post_zs[0],
        "final_zeroshot_top1": final_zs[0], "n_eval": final_zs[1],
        "post_prune_teacher_cos": post_cos, "final_teacher_cos": final_cos,
        "pre_params": pre_params, "post_params": post_params,
        "pre_macs": pre_macs, "post_macs": post_macs,
        "param_reduction_pct": 100 * (1 - post_params / pre_params),
        "macs_reduction_pct": 100 * (1 - post_macs / pre_macs) if pre_macs else None,
        "actual_pct": actual_pct, "n_pruning_steps": n_steps,
        "round_to": args.round_to or None,
        "duration_s": {"finetune": t_ft}, "output_file": str(out_path),
        "recipe_ft": recipe, "pruning_protocol": protocol,
        "seed": args.seed, "image_size": args.image_size,
        "layer_structure": extract_layer_structure(student),
        "ft_history": history,
    }
    log_path.write_text(json.dumps(result, indent=2, default=str))
    fz = f"{final_zs[0]:.2f}%" if final_zs[0] is not None else "n/a"
    print(f"  ╔══ DONE  zero-shot {ref_zs[0] if ref_zs[0] else 'n/a'}→{fz}  "
          f"teacher_cos {post_cos:.4f}→{final_cos:.4f}  "
          f"params -{result['param_reduction_pct']:.1f}%  → {out_path.name}", flush=True)
    return result


def run_iterative(imp_name, checkpoints, args, device, models_dir, pruned_dir, log_dir):
    """LTH-style trajectory with the distillation TARGET pinned to the dense original.

    Same decision, and the same reasons, as families/classification/clip_rn50/prune.py's
    run_iterative -- read that docstring for the argument. In one line: LTH
    warm-starts the STUDENT, never the supervision target, so the teacher is one
    `copy.deepcopy` taken before the first cut and never refreshed. Letting it roll
    forward would be chained self-distillation, a different method, and it would
    also decouple the embedding from the OFFLINE text vectors that
    `--text_emb` scores against.

    `teacher_cos` matters more here than on clip_rn50: it is the metric that is
    always available (no text embeddings needed), and with a pinned teacher it keeps
    meaning the same thing at every rung -- cosine to the DENSE model. Under a
    rolling teacher it would silently become cosine-to-the-previous-rung, which
    drifts upward while the model gets worse.
    """
    import torch_pruning as tp
    protocol = dict(PRUNING_PROTOCOL, global_pruning=bool(args.global_pruning))
    mode_sfx = mode_suffix(args.prune_mode)   # "_iter"

    def out_path_for(cp):
        return pruned_dir / f"{args.baseline_name}_pruned{cp}pct_{imp_name}{mode_sfx}.pt"

    def log_path_for(cp):
        return log_dir / f"{args.baseline_name}_{imp_name}_{cp}pct{mode_sfx}.json"

    print(f"\n{'━'*70}\n  RELU_CLIP ITERATIVE — {imp_name.upper()} — {checkpoints}%"
          f"\n{'━'*70}", flush=True)
    baseline_path = args.baseline_path or (models_dir / "relu_clip.pt")
    seed_everything(args.seed)

    dense = torch.load(str(baseline_path), map_location="cpu", weights_only=False).to(device)
    teacher = copy.deepcopy(dense).to(device)   # PINNED for the whole trajectory

    loader = build_distill_loader(Path(args.data_root) / "train", args.image_size,
                                  args.batch_size, args.seed)
    embs = labels = None
    if args.text_emb and args.val_images_dir:
        embs, labels = load_text_emb(args.text_emb)

    def zs_of(m):
        if embs is None:
            return (None, 0)
        return zeroshot_top1(m, args.val_images_dir, embs, labels, device,
                             args.image_size, args.eval_images)

    # Dense reference: fixed for every rung's JSON AND for the ratio targets below.
    pre_params = count_params(dense)
    pre_macs = count_macs(dense, args.image_size)
    ref_zs = zs_of(teacher)
    if ref_zs[0] is not None:
        print(f"  Dense zero-shot top-1: {ref_zs[0]:.2f}% (n={ref_zs[1]})", flush=True)

    start_idx, resume_path = resume_point(checkpoints, out_path_for, force=args.force)
    if start_idx >= len(checkpoints):
        print(f"  All {len(checkpoints)} checkpoints already present, skip.")
        return []
    if resume_path is not None:
        print(f"  [resume] {start_idx} checkpoint(s) present; continuing from "
              f"{checkpoints[start_idx-1]}% ({resume_path.name})", flush=True)
        student = torch.load(str(resume_path), map_location="cpu",
                             weights_only=False).to(device)
        del dense
    else:
        student = dense

    # ONE pruner for the trajectory: iterative_steps becomes a cumulative budget and the cap stays anchored.
    ignored = get_ignored_layers(student)
    ex = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    pruner = tp.pruner.MetaPruner(
        model=student, example_inputs=ex, importance=get_importance(imp_name),
        pruning_ratio=1.0, iterative_steps=protocol["iterative_steps"],
        global_pruning=protocol["global_pruning"],
        max_pruning_ratio=protocol["max_pruning_ratio"],
        round_to=args.round_to or None, ignored_layers=ignored)

    results = []
    for cp in checkpoints[start_idx:]:
        print(f"\n  ── iterative step → {cp}% ──", flush=True)
        t0 = time.time()
        # initial_params pins the target to the DENSE count. See prune.ladder.
        n_steps, actual_pct, post_params = progressive_pruning_to_target(
            student, pruner, cp, initial_params=pre_params)
        post_macs = count_macs(student, args.image_size)
        print(f"  [Pruning] +{n_steps} steps in {time.time()-t0:.0f}s — {post_params:,} "
              f"params ({actual_pct:.1f}% of dense pruned), {post_macs:,.0f} MACs "
              f"(global={protocol['global_pruning']}, round_to={args.round_to or None})",
              flush=True)
        post_zs = zs_of(student)
        if post_zs[0] is not None:
            print(f"  Post-prune (no FT) zero-shot top-1: {post_zs[0]:.2f}%", flush=True)
        post_cos = teacher_cosine(student, teacher, loader, device)
        print(f"  Post-prune (no FT) cos(student, DENSE teacher): {post_cos:.4f}", flush=True)

        recipe = dict(RECIPE_FT)
        if args.lr is not None: recipe["lr"] = args.lr
        if args.weight_decay is not None: recipe["weight_decay"] = args.weight_decay
        epochs = args.final_epochs if args.final_epochs is not None else recipe["epochs"]
        print(f"  [Distill FT] {epochs} ep — loss=1-cos(student, DENSE teacher)", flush=True)

        t1 = time.time()
        history = distill_finetune(
            student, teacher, loader, device, epochs, recipe,
            loss_fn=cosine_loss,
            metric_fn=lambda: (teacher_cosine(student, teacher, loader, device), ""),
            metric_name="teacher_cos")
        t_ft = time.time() - t1

        final_zs = zs_of(student)
        final_cos = teacher_cosine(student, teacher, loader, device)
        torch.save(student, str(out_path_for(cp)))

        result = {
            "model": args.baseline_name, "importance": imp_name, "checkpoint_pct": cp,
            "prune_mode": "iterative_distill", "recovery": "distill",
            "warm_start": (start_idx > 0) or (cp != checkpoints[start_idx]),
            "teacher": "dense-pinned",
            "dense_zeroshot_top1": ref_zs[0], "post_prune_zeroshot_top1": post_zs[0],
            "final_zeroshot_top1": final_zs[0], "n_eval": final_zs[1],
            "post_prune_teacher_cos": post_cos, "final_teacher_cos": final_cos,
            "pre_params": pre_params, "post_params": post_params,
            "pre_macs": pre_macs, "post_macs": post_macs,
            "param_reduction_pct": 100 * (1 - post_params / pre_params),
            "macs_reduction_pct": 100 * (1 - post_macs / pre_macs) if pre_macs else None,
            "actual_pct": actual_pct, "n_pruning_steps": n_steps,
            "round_to": args.round_to or None,
            "duration_s": {"finetune": t_ft},
            "output_file": str(out_path_for(cp)),
            "recipe_ft": recipe, "pruning_protocol": protocol,
            "seed": args.seed, "image_size": args.image_size,
            "layer_structure": extract_layer_structure(student),
            "ft_history": history,
        }
        log_path_for(cp).write_text(json.dumps(result, indent=2, default=str))
        fz = f"{final_zs[0]:.2f}%" if final_zs[0] is not None else "n/a"
        print(f"  ╔══ DONE @{cp}%  zero-shot {ref_zs[0] if ref_zs[0] else 'n/a'}→{fz}  "
              f"teacher_cos {post_cos:.4f}→{final_cos:.4f}  "
              f"params -{result['param_reduction_pct']:.1f}%  "
              f"→ {out_path_for(cp).name}", flush=True)
        results.append(result)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_root", required=True,
                   help="ImageNet root holding train/ (distillation images; labels unused)")
    p.add_argument("--baseline_path", type=Path, default=None,
                   help="baseline .pt (default: MODELS_DIR/relu_clip.pt)")
    p.add_argument("--baseline_name", default="relu_clip_imagenet",
                   help="output filename prefix (default: relu_clip_imagenet)")
    p.add_argument("--text_emb", default=None,
                   help="text-embedding .npz for zero-shot eval (keys embs|text_emb + labels)")
    p.add_argument("--val_images_dir", default=None,
                   help="labeled val folder (ImageFolder) for zero-shot eval")
    p.add_argument("--eval_images", type=int, default=0,
                   help="cap zero-shot eval images (0=all; small N speeds smoke runs)")
    p.add_argument("--image_size", type=int, default=INPUT_SIZE)
    p.add_argument("--batch_size", type=int, default=RECIPE_FT["batch_size"])
    p.add_argument("--checkpoints", nargs="+", type=int, required=True)
    p.add_argument("--importance", nargs="+", default=None, choices=ALL_IMPORTANCES)
    p.add_argument("--final_epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--global_pruning", type=int, default=0,
                   help="1 = rank channels across the whole backbone, as clip_rn50 "
                        "does. Default 0 (per-layer): see the header for what "
                        "global scope did to the RN50 tower's stem")
    p.add_argument("--round_to", type=int, default=0,
                   help="round surviving channel counts up to a multiple of N "
                        "(0 = off). The Edge TPU's systolic array is 64 wide and "
                        "channels below it buy no time (PRUNING_HAZARDS section 2); "
                        "untested in this repo, so off by default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--prune_mode", choices=["independent", "iterative"],
                   default="independent",
                   help="independent (default): prune the dense baseline to each ratio "
                        "separately + distil. iterative: ONE trajectory, prune->distil->save "
                        "at each ratio in ascending order, with the teacher PINNED to the "
                        "dense original throughout (see run_iterative). Resumable; writes "
                        "an `_iter` filename so the two modes cannot overwrite each other.")
    args = p.parse_args(argv)

    outputs = Path(__file__).parents[3] / "outputs"
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
    print("PRUNING relu_clip (efflite4 / CLIP ViT-L/14 student) — cosine distillation recovery")
    print(f"  device={device}  data_root={args.data_root}")
    print(f"  importances={importances}  checkpoints={checkpoints}%")
    print(f"  zero-shot eval={'on' if args.text_emb and args.val_images_dir else 'off (teacher cosine only)'}")
    print("=" * 70, flush=True)

    results = []
    for imp in importances:
        if args.prune_mode == "iterative":
            # One try/except for the WHOLE trajectory: a dead rung leaves every later rung unreachable anyway.
            try:
                results.extend(run_iterative(imp, checkpoints, args, device,
                                             models_dir, pruned_dir, log_dir))
            except Exception as e:
                import traceback
                print(f"\n  [{imp} iterative] FAILED {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            continue
        for cp in checkpoints:
            try:
                r = run_one(imp, cp, args, device, models_dir, pruned_dir, log_dir)
                if r:
                    results.append(r)
            except Exception as e:
                import traceback
                print(f"\n  [{imp} @{cp}%] FAILED {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
    print(f"\n{'='*70}\nDONE — {len(results)} runs", flush=True)


if __name__ == "__main__":
    main()
