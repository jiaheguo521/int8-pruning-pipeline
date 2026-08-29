#!/usr/bin/env python3
"""CLIP RN50 image-tower pruning (vision-language branch).

Structured **backbone-only** pruning of the CLIP RN50 image tower
(`clip_rn50`), with **cosine feature-distillation** recovery — the Phase-1
worker for the multimodal / open-vocabulary axis.

Why a separate worker (not `families/classification/imagenet_backbones/prune.py`):
CLIP has no class labels: zero-shot is image-embedding × text-embedding cosine.
So the classification worker's CrossEntropy recovery FT and top-1-vs-label eval
do NOT apply. This worker instead:
  - **Recovery objective** = `1 - cos(student_embed(x), teacher_embed(x))` against
    a FROZEN copy of the unpruned tower, on UNLABELED images (ImageNet train).
    No text, no labels — it restores the embedding direction the pruning broke.
  - **Eval metric** = zero-shot top-1 on a labeled val folder using precomputed
    text embeddings (`families/classification/clip_rn50/text_embeddings.py`), if provided; else the
    held-out cosine-to-teacher.

Pruning mechanics (verified at bring-up):
torch_pruning MetaPruner with the AttentionPool2d head's Linears (q/k/v/proj) in
`ignored_layers`. That pins the 1024-d embedding output and the attention dim;
the conv body still prunes (its out_channels propagate into q/k/v *in*_features +
the head positional embedding, which Torch-Pruning resizes automatically). Output
stays `[B, 1024]`, so the pruned .pt flows through Phase 2 (per-channel int8) and
Phase 3 (edgetpu) unchanged.

Outputs (naming per the repo convention; recovery FT runs on ImageNet → _imagenet)
    outputs/pytorch_pruned/clip_rn50_imagenet_pruned<P>pct_<imp>.pt
    outputs/pruning_logs/clip_rn50_imagenet_<imp>_<P>pct.json

Usage (smoke on the local Imagenette tree):
    python families/classification/clip_rn50/prune.py \
        --data_root data/datasets/Imagenet_1k \
        --val_images_dir data/datasets/Imagenet_1k/train \
        --text_emb outputs/models/clip_rn50_text_imagenette.npy \
        --checkpoints 30 --importance magnitude_l2 --final_epochs 1 --force
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
from PIL import Image, ImageFile
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent))
from int8_pruning.data.classification import build_transforms, norm_for_model
from int8_pruning.prune.core import (
    ALL_IMPORTANCES, count_params, get_importance,
    progressive_pruning_to_target, seed_everything,
)
from int8_pruning.prune.ladder import mode_suffix, resume_point
from int8_pruning.prune.recover import cosine_loss, distill_finetune
from eval import WNID_TO_NAME  # sibling: families/classification/clip_rn50/eval.py

IMAGE_SIZE_DEFAULT = 224

# Backbone-only protocol, mirroring imagenet_backbones/prune.py so the breadth axis stays comparable.
PRUNING_PROTOCOL = {"iterative_steps": 200, "global_pruning": True,
                    "max_pruning_ratio": 0.9}
# Distillation recovery FT (AdamW on the cosine-distill loss): realigning embeddings, not a head.
RECIPE_FT = {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 1e-4,
             "scheduler": "cosine", "epochs": 10, "batch_size": 64}








def get_ignored_layers(model):
    """Protect the AttentionPool2d head: every Linear under model.head (q/k/v/proj).
    Pins the 1024-d embedding output + attention dim; the conv body still prunes."""
    head = getattr(model, "head", None)
    ignored = [m for m in (head.modules() if head is not None else [])
               if isinstance(m, nn.Linear)]
    if not ignored:  # fallback: last Linear in the tree
        last = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last = m
        if last is not None:
            ignored = [last]
    return ignored




# Data
def build_distill_loader(train_dir, image_size, batch_size, mean, std, seed):
    """Unlabeled image loader (labels ignored) for distillation. ImageFolder over
    <data_root>/train, with the CLIP-norm TRAIN transform."""
    train_tf, _ = build_transforms(image_size, mean=mean, std=std)
    ds = ImageFolder(str(train_dir), transform=train_tf)
    g = torch.Generator(); g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                       num_workers=4, pin_memory=True,
                                       drop_last=True, generator=g)


@torch.no_grad()
def zeroshot_top1(model, val_dir, text_emb, labels, device, mean, std,
                  image_size, num_images=0):
    """Zero-shot top-1 of `model` on val_dir vs text_emb (cosine). Folder name ->
    label via WNID_TO_NAME, else the folder name itself."""
    _, eval_tf = build_transforms(image_size, mean=mean, std=std)
    text = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    text_t = torch.from_numpy(text.astype(np.float32)).to(device)
    name_to_idx = {n: i for i, n in enumerate(labels)}
    model.eval()
    c = n = 0
    for sub in sorted(p for p in Path(val_dir).iterdir() if p.is_dir()):
        lname = WNID_TO_NAME.get(sub.name, sub.name)
        if lname not in name_to_idx:
            continue
        lbl = name_to_idx[lname]
        files = []
        for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.png"):
            files += sorted(sub.glob(ext))
        for f in files:
            x = eval_tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
            emb = model(x).float()
            emb = emb / (emb.norm() + 1e-12)
            pred = int((text_t @ emb.squeeze(0)).argmax())
            c += int(pred == lbl); n += 1
            if num_images and n >= num_images:
                break
        if num_images and n >= num_images:
            break
    return (100.0 * c / n if n else 0.0), n


def run_one(imp_name, pct, args, device, models_dir, pruned_dir, log_dir):
    import torch_pruning as tp
    mode_sfx = mode_suffix(args.prune_mode)   # "" here -- see prune.ladder
    out_name = f"{args.baseline_name}_pruned{pct}pct_{imp_name}{mode_sfx}"
    out_path = pruned_dir / f"{out_name}.pt"
    log_path = log_dir / f"{args.baseline_name}_{imp_name}_{pct}pct{mode_sfx}.json"
    if out_path.exists() and not args.force:
        print(f"  [{out_name}] present, skip."); return None

    print(f"\n{'━'*70}\n  CLIP_RN50 — {imp_name.upper()} — {pct}%\n{'━'*70}", flush=True)
    baseline_path = args.baseline_path or (models_dir / f"{args.baseline_name}.pt")
    mean, std = norm_for_model("clip_rn50")
    seed_everything(args.seed)

    student = torch.load(str(baseline_path), map_location="cpu", weights_only=False).to(device)
    teacher = copy.deepcopy(student).to(device)

    loader = build_distill_loader(Path(args.data_root) / "train", args.image_size,
                                  args.batch_size, mean, std, args.seed)

    text_emb = labels = None
    if args.text_emb and args.val_images_dir:
        text_emb = np.load(args.text_emb)
        labels_path = Path(str(Path(args.text_emb).with_suffix("")) + ".labels.json")
        labels = json.loads(labels_path.read_text())
    eval_fn = (lambda: zeroshot_top1(student, args.val_images_dir, text_emb, labels,
                                     device, mean, std, args.image_size, args.eval_images)) \
        if text_emb is not None else None

    pre_params = count_params(student)
    ref_zs = eval_fn() if eval_fn else (None, 0)
    if ref_zs[0] is not None:
        print(f"  Dense zero-shot top-1: {ref_zs[0]:.2f}% (n={ref_zs[1]})", flush=True)

    ignored = get_ignored_layers(student)
    ex = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    pruner = tp.pruner.MetaPruner(
        model=student, example_inputs=ex, importance=get_importance(imp_name),
        pruning_ratio=1.0, iterative_steps=PRUNING_PROTOCOL["iterative_steps"],
        global_pruning=PRUNING_PROTOCOL["global_pruning"],
        max_pruning_ratio=PRUNING_PROTOCOL["max_pruning_ratio"], ignored_layers=ignored)

    t0 = time.time()
    n_steps, actual_pct, post_params = progressive_pruning_to_target(student, pruner, pct)
    print(f"  [Pruning] {n_steps} steps in {time.time()-t0:.0f}s — {post_params:,} params "
          f"({actual_pct:.1f}% pruned)", flush=True)
    post_zs = eval_fn() if eval_fn else (None, 0)
    if post_zs[0] is not None:
        print(f"  Post-prune (no FT) zero-shot top-1: {post_zs[0]:.2f}%", flush=True)

    recipe = dict(RECIPE_FT)
    if args.lr is not None: recipe["lr"] = args.lr
    if args.weight_decay is not None: recipe["weight_decay"] = args.weight_decay
    epochs = args.final_epochs if args.final_epochs is not None else recipe["epochs"]
    print(f"  [Distill FT] {epochs} ep, AdamW lr={recipe['lr']:g} wd={recipe['weight_decay']:g}, "
          f"cosine — loss=1-cos(student,teacher)", flush=True)
    def zs_metric():
        top1, n = eval_fn()
        return top1, f" (n={n})"

    t1 = time.time()
    history = distill_finetune(student, teacher, loader, device, epochs, recipe,
                               loss_fn=cosine_loss,
                               metric_fn=zs_metric if eval_fn else None,
                               metric_name="zeroshot_top1")
    t_ft = time.time() - t1

    final_zs = eval_fn() if eval_fn else (None, 0)
    torch.save(student, str(out_path))

    result = {
        "model": args.baseline_name, "importance": imp_name, "checkpoint_pct": pct,
        "prune_mode": "independent_distill",
        "dense_zeroshot_top1": ref_zs[0], "post_prune_zeroshot_top1": post_zs[0],
        "final_zeroshot_top1": final_zs[0], "n_eval": final_zs[1],
        "pre_params": pre_params, "post_params": post_params,
        "param_reduction_pct": 100 * (1 - post_params / pre_params),
        "actual_pct": actual_pct, "n_pruning_steps": n_steps,
        "duration_s": {"finetune": t_ft}, "output_file": str(out_path),
        "recipe_ft": recipe, "pruning_protocol": PRUNING_PROTOCOL,
        "seed": args.seed, "image_size": args.image_size,
        "ft_history": history,
    }
    log_path.write_text(json.dumps(result, indent=2, default=str))
    fz = f"{final_zs[0]:.2f}%" if final_zs[0] is not None else "n/a"
    print(f"  ╔══ DONE  zero-shot {ref_zs[0] if ref_zs[0] else 'n/a'}→{fz}  "
          f"params -{result['param_reduction_pct']:.1f}%  → {out_path.name}", flush=True)
    return result


def run_iterative(imp_name, checkpoints, args, device, models_dir, pruned_dir, log_dir):
    """LTH-style trajectory: prune -> distil -> save -> keep pruning the RECOVERED model.

    The teacher is PINNED to the dense original for the whole trajectory -- one
    `copy.deepcopy` taken before the first cut, never refreshed. That is the
    decision this mode turns on, and it is what keeps it the same experiment as
    `families/classification/imagenet_backbones/prune.py --prune_mode iterative`: LTH warm-starts the
    STUDENT, never the supervision target. classification's target is labels, which
    do not change between rungs; here it is the dense model, which therefore must
    not either.

    Rolling the teacher forward (teacher = the recovered previous rung) is a
    DIFFERENT method -- chained self-distillation -- and would compound: this family
    measures dense 99.0 -> post-prune 0.0 -> recovered 70.75 at only -30%
    (outputs/pruning_logs/clip_rn50_imagenet_magnitude_l2_30pct.json), so rung two
    would already be aiming at a 70.75 target. It would also break the eval outright:
    zero-shot is image-embedding x text-embedding cosine against text vectors
    computed OFFLINE (`--text_emb ....npy`), so an image space that drifts with a
    rolling teacher stops being comparable to that fixed basis. If you ever want it,
    give it its own prune_mode and its own MODE_SUFFIX entry -- reusing `_iter`
    would put two different methods on one filename again.
    """
    import torch_pruning as tp
    mode_sfx = mode_suffix(args.prune_mode)   # "_iter"

    def out_path_for(cp):
        return pruned_dir / f"{args.baseline_name}_pruned{cp}pct_{imp_name}{mode_sfx}.pt"

    def log_path_for(cp):
        return log_dir / f"{args.baseline_name}_{imp_name}_{cp}pct{mode_sfx}.json"

    print(f"\n{'━'*70}\n  CLIP_RN50 ITERATIVE — {imp_name.upper()} — {checkpoints}%"
          f"\n{'━'*70}", flush=True)
    baseline_path = args.baseline_path or (models_dir / f"{args.baseline_name}.pt")
    mean, std = norm_for_model("clip_rn50")
    seed_everything(args.seed)

    dense = torch.load(str(baseline_path), map_location="cpu", weights_only=False).to(device)
    teacher = copy.deepcopy(dense).to(device)   # PINNED for the whole trajectory

    loader = build_distill_loader(Path(args.data_root) / "train", args.image_size,
                                  args.batch_size, mean, std, args.seed)
    text_emb = labels = None
    if args.text_emb and args.val_images_dir:
        text_emb = np.load(args.text_emb)
        labels_path = Path(str(Path(args.text_emb).with_suffix("")) + ".labels.json")
        labels = json.loads(labels_path.read_text())

    def zs_of(m):
        if text_emb is None:
            return (None, 0)
        return zeroshot_top1(m, args.val_images_dir, text_emb, labels, device,
                             mean, std, args.image_size, args.eval_images)

    # Dense reference, fixed for every rung's JSON AND for the ratio targets below.
    pre_params = count_params(dense)
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
        pruning_ratio=1.0, iterative_steps=PRUNING_PROTOCOL["iterative_steps"],
        global_pruning=PRUNING_PROTOCOL["global_pruning"],
        max_pruning_ratio=PRUNING_PROTOCOL["max_pruning_ratio"], ignored_layers=ignored)

    results = []
    for cp in checkpoints[start_idx:]:
        print(f"\n  ── iterative step → {cp}% ──", flush=True)
        t0 = time.time()
        # initial_params pins the target to the DENSE count; without it "40%" means 40% of an already-pruned model.
        n_steps, actual_pct, post_params = progressive_pruning_to_target(
            student, pruner, cp, initial_params=pre_params)
        print(f"  [Pruning] +{n_steps} steps in {time.time()-t0:.0f}s — {post_params:,} "
              f"params ({actual_pct:.1f}% of dense pruned)", flush=True)
        post_zs = zs_of(student)
        if post_zs[0] is not None:
            print(f"  Post-prune (no FT) zero-shot top-1: {post_zs[0]:.2f}%", flush=True)

        recipe = dict(RECIPE_FT)
        if args.lr is not None: recipe["lr"] = args.lr
        if args.weight_decay is not None: recipe["weight_decay"] = args.weight_decay
        epochs = args.final_epochs if args.final_epochs is not None else recipe["epochs"]
        print(f"  [Distill FT] {epochs} ep — loss=1-cos(student, DENSE teacher)", flush=True)

        def zs_metric():
            top1, n = zs_of(student)
            return top1, f" (n={n})"

        t1 = time.time()
        history = distill_finetune(student, teacher, loader, device, epochs, recipe,
                                   loss_fn=cosine_loss,
                                   metric_fn=zs_metric if text_emb is not None else None,
                                   metric_name="zeroshot_top1")
        t_ft = time.time() - t1
        final_zs = zs_of(student)
        torch.save(student, str(out_path_for(cp)))

        result = {
            "model": args.baseline_name, "importance": imp_name, "checkpoint_pct": cp,
            "prune_mode": "iterative_distill",
            "warm_start": (start_idx > 0) or (cp != checkpoints[start_idx]),
            "teacher": "dense-pinned",
            "dense_zeroshot_top1": ref_zs[0], "post_prune_zeroshot_top1": post_zs[0],
            "final_zeroshot_top1": final_zs[0], "n_eval": final_zs[1],
            "pre_params": pre_params, "post_params": post_params,
            "param_reduction_pct": 100 * (1 - post_params / pre_params),
            "actual_pct": actual_pct, "n_pruning_steps": n_steps,
            "duration_s": {"finetune": t_ft},
            "output_file": str(out_path_for(cp)),
            "recipe_ft": recipe, "pruning_protocol": PRUNING_PROTOCOL,
            "seed": args.seed, "image_size": args.image_size,
            "ft_history": history,
        }
        log_path_for(cp).write_text(json.dumps(result, indent=2, default=str))
        fz = f"{final_zs[0]:.2f}%" if final_zs[0] is not None else "n/a"
        print(f"  ╔══ DONE @{cp}%  zero-shot {ref_zs[0] if ref_zs[0] else 'n/a'}→{fz}  "
              f"params -{result['param_reduction_pct']:.1f}%  "
              f"→ {out_path_for(cp).name}", flush=True)
        results.append(result)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_root", required=True,
                   help="ImageNet root holding train/ (distillation images; labels unused)")
    p.add_argument("--baseline_path", type=Path, default=None,
                   help="baseline .pt (default: MODELS_DIR/clip_rn50.pt)")
    p.add_argument("--baseline_name", default="clip_rn50_imagenet",
                   help="output filename prefix (default: clip_rn50_imagenet)")
    p.add_argument("--text_emb", default=None,
                   help="text-embedding .npy for zero-shot eval (sibling .labels.json)")
    p.add_argument("--val_images_dir", default=None,
                   help="labeled val folder (ImageFolder) for zero-shot eval")
    p.add_argument("--eval_images", type=int, default=0,
                   help="cap zero-shot eval images (0=all; small N speeds smoke runs)")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE_DEFAULT)
    p.add_argument("--batch_size", type=int, default=RECIPE_FT["batch_size"])
    p.add_argument("--checkpoints", nargs="+", type=int, required=True)
    p.add_argument("--importance", nargs="+", default=None, choices=ALL_IMPORTANCES)
    p.add_argument("--final_epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
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
    print(f"PRUNING clip_rn50 (CLIP RN50 image tower) — cosine distillation recovery")
    print(f"  device={device}  data_root={args.data_root}")
    print(f"  importances={importances}  checkpoints={checkpoints}%")
    print(f"  zero-shot eval={'on' if args.text_emb and args.val_images_dir else 'off (cosine-to-teacher only)'}")
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
