#!/usr/bin/env python3
"""Structured pruning of classification models following the PruningBench protocol (Li et al. 2024, arXiv:2406.12315).

The worker is family-agnostic: it `torch.load`s a full nn.Module baseline and
uses an architecture-agnostic `get_ignored_layers` (classifier attr or last
Linear fallback). Families wired through `scripts/pruning.sh` (CLS_MODEL):
    mobilenetv2, efficientnet_lite0 (timm), mnasnet1_0, squeezenet1_1, resnet50
Input normalization is per-family (see int8_pruning.data.classification.norm_for_model,
keyed on --baseline_name): efficientnet_lite0 uses mean=std=0.5, the rest use
ImageNet stats. flowers102/inat_plant baselines exist only for mobilenetv2.

Supported datasets (`--dataset`):
  - `imagenet` (default): ImageNet-1k (ILSVRC2012), 1000 classes, read
    straight from the server ImageFolder tree ($DATASET_DIR/Imagenet_1k).
    The prune baseline IS the torchvision pretrained `mobilenetv2_imagenet.pt`
    (head already 1280->1000), so no finetune is needed.
    ⚠ The 100 ep PruningBench recovery FT is enormous on 1.28M images --
    pass --final_epochs <small> for real runs.
  - `flowers102` (fast bring-up): Oxford Flowers-102, 102 classes,
    downloaded by torchvision. 100 ep recovery FT is fast here
    (~30 min on A6000 for the full sweep).
  - `inat_plant`: iNat 2017 Plantae, 2101 classes. Requires the full
    iNat 2017 download (~185 GB -> ~30-40 GB after Plantae filtering).

Differences vs the CIFAR-100 pruning study this started from (7 models, 11
importances):
  - Dataset: Flowers-102 (default) or iNat 2017 Plantae (opt-in) instead
    of the CIFAR-100 pickle.
  - Model  : one baseline per invocation (--baseline_name), 224x224,
    per-family normalization.
  - Importances: only the 5 robust *data-free* subset:
        magnitude_l1, magnitude_l2, fpgm, random, lamp
    (removed: bn_scale, taylor, obdc, fisher, group_lasso, hrank -- too costly
    or not robust at ImageNet scale).
  - No sparsity learning (all 5 importances are data-free).
  - No outer loop over models: single `--baseline_name`.

PruningBench protocol kept identical:
  - iterative_steps=400, global_pruning=True, max_pruning_ratio=0.9
  - Recovery FT: PruningBench's *ImageNet* recipe -- SGD-Nesterov lr=0.1
    m=0.9 wd=1e-4, MultiStep [30,60] γ=0.1, 90 ep (see RECIPE_FT below).
    NOT the CIFAR-100 recipe of the DSD paper (lr=0.01 wd=5e-4 [60,80] 100 ep);
    this worker is a fork onto Flowers-102/iNat/ImageNet and does not
    reproduce the published sweep.

Baseline contract:
    outputs/models/<baseline_name>.pt must be an `nn.Module` object saved
    via `torch.save(model, ...)` (not a state_dict). The classification head
    must be reachable as a `classifier` attribute (Linear / Conv2d, possibly
    inside a Sequential) or, failing that, as the last Linear in the module
    tree (ResNet's `.fc`). Baselines come from
    `download_and_finetune_models.sh` (raw ImageNet downloads) or
    `families/classification/imagenet_backbones/finetune.py` (finetuned MobileNetV2 variants).

Outputs:
    outputs/pytorch_pruned/<baseline_name>_pruned<P>pct_<imp>.pt
    outputs/pruning_logs/<baseline_name>_<imp>_<P>pct.json
    outputs/pruning_logs/<baseline_name>_pruning_summary.json

Usage -- Flowers-102 (default):
    python families/classification/imagenet_backbones/prune.py \\
        --data_root <path>/datasets/flowers102 \\
        --checkpoints 10 20 30 40 50 60 70 80 90 \\
        --importance magnitude_l1 magnitude_l2 fpgm random lamp

Usage -- iNat 2017 Plantae:
    python families/classification/imagenet_backbones/prune.py \\
        --dataset inat_plant \\
        --inat_train_json <path>/train2017.json \\
        --inat_val_json   <path>/val2017.json \\
        --inat_image_root <path>/train_val_images \\
        --baseline_name mobilenetv2_inat \\
        --checkpoints 10 20 30 40 50 60 70 80 90 \\
        --importance magnitude_l1 magnitude_l2 fpgm random lamp

    # Smoke (1 ep FT):
    python families/classification/imagenet_backbones/prune.py \\
        --checkpoints 30 --importance magnitude_l1 \\
        --final_epochs 1 --force
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent))
from int8_pruning.prune.ladder import mode_suffix, resume_point
from int8_pruning.prune.core import (
    ALL_IMPORTANCES, count_macs, count_params, extract_layer_structure,
    get_importance, progressive_pruning_to_target, seed_everything,
)
from int8_pruning.data.classification import (
    build_classification_dataloaders,
    norm_for_model,
    DATASET_CHOICES,
    DATASET_NUM_CLASSES,
)


DEFAULT_DATASET = "imagenet"
DEFAULT_BASELINE_NAME = {
    "imagenet": "mobilenetv2_imagenet",
    "flowers102": "mobilenetv2_flowers102",
    "inat_plant": "mobilenetv2_inat",
}
DEFAULT_NUM_CLASSES = dict(DATASET_NUM_CLASSES)  # {imagenet:1000, flowers102:102, inat_plant:2101}
IMAGE_SIZE_DEFAULT = 224



# PruningBench recipe (identical to 01_pruning.py)
RECIPE_FT = {
    "optimizer": "sgd", "momentum": 0.9, "weight_decay": 1e-4, "nesterov": True,
    "lr": 0.1,
    "scheduler": "multistep",
    "lr_decay_epochs": [30, 60], "lr_decay_rate": 0.1,
    "epochs": 90,
    "batch_size": 128,
    # PruningBench (Li et al. 2024) *ImageNet* recovery FT, not their CIFAR100 one.
    # Batch comes from --batch_size at runtime; lr=0.1 is tuned for bs=128.
    "ref": "PruningBench (Li et al. 2024) ImageNet recovery FT",
}

PRUNING_PROTOCOL = {
    "iterative_steps": 400,
    "global_pruning": True,
    "max_pruning_ratio": 0.9,
    "ref": "PruningBench progressive_pruning + 10% protection",
}


# Reproducibility


def get_dataloaders(args, image_size=224):
    """Dispatch on `args.dataset`. Returns (train_loader, val_loader, test_loader).

    Flowers-102 has a proper test split (6149 images, vs 1020 val), returned as a
    distinct loader for the final FP32 bootstrap CI95 eval. For iNat and ImageNet
    there is no separate test split — `test_loader` aliases `val_loader`.

    Normalization is per-family, keyed on --baseline_name (efficientnet_lite0
    expects mean=std=0.5, everything else ImageNet stats).
    """
    mean, std = norm_for_model(args.baseline_name)
    train_loader, val_loader, test_loader, _ = build_classification_dataloaders(
        args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
        image_size=image_size,
        inat_train_json=getattr(args, "inat_train_json", None),
        inat_val_json=getattr(args, "inat_val_json", None),
        inat_image_root=getattr(args, "inat_image_root", None),
        mean=mean, std=std,
    )
    return train_loader, val_loader, test_loader


# Evaluation
def evaluate(model, val_loader, device):
    """Top-1 accuracy on val_loader, in %."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_with_topk_ci(model, val_loader, device, n_bootstrap=1000, seed=0):
    """FP32 final eval: Top-1 + Top-5 + bootstrap CI95 (Efron 1979)."""
    model.eval()
    correct1, correct5 = [], []
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(images)
        top5 = out.topk(5, dim=1).indices
        correct1.append((top5[:, 0] == labels).cpu().numpy().astype(np.uint8))
        correct5.append((top5 == labels.unsqueeze(1)).any(dim=1).cpu().numpy().astype(np.uint8))

    arr1 = np.concatenate(correct1) if correct1 else np.array([], dtype=np.uint8)
    arr5 = np.concatenate(correct5) if correct5 else np.array([], dtype=np.uint8)
    n = len(arr1)
    if n == 0:
        return {"top1_pct": 0.0, "top5_pct": 0.0,
                "top1_ci95_lo": 0.0, "top1_ci95_hi": 0.0,
                "top5_ci95_lo": 0.0, "top5_ci95_hi": 0.0, "n_eval": 0}

    rng = np.random.RandomState(seed)
    means1 = np.empty(n_bootstrap, dtype=np.float64)
    means5 = np.empty(n_bootstrap, dtype=np.float64)
    for k in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        means1[k] = arr1[idx].mean()
        means5[k] = arr5[idx].mean()
    return {
        "top1_pct": float(100.0 * arr1.mean()),
        "top5_pct": float(100.0 * arr5.mean()),
        "top1_ci95_lo": float(100.0 * np.percentile(means1, 2.5)),
        "top1_ci95_hi": float(100.0 * np.percentile(means1, 97.5)),
        "top5_ci95_lo": float(100.0 * np.percentile(means5, 2.5)),
        "top5_ci95_hi": float(100.0 * np.percentile(means5, 97.5)),
        "n_eval": int(n),
    }




def train_epoch(model, train_loader, criterion, optimizer, device):
    """One SGD epoch. Returns (mean loss, acc %)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total


# Recovery fine-tune (PruningBench)
def effective_recipe(args):
    """RECIPE_FT with any non-None CLI overrides applied (lr / weight_decay /
    scheduler / lr_decay_epochs). Mirrors the detection worker's recipe knobs."""
    eff = dict(RECIPE_FT)
    for k, v in {"lr": getattr(args, "lr", None),
                 "weight_decay": getattr(args, "weight_decay", None),
                 "scheduler": getattr(args, "scheduler", None),
                 "lr_decay_epochs": getattr(args, "lr_decay_epochs", None)}.items():
        if v is not None:
            eff[k] = v
    return eff


def finetune_pruningbench(model, train_loader, val_loader, device, total_epochs,
                          label, recipe=None):
    if total_epochs <= 0:
        return 0.0, []

    eff = recipe if recipe is not None else RECIPE_FT
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=eff["lr"],
                          momentum=eff["momentum"],
                          weight_decay=eff["weight_decay"],
                          nesterov=eff.get("nesterov", False))

    sched_name = eff.get("scheduler", "multistep")
    if sched_name == "cosine":
        # Anneals over whatever budget, which is what short per-step FT needs: MultiStep[30,60] never fires.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs), eta_min=0.0)
        sched_desc = f"cosine T_max={total_epochs}"
    else:
        ms = [m for m in eff["lr_decay_epochs"] if m < total_epochs]
        if not ms:
            ms = [max(1, total_epochs - 1)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms,
                                                   gamma=eff["lr_decay_rate"])
        sched_desc = f"MultiStep {ms} γ={eff['lr_decay_rate']}"

    print(f"  [{label}] {total_epochs} ep, SGD lr={eff['lr']:g} m={eff['momentum']} "
          f"wd={eff['weight_decay']:g} nesterov={eff.get('nesterov', False)}, {sched_desc}",
          flush=True)

    best_acc, best_state, best_epoch = 0.0, None, -1
    history = []
    for epoch in range(total_epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        improved = val_acc > best_acc
        if improved:
            best_acc, best_epoch = val_acc, epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "train_acc": train_acc, "val_acc": val_acc,
                        "lr": optimizer.param_groups[0]["lr"], "duration_s": dt})

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == total_epochs - 1 or improved:
            star = " *" if improved else ""
            print(f"    Ep {epoch+1:3d}/{total_epochs}  loss={train_loss:.4f}  "
                  f"train={train_acc:5.2f}%  val={val_acc:5.2f}%  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  ({dt:.0f}s){star}",
                  flush=True)

    if best_state:
        model.load_state_dict(best_state); model.to(device)
    print(f"  [{label}] Best val : {best_acc:.2f}% @ ep {best_epoch}/{len(history)}", flush=True)
    return best_acc, history


# Pruning utilities






def get_ignored_layers(model):
    """MobileNetV2 (torchvision): classifier = Sequential(Dropout, Linear(1280, N)).

    Protect every Linear / Conv2d found in `model.classifier`. Fallback:
    if the model has no `classifier` attribute, take the last Linear/Conv2d
    encountered during traversal.
    """
    ignored = []
    if hasattr(model, "classifier"):
        cls = model.classifier
        if isinstance(cls, nn.Sequential):
            for layer in cls:
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    ignored.append(layer)
        elif isinstance(cls, (nn.Linear, nn.Conv2d)):
            ignored.append(cls)
    if not ignored:
        last = None
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                last = m
        if last is not None:
            ignored.append(last)
    return ignored


# Progressive pruning (data-free only)


# Single run: (importance, checkpoint_pct)
def run_one(imp_name, checkpoint_pct, args, device,
            models_dir, pruned_dir, log_dir):
    """Run for a triplet (fixed baseline, importance, checkpoint_pct)."""
    import torch_pruning as tp

    seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    mode_sfx = mode_suffix(args.prune_mode)   # "" here -- see ladder.mode_suffix
    out_name = (f"{args.baseline_name}_pruned{checkpoint_pct}pct_"
                f"{imp_name}{mode_sfx}{seed_suffix}")
    out_path = pruned_dir / f"{out_name}.pt"
    log_path = (log_dir / f"{args.baseline_name}_{imp_name}_"
                          f"{checkpoint_pct}pct{mode_sfx}{seed_suffix}.json")

    if out_path.exists() and not args.force:
        print(f"  [{out_name}] Already present, skip.")
        return None

    print(f"\n{'━' * 70}")
    print(f"  {args.baseline_name.upper()} — {imp_name.upper()} — "
          f"{checkpoint_pct}%  (seed={args.seed})")
    print(f"{'━' * 70}", flush=True)

    baseline_path = args.baseline_path or (models_dir / f"{args.baseline_name}.pt")
    seed_everything(args.seed)
    model = torch.load(str(baseline_path), map_location="cpu", weights_only=False)
    model = model.to(device)

    train_loader, val_loader, test_loader = get_dataloaders(args, image_size=args.image_size)
    ref_acc = evaluate(model, val_loader, device)
    pre_params = count_params(model)
    pre_macs = count_macs(model, args.image_size)
    print(f"  Pre-pruning: {ref_acc:.2f}% acc, {pre_params:,} params, "
          f"{pre_macs:,.0f} MACs (loaded from {baseline_path.name})", flush=True)

    importance = get_importance(imp_name)
    ignored_layers = get_ignored_layers(model)
    example_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)

    pruner = tp.pruner.MetaPruner(
        model=model, example_inputs=example_input,
        importance=importance,
        pruning_ratio=1.0,
        iterative_steps=PRUNING_PROTOCOL["iterative_steps"],
        global_pruning=PRUNING_PROTOCOL["global_pruning"],
        max_pruning_ratio=PRUNING_PROTOCOL["max_pruning_ratio"],
        ignored_layers=ignored_layers,
    )

    print(f"  [Pruning] target={checkpoint_pct}%, "
          f"iterative_steps={PRUNING_PROTOCOL['iterative_steps']}, "
          f"global={PRUNING_PROTOCOL['global_pruning']}, "
          f"max_per_layer={PRUNING_PROTOCOL['max_pruning_ratio']}, "
          f"ignored={[type(m).__name__ for m in ignored_layers]}",
          flush=True)
    t_prune_start = time.time()
    n_steps, actual_pct, post_params = progressive_pruning_to_target(
        model, pruner, checkpoint_pct)
    t_prune = time.time() - t_prune_start
    post_macs = count_macs(model, args.image_size)
    val_acc_post = evaluate(model, val_loader, device)
    print(f"  [Pruning] {n_steps} steps in {t_prune:.0f}s -- {post_params:,} params "
          f"({actual_pct:.1f}% pruned), val={val_acc_post:.2f}%", flush=True)

    eff_recipe = effective_recipe(args)
    final_epochs = args.final_epochs if args.final_epochs is not None else eff_recipe["epochs"]
    t_ft_start = time.time()
    final_acc, ft_history = finetune_pruningbench(
        model, train_loader, val_loader, device, final_epochs,
        label=f"Final FT @{checkpoint_pct}%", recipe=eff_recipe)
    t_ft = time.time() - t_ft_start

    torch.save(model, str(out_path))

    eval_split = "test" if args.dataset == "flowers102" else "val"
    print(f"  [FP32 test] Bootstrap CI95 on {eval_split} set ({args.dataset})...",
          flush=True)
    fp32_acc = evaluate_with_topk_ci(model, test_loader, device)
    print(f"  [FP32 test] Top-1 = {fp32_acc['top1_pct']:.2f}% "
          f"[{fp32_acc['top1_ci95_lo']:.2f}, {fp32_acc['top1_ci95_hi']:.2f}]  "
          f"Top-5 = {fp32_acc['top5_pct']:.2f}%  (n={fp32_acc['n_eval']})",
          flush=True)

    layer_structure = extract_layer_structure(model)
    n_conv = sum(1 for x in layer_structure if x["kind"] == "Conv2d")
    n_lin = sum(1 for x in layer_structure if x["kind"] == "Linear")
    print(f"  [structure]  {n_conv} Conv2d + {n_lin} Linear layers post-pruning",
          flush=True)

    result = {
        "model": args.baseline_name, "importance": imp_name,
        "checkpoint_pct": checkpoint_pct,
        "prune_mode": args.prune_mode,
        "ref_acc": ref_acc, "post_prune_acc": val_acc_post, "final_acc": final_acc,
        "acc_delta": final_acc - ref_acc,
        "pre_params": pre_params, "post_params": post_params,
        "param_reduction_pct": 100 * (1 - post_params / pre_params),
        "actual_pct": actual_pct,
        "pre_macs": pre_macs, "post_macs": post_macs,
        "macs_reduction_pct": 100 * (1 - post_macs / pre_macs) if pre_macs > 0 else 0,
        "n_pruning_steps": n_steps,
        "duration_s": {"pruning": t_prune, "finetune": t_ft, "total": t_prune + t_ft},
        "output_file": str(out_path),
        "fp32_test_top1_pct":     fp32_acc["top1_pct"],
        "fp32_test_top5_pct":     fp32_acc["top5_pct"],
        "fp32_test_top1_ci95_lo": fp32_acc["top1_ci95_lo"],
        "fp32_test_top1_ci95_hi": fp32_acc["top1_ci95_hi"],
        "fp32_test_top5_ci95_lo": fp32_acc["top5_ci95_lo"],
        "fp32_test_top5_ci95_hi": fp32_acc["top5_ci95_hi"],
        "n_eval_test":            fp32_acc["n_eval"],
        "layer_structure_post":   layer_structure,
    }

    norm_mean, norm_std = norm_for_model(args.baseline_name)
    full_log = {
        **result,
        "dataset": args.dataset,
        "data_root": args.data_root,
        "inat_train_json": args.inat_train_json,
        "inat_val_json": args.inat_val_json,
        "inat_image_root": args.inat_image_root,
        "normalization": {"mean": norm_mean, "std": norm_std},
        "recipe_ft": eff_recipe,
        "pruning_protocol": PRUNING_PROTOCOL,
        "seed": args.seed, "device": str(device),
        "image_size": args.image_size,
        "num_classes": args.num_classes,
        "ft_history": ft_history,
    }
    with open(log_path, "w") as f:
        json.dump(full_log, f, indent=2, default=str)

    print(f"  ╔══ DONE ══")
    print(f"  ║ Accuracy  : {ref_acc:.2f}% → {final_acc:.2f}% (Δ={final_acc - ref_acc:+.2f})")
    print(f"  ║ Params    : {pre_params:,} → {post_params:,} "
          f"({result['param_reduction_pct']:.1f}% reduced)")
    if pre_macs > 0:
        print(f"  ║ MACs      : {pre_macs:,.0f} → {post_macs:,.0f} "
              f"({result['macs_reduction_pct']:.1f}% reduced)")
    print(f"  ║ Duration  : pruning={t_prune:.0f}s, FT={t_ft:.0f}s")
    print(f"  ║ → {out_path.name} ({out_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"  ║ → {log_path.name}")
    print(f"  ╚{'═' * 50}", flush=True)

    return result


def _build_cls_log(result, args, device, recipe):
    """Assemble the full per-checkpoint JSON log (shared by both prune modes)."""
    norm_mean, norm_std = norm_for_model(args.baseline_name)
    return {
        **result,
        "dataset": args.dataset,
        "data_root": args.data_root,
        "inat_train_json": args.inat_train_json,
        "inat_val_json": args.inat_val_json,
        "inat_image_root": args.inat_image_root,
        "normalization": {"mean": norm_mean, "std": norm_std},
        "recipe_ft": recipe,
        "pruning_protocol": PRUNING_PROTOCOL,
        "seed": args.seed, "device": str(device),
        "image_size": args.image_size,
        "num_classes": args.num_classes,
    }


# Iterative run: ONE trajectory -> all checkpoints (LTH-style IMP)
def run_iterative(imp_name, checkpoints, args, device,
                  models_dir, pruned_dir, log_dir):
    """LTH-style iterative magnitude pruning (Han 2015; Frankle & Carbin 2019).

    A SINGLE continuous trajectory: prune to the smallest ratio, fine-tune,
    save a checkpoint, then keep pruning the (recovered) model to the next
    ratio, fine-tune, save, ... Each ratio is warm-started from the previous
    recovered model. The literature (Frankle & Carbin, ICLR 2019) reports that
    recovering high ratios far better than one-shot pruning the dense baseline to
    each ratio; NOT MEASURED HERE -- every ladder under results/ is independent
    mode, so on these models it is an expectation, not a result.

    NOTE: this is NOT the PruningBench protocol (which prunes the dense baseline
    to each target independently). Cite it as iterative magnitude pruning.

    Resumable: leading checkpoints whose .pt already exists are skipped and the
    last present one is reloaded as the starting model, so a requeued / split
    job continues the trajectory instead of restarting it.
    """
    import torch_pruning as tp
    checkpoints = sorted(checkpoints)
    seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"

    mode_sfx = mode_suffix(args.prune_mode)   # "_iter" here -- see ladder.mode_suffix

    def out_path_for(cp):
        return pruned_dir / (f"{args.baseline_name}_pruned{cp}pct_"
                             f"{imp_name}{mode_sfx}{seed_suffix}.pt")

    def log_path_for(cp):
        return log_dir / (f"{args.baseline_name}_{imp_name}_"
                          f"{cp}pct{mode_sfx}{seed_suffix}.json")

    print(f"\n{'━' * 70}")
    print(f"  ITERATIVE  {args.baseline_name.upper()} — {imp_name.upper()} — "
          f"{checkpoints}%  (seed={args.seed})")
    print(f"{'━' * 70}", flush=True)

    baseline_path = args.baseline_path or (models_dir / f"{args.baseline_name}.pt")
    seed_everything(args.seed)
    base_model = torch.load(str(baseline_path), map_location="cpu",
                            weights_only=False).to(device)
    train_loader, val_loader, test_loader = get_dataloaders(args, image_size=args.image_size)

    # Dense reference, and the param count the per-step ratio targets are measured against.
    ref_acc = evaluate(base_model, val_loader, device)
    pre_params = count_params(base_model)
    pre_macs = count_macs(base_model, args.image_size)
    print(f"  Dense baseline: {ref_acc:.2f}% acc, {pre_params:,} params, "
          f"{pre_macs:,.0f} MACs ({baseline_path.name})", flush=True)

    # Resume: how many leading checkpoints are already done? See prune.ladder.
    start_idx, resume_path = resume_point(checkpoints, out_path_for, force=args.force)
    if start_idx >= len(checkpoints):
        print(f"  All {len(checkpoints)} checkpoints already present, skip.")
        return []

    if resume_path is not None:
        print(f"  [resume] {start_idx} checkpoint(s) present; continuing from "
              f"{checkpoints[start_idx - 1]}% ({resume_path.name})", flush=True)
        model = torch.load(str(resume_path), map_location="cpu",
                           weights_only=False).to(device)
        del base_model
    else:
        model = base_model

    importance = get_importance(imp_name)
    ignored_layers = get_ignored_layers(model)
    example_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    pruner = tp.pruner.MetaPruner(
        model=model, example_inputs=example_input,
        importance=importance,
        pruning_ratio=1.0,
        iterative_steps=PRUNING_PROTOCOL["iterative_steps"],
        global_pruning=PRUNING_PROTOCOL["global_pruning"],
        max_pruning_ratio=PRUNING_PROTOCOL["max_pruning_ratio"],
        ignored_layers=ignored_layers,
    )

    eff_recipe = effective_recipe(args)
    final_epochs = args.final_epochs if args.final_epochs is not None else eff_recipe["epochs"]
    results = []
    for cp in checkpoints[start_idx:]:
        print(f"\n  ── iterative step → {cp}% ──", flush=True)
        t_prune0 = time.time()
        n_steps, actual_pct, post_params = progressive_pruning_to_target(
            model, pruner, cp, initial_params=pre_params)
        t_prune = time.time() - t_prune0
        post_macs = count_macs(model, args.image_size)
        post_prune_acc = evaluate(model, val_loader, device)
        print(f"  [Pruning] +{n_steps} steps in {t_prune:.0f}s -- {post_params:,} params "
              f"({actual_pct:.1f}% of dense pruned), val={post_prune_acc:.2f}%", flush=True)

        t_ft0 = time.time()
        final_acc, ft_history = finetune_pruningbench(
            model, train_loader, val_loader, device, final_epochs,
            label=f"Iter FT @{cp}%", recipe=eff_recipe)
        t_ft = time.time() - t_ft0

        torch.save(model, str(out_path_for(cp)))

        eval_split = "test" if args.dataset == "flowers102" else "val"
        print(f"  [FP32 {eval_split}] Bootstrap CI95 ({args.dataset})...", flush=True)
        fp32_acc = evaluate_with_topk_ci(model, test_loader, device)
        layer_structure = extract_layer_structure(model)

        result = {
            "model": args.baseline_name, "importance": imp_name,
            "checkpoint_pct": cp,
            "prune_mode": "iterative",
            "warm_start": (start_idx > 0) or (cp != checkpoints[start_idx]),
            "ref_acc": ref_acc, "post_prune_acc": post_prune_acc, "final_acc": final_acc,
            "acc_delta": final_acc - ref_acc,
            "pre_params": pre_params, "post_params": post_params,
            "param_reduction_pct": 100 * (1 - post_params / pre_params),
            "actual_pct": actual_pct,
            "pre_macs": pre_macs, "post_macs": post_macs,
            "macs_reduction_pct": 100 * (1 - post_macs / pre_macs) if pre_macs > 0 else 0,
            "n_pruning_steps": n_steps,
            "duration_s": {"pruning": t_prune, "finetune": t_ft, "total": t_prune + t_ft},
            "output_file": str(out_path_for(cp)),
            "fp32_test_top1_pct":     fp32_acc["top1_pct"],
            "fp32_test_top5_pct":     fp32_acc["top5_pct"],
            "fp32_test_top1_ci95_lo": fp32_acc["top1_ci95_lo"],
            "fp32_test_top1_ci95_hi": fp32_acc["top1_ci95_hi"],
            "fp32_test_top5_ci95_lo": fp32_acc["top5_ci95_lo"],
            "fp32_test_top5_ci95_hi": fp32_acc["top5_ci95_hi"],
            "n_eval_test":            fp32_acc["n_eval"],
            "layer_structure_post":   layer_structure,
        }
        full_log = {**_build_cls_log(result, args, device, eff_recipe), "ft_history": ft_history}
        with open(log_path_for(cp), "w") as f:
            json.dump(full_log, f, indent=2, default=str)

        print(f"  ║ @{cp}%  acc {ref_acc:.2f}→{final_acc:.2f} (Δ{final_acc-ref_acc:+.2f})  "
              f"params -{result['param_reduction_pct']:.1f}%  MACs -{result['macs_reduction_pct']:.1f}%  "
              f"→ {out_path_for(cp).name}", flush=True)
        results.append(result)

    return results


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Pruning MobileNet V2 (PruningBench) on Flowers-102 or iNat 2017 Plantae")

    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=DEFAULT_DATASET,
                        help=f"Classification dataset (default: {DEFAULT_DATASET}). "
                             f"`imagenet` reads $DATASET_DIR/Imagenet_1k (no finetune); "
                             f"`flowers102` is the fast bring-up; "
                             f"`inat_plant` requires iNat 2017 downloaded locally.")
    # Flowers-102 / ImageNet args (both use --data_root)
    parser.add_argument("--data_root", type=str, default=None,
                        help="(dataset=flowers102) root for the torchvision `flowers-102/` "
                             "cache; (dataset=imagenet) ImageNet-1k root holding train/ + val/ "
                             "ImageFolder subtrees")
    # iNat args (legacy)
    parser.add_argument("--inat_train_json", type=str, default=None,
                        help="(dataset=inat_plant) train2017.json annotation")
    parser.add_argument("--inat_val_json", type=str, default=None,
                        help="(dataset=inat_plant) val2017.json annotation")
    parser.add_argument("--inat_image_root", type=str, default=None,
                        help="(dataset=inat_plant) image root for the relative file_name "
                             "entries in the JSON")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Override on the number of classes. By default, derived from the dataset "
                             "(102 for flowers102, 2101 for inat_plant).")
    parser.add_argument("--baseline_name", type=str, default=None,
                        help="Prefix of the baseline in models/. By default derived from the dataset.")
    parser.add_argument("--baseline_path", type=Path, default=None,
                        help="Override the baseline .pt path. By default: "
                             "MODELS_DIR / f'{baseline_name}.pt'. Use this to decouple "
                             "the input file from the output prefix (e.g. read "
                             "mobilenetv2.pt while still emitting mobilenetv2_imagenet_pruned*.pt).")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--checkpoints", nargs="+", type=int, required=True,
                        help="Pruning levels in %% (e.g. 10 20 30 40 50 60 70 80 90). "
                             "Each level = 1 independent run from the baseline.")
    parser.add_argument("--importance", nargs="+", default=None,
                        choices=ALL_IMPORTANCES,
                        help=f"Subset of importances (default: all = {ALL_IMPORTANCES})")

    parser.add_argument("--prune_mode", choices=["independent", "iterative"],
                        default="independent",
                        help="independent (default): prune the dense baseline to each ratio "
                             "separately + FT (PruningBench one-shot-per-ratio). "
                             "iterative: ONE trajectory, prune->FT->save at each ratio in "
                             "ascending order (LTH-style IMP; warm-started, one run for all "
                             "ratios, resumable). Its effect vs independent is unmeasured "
                             "here -- every published ladder is independent mode.")
    parser.add_argument("--final_epochs", type=int, default=None,
                        help="Override the number of final FT epochs (default: 90, PruningBench ImageNet)")
    # Recovery-FT overrides, mirroring the detection worker's DET_LR/DET_WD/DET_SCHED/DET_MILESTONES.
    parser.add_argument("--lr", type=float, default=None,
                        help="Override FT learning rate (default: RECIPE_FT lr)")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Override FT weight decay (default: RECIPE_FT wd)")
    parser.add_argument("--scheduler", choices=["multistep", "cosine"], default=None,
                        help="FT LR schedule. 'cosine' anneals over the full budget "
                             "(use for short per-step FT in iterative mode); 'multistep' "
                             "uses --lr_decay_epochs. Default: RECIPE_FT (multistep).")
    parser.add_argument("--lr_decay_epochs", nargs="+", type=int, default=None,
                        help="MultiStep milestones (default: RECIPE_FT [30,60])")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if the .pt exists")
    args = parser.parse_args()

    # Defaults that depend on --dataset
    if args.num_classes is None:
        args.num_classes = DEFAULT_NUM_CLASSES[args.dataset]
    if args.baseline_name is None:
        args.baseline_name = DEFAULT_BASELINE_NAME[args.dataset]

    OUTPUTS_DIR = Path(__file__).parents[3] / "outputs"
    MODELS_DIR = OUTPUTS_DIR / "models"
    PRUNED_DIR = OUTPUTS_DIR / "pytorch_pruned"
    LOG_DIR = OUTPUTS_DIR / "pruning_logs"
    PRUNED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    importance_list = args.importance if args.importance else ALL_IMPORTANCES
    checkpoints = sorted(args.checkpoints)

    baseline_path = args.baseline_path or (MODELS_DIR / f"{args.baseline_name}.pt")
    if not baseline_path.exists():
        print(f"\n[ERROR] Baseline not found: {baseline_path}\n"
              f"Run families/classification/imagenet_backbones/finetune.py for --dataset {args.dataset} "
              f"to produce {args.baseline_name}.pt in {MODELS_DIR}/")
        return

    n_runs = len(importance_list) * len(checkpoints)

    dataset_label = {"imagenet": "ImageNet-1k",
                     "flowers102": "Oxford Flowers-102",
                     "inat_plant": "iNat 2017 Plantae"}[args.dataset]
    norm_mean, norm_std = norm_for_model(args.baseline_name)
    print("=" * 70)
    print(f"PRUNING {args.baseline_name} {dataset_label} (PruningBench)")
    print("=" * 70)
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Baseline     : {args.baseline_name} ({baseline_path.name})")
    print(f"  Image size   : {args.image_size}  (norm mean={norm_mean}, std={norm_std})")
    print(f"  Num classes  : {args.num_classes}")
    if args.dataset in ("flowers102", "imagenet"):
        print(f"  Data root    : {args.data_root}")
    else:
        print(f"  iNat train   : {args.inat_train_json}")
        print(f"  iNat val     : {args.inat_val_json}")
        print(f"  Image root   : {args.inat_image_root}")
    print(f"  Importances  : {importance_list}")
    print(f"  Levels       : {checkpoints}%")
    print(f"  Prune mode   : {args.prune_mode}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Seed         : {args.seed}")
    print(f"  Total runs   : {n_runs}  ({len(importance_list)} × {len(checkpoints)})")
    if args.final_epochs is not None:
        print(f"  ⚠ Override final_epochs={args.final_epochs}")
    print("=" * 70, flush=True)

    all_results = []
    t_start = time.time()

    for imp_name in importance_list:
        if args.prune_mode == "iterative":
            try:
                rs = run_iterative(imp_name, checkpoints, args, device,
                                   MODELS_DIR, PRUNED_DIR, LOG_DIR)
                all_results.extend(rs)
            except Exception as e:
                print(f"\n  [{imp_name} iterative] [FAILED] {type(e).__name__}: {e}",
                      flush=True)
                import traceback; traceback.print_exc()
            continue
        for cp in checkpoints:
            try:
                r = run_one(imp_name, cp, args, device,
                            MODELS_DIR, PRUNED_DIR, LOG_DIR)
                if r is not None:
                    all_results.append(r)
            except Exception as e:
                print(f"\n  [{imp_name} @{cp}%] [FAILED] {type(e).__name__}: {e}",
                      flush=True)
                import traceback; traceback.print_exc()

    total_min = (time.time() - t_start) / 60

    print(f"\n{'=' * 70}")
    print(f"SUMMARY ({total_min:.1f} min total, {len(all_results)} runs)")
    print(f"{'=' * 70}")
    for r in all_results:
        print(f"  [{r['importance']:<14s}] @{r['checkpoint_pct']:>2d}%  "
              f"{r['ref_acc']:.2f}% → {r['final_acc']:.2f}%  "
              f"({r['param_reduction_pct']:.1f}% params, "
              f"Δacc={r['acc_delta']:+.2f})")
    if all_results:
        summary_path = LOG_DIR / f"{args.baseline_name}_pruning_summary.json"
        existing = []
        if summary_path.exists():
            with open(summary_path) as f:
                existing = json.load(f)
        seen = {(r["model"], r["importance"], r["checkpoint_pct"]) for r in all_results}
        existing = [r for r in existing
                    if (r["model"], r["importance"], r["checkpoint_pct"]) not in seen]
        merged = existing + all_results
        with open(summary_path, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"\n→ {summary_path}  ({len(merged)} cumulative runs)")
    else:
        print("\nNo new runs computed.")


if __name__ == "__main__":
    main()
