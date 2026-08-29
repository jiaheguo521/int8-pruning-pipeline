#!/usr/bin/env python3
"""Builds the plant-classification baseline that imagenet_backbones/prune.py consumes.

Fine-tune `torchvision.models.mobilenet_v2` (ImageNet pretrained) on a
plant classification dataset, then save the result **as a complete
`nn.Module`** ready to be consumed by
`families/classification/imagenet_backbones/prune.py`.

Supported datasets (`--dataset`):
  - `flowers102` (default, fast bring-up): Oxford Flowers-102, 102 classes,
    ~330 MB, downloaded automatically by torchvision. Recommended as long
    as the iNat 2017 download (~185 GB) is not finalized.
  - `inat_plant`: iNaturalist 2017 Plantae subset, 2101 classes (~30-40 GB
    after filtering). Select explicitly once the dataset is available
    locally.

Pipeline:
  1. Loads MobileNetV2 + ImageNet weights from `--imagenet_weights`
     (produced by `scripts/download_and_finetune_models.sh`).
     Accepts both file formats (`full nn.Module` or state_dict).
  2. Replaces `model.classifier[-1]`: `Linear(1280, 1000)` -> `Linear(1280, num_classes)`.
  3. Optional `--head_only`: freezes `model.features` (BN included) and only
     trains the new Linear.
  4. SGD + CosineAnnealing, standard transfer-learning recipe.
  5. Saves the best (by val acc) as a complete `nn.Module` via
     `torch.save(model, out)`. No state_dict.

Output:
    outputs/models/<output_name>.pt
        default: mobilenetv2_flowers102.pt (Flowers-102) or mobilenetv2_inat.pt (iNat)
    outputs/models/<output_name>.finetune.json  (history + recipe)

Default recipe (transfer learning):
    SGD lr=0.01 momentum=0.9 wd=1e-4, CosineAnnealingLR(T_max=epochs),
    batch=64, image=224x224, 20 ep.

Usage -- Flowers-102 (default, ~30 min on A6000):
    python families/classification/imagenet_backbones/finetune.py \\
        --imagenet_weights /path/to/mobilenetv2_imagenet.pt \\
        --data_root /path/to/datasets/flowers102 \\
        --epochs 20

Usage -- iNat 2017 Plantae (~4-5 h on A6000):
    python families/classification/imagenet_backbones/finetune.py \\
        --dataset inat_plant \\
        --imagenet_weights /path/to/mobilenetv2_imagenet.pt \\
        --inat_train_json  /path/to/inat2017_plant/train2017.json \\
        --inat_val_json    /path/to/inat2017_plant/val2017.json \\
        --inat_image_root  /path/to/inat2017_plant \\
        --num_classes 2101 --epochs 20

    # Head-only (~3x faster, ~3-5 pts Top-1 below):
    python families/classification/imagenet_backbones/finetune.py ... --head_only --epochs 10

    # Smoke (1 ep, overwrites output):
    python families/classification/imagenet_backbones/finetune.py ... --epochs 1 --force
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
from torchvision.models import mobilenet_v2

# Local module (same directory), so the script runs without PYTHONPATH gymnastics.
sys.path.insert(0, str(Path(__file__).parent))
from int8_pruning.data.classification import (
    build_classification_dataloaders,
    DATASET_NUM_CLASSES,
)


# Only the two from-scratch-head datasets: ImageNet uses the torchvision pretrained head directly.
DATASET_CHOICES = ("flowers102", "inat_plant")
DEFAULT_DATASET = "flowers102"
DEFAULT_OUT_NAME = {
    "flowers102": "mobilenetv2_flowers102",
    "inat_plant": "mobilenetv2_inat",
}


# Reproducibility
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(args, image_size=224):
    """Dispatch on `args.dataset`. Returns (train_loader, val_loader, num_classes)."""
    train_loader, val_loader, _, num_classes = build_classification_dataloaders(
        args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
        image_size=image_size,
        inat_train_json=getattr(args, "inat_train_json", None),
        inat_val_json=getattr(args, "inat_val_json", None),
        inat_image_root=getattr(args, "inat_image_root", None),
    )
    return train_loader, val_loader, num_classes


# Model construction
def load_mobilenetv2_starting_point(weights_path, num_classes, device):
    """Builds MobileNetV2, loads ImageNet weights, replaces the head.

    `weights_path` can be:
      - (a) a complete `nn.Module` (case of `download_and_finetune_models.sh` after
        switching to `torch.save(m, out)`);
      - (b) a `state_dict` (legacy compat, old download format).
    """
    obj = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict):
        model = mobilenet_v2(weights=None)
        try:
            model.load_state_dict(obj)
        except RuntimeError:
            missing, unexpected = model.load_state_dict(obj, strict=False)
            print(f"  [load] state_dict mismatch (strict=False) — "
                  f"missing={len(missing)} unexpected={len(unexpected)}",
                  flush=True)
    else:
        raise RuntimeError(
            f"Unexpected format for {weights_path}: {type(obj).__name__}. "
            f"Expected: complete nn.Module or state_dict (dict)."
        )

    # MobileNetV2.classifier = Sequential(Dropout(0.2), Linear(1280, 1000)); swap the last Linear.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(device)


def freeze_features(model):
    """Freezes `model.features` (weights + BN running stats). Used by --head_only."""
    for p in model.features.parameters():
        p.requires_grad = False
    for m in model.features.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# Train / eval
def evaluate(model, val_loader, device):
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


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV2 (ImageNet) → Flowers-102 (default) or iNat 2017 Plantae")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=DEFAULT_DATASET,
                        help=f"Fine-tune dataset (default: {DEFAULT_DATASET}). "
                             f"`flowers102` is the fast bring-up; "
                             f"`inat_plant` requires the full iNat 2017 download.")
    parser.add_argument("--imagenet_weights", type=str, required=True,
                        help="Path to mobilenetv2_imagenet.pt "
                             "(produced by scripts/download_and_finetune_models.sh)")
    # Flowers-102 args
    parser.add_argument("--data_root", type=str, default=None,
                        help="(dataset=flowers102) root where torchvision downloads / reads "
                             "the `flowers-102/` cache")
    # iNat args (legacy, kept for the iNat path)
    parser.add_argument("--inat_train_json", type=str, default=None,
                        help="(dataset=inat_plant) path to train2017.json")
    parser.add_argument("--inat_val_json", type=str, default=None,
                        help="(dataset=inat_plant) path to val2017.json")
    parser.add_argument("--inat_image_root", type=str, default=None,
                        help="(dataset=inat_plant) image root")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Override on the number of classes. By default, derived from the dataset "
                             "(102 for Flowers-102, 2101 for iNat Plantae).")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--head_only", action="store_true",
                        help="Freezes the backbone and only trains the classifier head")
    parser.add_argument("--amp", action="store_true",
                        help="Mixed precision (torch.cuda.amp) -- ~1.5x speedup")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Name (without .pt) under models/. By default derived from the dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output file if it exists")
    args = parser.parse_args()

    # Defaults that depend on --dataset
    if args.num_classes is None:
        args.num_classes = DATASET_NUM_CLASSES[args.dataset]
    if args.output_name is None:
        args.output_name = DEFAULT_OUT_NAME[args.dataset]

    seed_everything(args.seed)

    MODELS_DIR = Path(__file__).parents[3] / "outputs" / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / f"{args.output_name}.pt"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} already exists (use --force to overwrite)")
        return

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_label = ("Oxford Flowers-102" if args.dataset == "flowers102"
                     else "iNaturalist 2017 Plantae")
    print("=" * 70)
    print(f"FINE-TUNE MobileNetV2 → {dataset_label}")
    print("=" * 70)
    print(f"  Device           : {device}")
    if device.type == "cuda":
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print(f"  Starting weights : {args.imagenet_weights}")
    print(f"  Dataset          : {args.dataset}")
    print(f"  Num classes      : {args.num_classes}")
    if args.dataset == "flowers102":
        print(f"  Data root        : {args.data_root}")
    else:
        print(f"  iNat train JSON  : {args.inat_train_json}")
        print(f"  iNat val JSON    : {args.inat_val_json}")
        print(f"  Image root       : {args.inat_image_root}")
    print(f"  Image size       : {args.image_size}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  LR / m / wd      : {args.lr} / {args.momentum} / {args.weight_decay}")
    print(f"  Schedule         : CosineAnnealingLR (T_max={args.epochs})")
    print(f"  Head-only        : {args.head_only}")
    print(f"  AMP              : {args.amp}")
    print(f"  Seed             : {args.seed}")
    print(f"  Output           : {out_path}")
    print("=" * 70, flush=True)

    train_loader, val_loader, n_data_classes = get_dataloaders(
        args, image_size=args.image_size)
    if n_data_classes != args.num_classes:
        print(f"  ⚠ [WARN] dataset reports {n_data_classes} classes "
              f"but --num_classes={args.num_classes}. Classifier head built "
              f"with {args.num_classes}; mismatch may break training if dataset "
              f"labels exceed {args.num_classes-1}.", flush=True)

    model = load_mobilenetv2_starting_point(args.imagenet_weights,
                                            args.num_classes, device)
    if args.head_only:
        freeze_features(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  [head_only] backbone frozen, "
              f"{sum(p.numel() for p in trainable):,} trainable params",
              flush=True)
    else:
        trainable = list(model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(trainable, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = (torch.cuda.amp.GradScaler()
              if (args.amp and device.type == "cuda") else None)

    best_acc, best_state, best_epoch = 0.0, None, -1
    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device, scaler)
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
        star = " *" if improved else ""
        print(f"  Ep {epoch+1:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"train={train_acc:5.2f}%  val={val_acc:5.2f}%  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  ({dt:.0f}s){star}",
              flush=True)

    if best_state is not None:
        model.load_state_dict(best_state); model.to(device)
    print(f"\n  Best val : {best_acc:.2f}% @ ep {best_epoch}/{args.epochs}",
          flush=True)

    # Save: FULL nn.Module (not state_dict) -- contract from families/classification/imagenet_backbones/prune.py
    torch.save(model.cpu(), str(out_path))
    print(f"  → {out_path} "
          f"({out_path.stat().st_size/1024/1024:.1f} MB, full nn.Module)",
          flush=True)

    log_path = out_path.with_suffix(".finetune.json")
    log = {
        "dataset":          args.dataset,
        "data_root":        args.data_root,
        "inat_train_json":  args.inat_train_json,
        "inat_val_json":    args.inat_val_json,
        "inat_image_root":  args.inat_image_root,
        "imagenet_weights": str(args.imagenet_weights),
        "num_classes":      args.num_classes,
        "image_size":       args.image_size,
        "batch_size":       args.batch_size,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "momentum":         args.momentum,
        "weight_decay":     args.weight_decay,
        "head_only":        args.head_only,
        "amp":              args.amp,
        "seed":             args.seed,
        "best_val_acc":     best_acc,
        "best_epoch":       best_epoch,
        "history":          history,
        "total_minutes":    (time.time() - t_start) / 60.0,
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"  → {log_path}", flush=True)


if __name__ == "__main__":
    main()
