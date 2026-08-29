#!/usr/bin/env python3
"""Structured pruning of the SSDLite320-MobileNetV3-Large backbone on COCO 2017.

Structured pruning of the **MobileNetV3-Large feature extractor of
SSDLite320-MobileNetV3-Large** (torchvision) on **COCO 2017**, following the
**PruningBench** protocol (Li et al. 2024). Second detection family next to
`families/detection/effdet/prune.py` (EfficientDet-Lite1 / effdet); the two workers
share constants + utilities but stay independently runnable.

Strategy: **backbone-only pruning**, same as the effdet worker.
  - The torchvision SSD forward is not a plain tensor graph (it runs
    GeneralizedRCNNTransform on a list of images, and postprocessing in eval
    mode), so the pruner is built over `export.SSDLitePerLevelHeads`
    — feature extractor + per-level head convs. The wrapper shares module
    objects with the SSD, so pruning through it mutates the SSD in place.
  - `ignored_layers` = both head conv module_lists (+ the stem conv).
    torch_pruning then freezes every backbone channel feeding a head and
    prunes only the convs internal to the feature extractor (MobileNetV3
    trunk AND the extra blocks are both prunable).
  - `--checkpoints` expresses the % of params pruned in the **feature
    extractor only** (`model.backbone`), like the effdet worker.

Recovery FT: torchvision-native — `model(images, targets)` in train mode
returns {'bbox_regression', 'classification'}; we backprop on their sum.
Same recipe defaults as the effdet worker (SGD lr=1e-3 m=0.9 wd=4e-5,
cosine + 1-epoch warmup, 100 ep, batch=16), kept as an own copy so the two
families can diverge later.

⚠️  batch_size must be >= 2 with drop_last: the SSD extra blocks BatchNorm
over [1, C, 1, 1] crashes on a trailing batch of 1.

Baseline contract:
    outputs/models/<baseline_name>.pt = full torchvision SSD nn.Module
    (saved by scripts/download_and_finetune_models.sh, COCO_V1 weights,
    91 classes with background=0, reduce_tail architecture).

Usage:
    python families/detection/ssdlite/prune.py \\
        --coco_root <path>/coco2017 \\
        --baseline_name ssdlite_mobilenetv3 \\
        --num_classes 91 --image_size 320 --batch_size 16 \\
        --checkpoints 10 20 30 40 50 60 70 80 90 \\
        --importance magnitude_l1 magnitude_l2 fpgm random lamp

    # Smoke (2 ep FT, leaky val2017 train subset):
    python families/detection/ssdlite/prune.py \\
        --coco_root <path>/coco2017 --coco_subset val2017 \\
        --checkpoints 30 --importance magnitude_l2 \\
        --final_epochs 2 --eval_every 1 --force
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent))
# Shared with the effdet worker: same subsets, importances, protocol and FT scheduler.
from int8_pruning.data.coco import COCO_TRAIN_SUBSETS
from int8_pruning.prune.core import (
    ALL_IMPORTANCES, count_macs, count_params, extract_layer_structure,
    get_importance, get_num_workers, progressive_pruning_to_target, seed_everything,
)
from int8_pruning.prune.detection import build_ft_scheduler

from export import SSDLitePerLevelHeads

# Own copy of values identical to effdet's today, so the two detection families can
# diverge and global_pruning=True stays an explicit choice, not an inherited default.
PRUNING_PROTOCOL = {
    "iterative_steps": 400,
    "global_pruning": True,
    "max_pruning_ratio": 0.9,
    "ref": "PruningBench progressive_pruning + 10% protection (backbone-only)",
}

IMAGE_SIZE_DEFAULT = 320   # SSDLite320 input (fixed by the internal transform)
NUM_CLASSES_DEFAULT = 91   # torchvision COCO-91 head, background = 0
BASELINE_NAME_DEFAULT = "ssdlite_mobilenetv3"


# Recovery FT recipe, same values as the effdet worker's RECIPE_FT and for the same
# reason (FT of a pretrained detector at bs=16). torchvision's own SSDLite recipe
# (SGD lr=0.15 @ bs=24x8, cosine, 660 ep) trains from scratch and is far too hot here.
RECIPE_FT = {
    "optimizer": "sgd", "momentum": 0.9, "weight_decay": 4e-5,
    "lr": 1e-3,
    "scheduler": "cosine",                # 'cosine' or 'multistep'
    "warmup_epochs": 1,                   # 0 to disable
    "warmup_start_factor": 0.01,          # lr starts at lr * this factor
    "lr_decay_epochs": [60, 80],          # used only when scheduler='multistep'
    "lr_decay_rate": 0.1,                 # used only when scheduler='multistep'
    "epochs": 100,
    "batch_size": 16,
    "warning": ("Recovery FT recipe mirrored from the effdet worker. "
                "train_loss is monitored; warning emitted if >50 in the first 5 epochs."),
    "ref": "EfficientDet FT adaptation (see RECIPE_FT in families/detection/effdet/prune.py) "
           "+ PruningBench (Li et al. 2024) protocol",
}


# SSD model — load / validate / wrap
def load_baseline_ssd(baseline_path):
    """Loads the `.pt` baseline and returns the full torchvision SSD module.

    Checks the attributes the rest of the worker relies on: `backbone`
    (feature extractor), `head` (classification_head + regression_head
    module_lists), `anchor_generator`, `transform`.
    """
    obj = torch.load(str(baseline_path), map_location="cpu", weights_only=False)
    for attr in ("backbone", "head", "anchor_generator", "transform"):
        if not hasattr(obj, attr):
            raise RuntimeError(
                f"Baseline {baseline_path}: loaded object lacks the `{attr}` "
                f"attribute expected on a torchvision SSD. Type={type(obj).__name__}"
            )
    return obj


# COCO data loading (torchvision-style, via pycocotools)
class CocoDetectionSSD(torch.utils.data.Dataset):
    """COCO -> (image float CHW [0,1] at original size, target dict).

    torchvision SSD does its own resize-to-320 + 0.5/0.5 normalization
    (GeneralizedRCNNTransform) on BOTH images and boxes, so the dataset
    stays raw-size and unnormalized.

    train=True:
      - drops iscrowd annotations and degenerate boxes (w<=0 or h<=0 —
        torchvision raises on x2<=x1 during loss computation),
      - drops images left with 0 valid boxes (empty gt breaks the matcher),
      - random horizontal flip p=0.5 (the only augmentation, mirroring the
        effdet loader's RandomFlip).
    train=False:
      - keeps every image, no augmentation; target only carries image_id
        (mAP eval reads the gt from the COCO object, not from targets).

    target: {"boxes": FloatTensor[n,4] xyxy abs, "labels": int64 native COCO
    category ids (1..90 — the COCO_V1 head is indexed by category id with
    background=0), "image_id": int}.
    """

    def __init__(self, img_dir, ann_file, train):
        from pycocotools.coco import COCO

        self.coco = COCO(str(ann_file))
        self.img_dir = Path(img_dir)
        self.train = train

        ids = sorted(self.coco.imgs.keys())
        if train:
            keep = []
            for img_id in ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
                if any(a["bbox"][2] > 0 and a["bbox"][3] > 0
                       for a in self.coco.loadAnns(ann_ids)):
                    keep.append(img_id)
            self.ids = keep
        else:
            self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        from torchvision.transforms.functional import to_tensor

        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        img_t = to_tensor(img)

        if not self.train:
            return img_t, {"image_id": img_id}

        boxes, labels = [], []
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        for a in self.coco.loadAnns(ann_ids):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if torch.rand(1).item() < 0.5:
            img_t = torch.flip(img_t, dims=[2])
            width = img_t.shape[2]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        return img_t, {"boxes": boxes, "labels": labels, "image_id": img_id}


def collate_detection(batch):
    """list of (img, target) -> (tuple of imgs, tuple of targets); torchvision
    detection models consume lists of variable-size images."""
    return tuple(zip(*batch))


def get_dataloaders_coco_ssd(coco_root, batch_size, seed,
                             train_subset="train2017"):
    """COCO train/val DataLoaders in torchvision detection format.

    `train_subset` semantics are identical to the effdet worker (see
    COCO_TRAIN_SUBSETS): train2017 / minitrain / val2017 (leaky smoke).
    Val loader always reads standard val2017 so mAP stays comparable.

    drop_last=True on train is REQUIRED: a trailing batch of 1 crashes the
    BatchNorm over [1, C, 1, 1] activations in the SSD extra blocks.

    Returns (train_loader, val_loader, coco_gt) — coco_gt is the pycocotools
    COCO object of instances_val2017.json, consumed by evaluate_coco_map_ssd.
    """
    if train_subset not in COCO_TRAIN_SUBSETS:
        raise ValueError(
            f"Unknown COCO train_subset='{train_subset}'. "
            f"Valid choices: {list(COCO_TRAIN_SUBSETS)}"
        )
    coco_root = Path(coco_root)
    train_ann, train_img = COCO_TRAIN_SUBSETS[train_subset]
    val_ann, val_img = COCO_TRAIN_SUBSETS["val2017"]
    for relpath in (train_ann, val_ann):
        if not (coco_root / relpath).exists():
            raise FileNotFoundError(
                f"COCO annotation file not found: {coco_root / relpath}\n"
                f"For 'minitrain', download via scripts/download_datasets.sh "
                f"(it fetches instances_minitrain2017.json into annotations/)."
            )

    train_ds = CocoDetectionSSD(coco_root / train_img, coco_root / train_ann, train=True)
    val_ds = CocoDetectionSSD(coco_root / val_img, coco_root / val_ann, train=False)

    nw = get_num_workers()
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=nw, pin_memory=True, collate_fn=collate_detection,
        generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, collate_fn=collate_detection)
    print(f"  [Dataset COCO] subset={train_subset}  "
          f"train: {len(train_ds)} images, val: {len(val_ds)} images, "
          f"batch={batch_size}, workers={nw}", flush=True)
    return train_loader, val_loader, val_ds.coco


# Training (torchvision-native detection loss)
def train_epoch_det_ssd(model, train_loader, optimizer, device, desc="train"):
    """One detection FT epoch. Returns (mean loss, max loss seen).

    Train-mode `model(images, targets)` returns
    {'bbox_regression': smooth-L1, 'classification': hard-negative-mined CE};
    we backprop on their sum.
    """
    model.train()
    running_loss, max_loss, n_batches = 0.0, 0.0, 0
    pbar = tqdm(train_loader, desc=desc, leave=False, dynamic_ncols=True,
                mininterval=1.0)
    for images, targets in pbar:
        images = [im.to(device, non_blocking=True) for im in images]
        targets = [{k: (v.to(device, non_blocking=True)
                        if isinstance(v, torch.Tensor) else v)
                    for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        running_loss += loss_val
        max_loss = max(max_loss, loss_val)
        n_batches += 1
        pbar.set_postfix(loss=f"{running_loss/n_batches:.3f}",
                         mx=f"{max_loss:.1f}")
    avg = running_loss / max(1, n_batches)
    return avg, max_loss


# Evaluation: COCO mAP via pycocotools
@torch.no_grad()
def evaluate_coco_map_ssd(model, val_loader, coco_gt, device):
    """Runs the full SSD in eval mode (its internal transform resizes to 320
    and postprocess rescales boxes back to original pixels), collects
    detections in COCO submission format, and computes mAP via pycocotools.

    torchvision `labels` ARE native COCO category ids (the COCO_V1 head is
    91-way with background=0 and postprocess_detections drops column 0), so
    `category_id = int(label)` with no remapping — unlike effdet's +1 story.
    """
    from pycocotools.cocoeval import COCOeval

    model.eval()
    detections = []
    for images, targets in tqdm(val_loader, desc="eval", leave=False,
                                dynamic_ncols=True, mininterval=1.0):
        images = [im.to(device, non_blocking=True) for im in images]
        outputs = model(images)
        for t, out in zip(targets, outputs):
            iid = int(t["image_id"])
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))
                if w <= 0 or h <= 0 or score <= 0:
                    continue
                detections.append({
                    "image_id": iid,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(score),
                })

    if not detections:
        print("  [COCO eval] No detection returned -- mAP=0", flush=True)
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0,
                "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0,
                "n_det": 0}

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    return {
        "mAP":        float(coco_eval.stats[0]),
        "mAP_50":     float(coco_eval.stats[1]),
        "mAP_75":     float(coco_eval.stats[2]),
        "mAP_small":  float(coco_eval.stats[3]),
        "mAP_medium": float(coco_eval.stats[4]),
        "mAP_large":  float(coco_eval.stats[5]),
        "n_det":      len(detections),
    }


# Pruning utilities
def count_macs_wrapper(wrapper, input_size, device):
    """MACs over the traceable wrapper (feature extractor + head convs) at
    `input_size`. NOTE: unlike the effdet worker (full model incl. BiFPN),
    this excludes nothing else SSD has — the wrapper IS the whole compute
    graph minus pre/postprocessing — but the JSON key spells out the scope.

    Delegates to the shared counter; the previous local copy had a bare
    `except: return 0` that swallowed the reason for a zero.
    """
    return count_macs(wrapper, input_size)


def get_ignored_layers_ssd(wrapper):
    """Backbone-only pruning: ignore both per-level head conv stacks (their
    interface channels then freeze automatically through the dependency
    graph) and the stem conv (protects in_channels=3, mirroring the effdet
    worker's conv_stem protection)."""
    ignored = [wrapper.cls_heads, wrapper.box_heads]
    first_conv = next(m for m in wrapper.backbone.modules()
                      if isinstance(m, nn.Conv2d))
    ignored.append(first_conv)
    return ignored


# Recovery FT (torchvision SSD, PruningBench recipe)
def finetune_pruningbench_ssd(model, train_loader, val_loader, coco_gt, device,
                              total_epochs, eval_every, label, recipe=None):
    """Detection FT mirroring `finetune_pruningbench_det` (effdet worker):
    same monitor (train_loss every epoch, WARN if max_loss > 50 in the first
    5 epochs, mAP every `eval_every` + last epoch, best-mAP state restored).
    """
    if total_epochs <= 0:
        return 0.0, []

    eff_recipe = dict(RECIPE_FT)
    if recipe:
        eff_recipe.update(recipe)

    optimizer = optim.SGD(model.parameters(),
                          lr=eff_recipe["lr"],
                          momentum=eff_recipe["momentum"],
                          weight_decay=eff_recipe["weight_decay"])
    scheduler, sched_desc = build_ft_scheduler(optimizer, total_epochs, eff_recipe)

    print(f"  [{label}] {total_epochs} ep, SGD lr={eff_recipe['lr']:g} "
          f"m={eff_recipe['momentum']} wd={eff_recipe['weight_decay']:g}, "
          f"{sched_desc}, eval_every={eval_every} ep", flush=True)

    best_map, best_state, best_epoch = -1.0, None, -1
    history = []

    for epoch in range(total_epochs):
        t0 = time.time()
        train_loss, max_loss = train_epoch_det_ssd(
            model, train_loader, optimizer, device,
            desc=f"{label} ep{epoch+1}/{total_epochs}",
        )
        scheduler.step()
        dt_train = time.time() - t0

        if epoch < 5 and max_loss > 50.0:
            print(f"    ⚠ [WARN] epoch {epoch+1}: max batch loss={max_loss:.1f} > 50 -- "
                  f"FT looks unstable. Current lr={eff_recipe['lr']:g}; "
                  f"try a smaller --lr (e.g. 5e-4) or longer --warmup_epochs.",
                  flush=True)

        do_eval = ((epoch + 1) % eval_every == 0) or (epoch == total_epochs - 1)
        eval_dict = None
        if do_eval:
            t_eval = time.time()
            eval_dict = evaluate_coco_map_ssd(model, val_loader, coco_gt, device)
            dt_eval = time.time() - t_eval
            improved = eval_dict["mAP"] > best_map
            if improved:
                best_map, best_epoch = eval_dict["mAP"], epoch + 1
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            star = " *" if improved else ""
            print(f"    Ep {epoch+1:3d}/{total_epochs}  loss={train_loss:.4f}  "
                  f"max_loss={max_loss:.2f}  mAP={eval_dict['mAP']:.4f}  "
                  f"mAP50={eval_dict['mAP_50']:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"({dt_train:.0f}s + eval {dt_eval:.0f}s){star}", flush=True)
        else:
            print(f"    Ep {epoch+1:3d}/{total_epochs}  loss={train_loss:.4f}  "
                  f"max_loss={max_loss:.2f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  ({dt_train:.0f}s)",
                  flush=True)

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "max_loss": max_loss,
            "lr": optimizer.param_groups[0]["lr"], "duration_s": dt_train,
            "eval": eval_dict,
        })

    if best_state is not None:
        model.load_state_dict(best_state); model.to(device)
    print(f"  [{label}] Best mAP : {best_map:.4f} @ ep {best_epoch}/{len(history)}",
          flush=True)
    return best_map, history


# Single run: (importance, checkpoint_pct)
def run_one(imp_name, checkpoint_pct, args, device,
            models_dir, pruned_dir, log_dir):
    """Run for a triplet (fixed SSDLite baseline, importance, %)."""
    import torch_pruning as tp

    seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    ds_tag = f"coco-{args.coco_subset}"
    out_name = f"{args.baseline_name}_{ds_tag}_pruned{checkpoint_pct}pct_{imp_name}{seed_suffix}"
    out_path = pruned_dir / f"{out_name}.pt"
    log_path = log_dir / f"{args.baseline_name}_{ds_tag}_{imp_name}_{checkpoint_pct}pct{seed_suffix}.json"

    if out_path.exists() and not args.force:
        print(f"  [{out_name}] Already present, skip.")
        return None

    print(f"\n{'━' * 70}")
    print(f"  {args.baseline_name.upper()} — {imp_name.upper()} — "
          f"{checkpoint_pct}% (backbone)  (seed={args.seed})")
    print(f"{'━' * 70}", flush=True)

    baseline_path = args.baseline_path or (models_dir / f"{args.baseline_name}.pt")
    seed_everything(args.seed)
    model = load_baseline_ssd(baseline_path)
    model = model.to(device)

    num_columns = getattr(model.head.classification_head, "num_columns", None)
    if num_columns is not None and num_columns != args.num_classes:
        print(f"  ⚠ [WARN] --num_classes={args.num_classes} but the loaded head "
              f"has {num_columns} columns; the checkpoint wins.", flush=True)

    # Data
    train_loader, val_loader, coco_gt = get_dataloaders_coco_ssd(
        args.coco_root, args.batch_size, args.seed,
        train_subset=args.coco_subset)

    # Initial eval (mAP on baseline)
    print(f"  [Pre-pruning eval] mAP on COCO val...", flush=True)
    t0 = time.time()
    ref_eval = evaluate_coco_map_ssd(model, val_loader, coco_gt, device)
    print(f"  Pre-pruning: mAP={ref_eval['mAP']:.4f} mAP50={ref_eval['mAP_50']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # The wrapper shares submodules with `model`, so pruning it prunes the SSD; eval() keeps BN stats put.
    wrapper = SSDLitePerLevelHeads(model).eval()

    pre_params_full = count_params(model)
    pre_params_backbone = count_params(model.backbone)
    pre_macs = count_macs_wrapper(wrapper, args.image_size, device)
    print(f"  Pre-pruning: {pre_params_full:,} params total, "
          f"{pre_params_backbone:,} params backbone, {pre_macs:,.0f} MACs "
          f"(loaded from {baseline_path.name})", flush=True)

    importance = get_importance(imp_name)
    ignored_layers = get_ignored_layers_ssd(wrapper)
    example_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)

    pruner = tp.pruner.MetaPruner(
        model=wrapper, example_inputs=example_input,
        importance=importance,
        pruning_ratio=1.0,
        iterative_steps=PRUNING_PROTOCOL["iterative_steps"],
        global_pruning=PRUNING_PROTOCOL["global_pruning"],
        max_pruning_ratio=PRUNING_PROTOCOL["max_pruning_ratio"],
        ignored_layers=ignored_layers,
    )

    print(f"  [Pruning] target={checkpoint_pct}% (backbone), "
          f"iterative_steps={PRUNING_PROTOCOL['iterative_steps']}, "
          f"global={PRUNING_PROTOCOL['global_pruning']}, "
          f"max_per_layer={PRUNING_PROTOCOL['max_pruning_ratio']}, "
          f"ignored={[type(m).__name__ for m in ignored_layers]}", flush=True)
    t_prune_start = time.time()
    # progressive_pruning_backbone reads count_params(model.backbone), and the SSD's `backbone` is the
    # wrapper's, so passing `model` works. scope=backbone is the old function's semantics.
    n_steps, actual_pct, post_params_backbone = progressive_pruning_to_target(
        model, pruner, checkpoint_pct, scope=lambda m: m.backbone)
    t_prune = time.time() - t_prune_start
    post_params_full = count_params(model)
    post_macs = count_macs_wrapper(wrapper, args.image_size, device)

    # Post-pruning eval (before FT)
    print(f"  [Post-pruning eval] mAP before FT...", flush=True)
    post_prune_eval = evaluate_coco_map_ssd(model, val_loader, coco_gt, device)
    print(f"  [Pruning] {n_steps} steps in {t_prune:.0f}s -- backbone "
          f"{pre_params_backbone:,} → {post_params_backbone:,} "
          f"({actual_pct:.1f}% pruned), mAP={post_prune_eval['mAP']:.4f}",
          flush=True)

    final_epochs = args.final_epochs if args.final_epochs is not None else RECIPE_FT["epochs"]
    cli_recipe = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "lr_decay_epochs": args.lr_decay_epochs,
    }
    t_ft_start = time.time()
    final_map, ft_history = finetune_pruningbench_ssd(
        model, train_loader, val_loader, coco_gt, device,
        total_epochs=final_epochs,
        eval_every=args.eval_every,
        label=f"Final FT @{checkpoint_pct}%",
        recipe=cli_recipe)
    t_ft = time.time() - t_ft_start

    # Save the full torchvision SSD module (best-mAP state already restored)
    torch.save(model, str(out_path))

    print(f"  [Final eval] mAP after FT...", flush=True)
    final_eval = evaluate_coco_map_ssd(model, val_loader, coco_gt, device)
    print(f"  [Final eval] mAP={final_eval['mAP']:.4f} mAP50={final_eval['mAP_50']:.4f} "
          f"mAP75={final_eval['mAP_75']:.4f}  (n_det={final_eval['n_det']})",
          flush=True)

    layer_structure_backbone = extract_layer_structure(model.backbone)
    n_conv = sum(1 for x in layer_structure_backbone if x["kind"] == "Conv2d")
    n_lin = sum(1 for x in layer_structure_backbone if x["kind"] == "Linear")
    print(f"  [structure] backbone: {n_conv} Conv2d + {n_lin} Linear post-pruning",
          flush=True)

    result = {
        "model": args.baseline_name, "importance": imp_name,
        "checkpoint_pct": checkpoint_pct,
        "ref_eval": ref_eval,
        "post_prune_eval": post_prune_eval,
        "final_eval": final_eval,
        "mAP_delta": final_eval["mAP"] - ref_eval["mAP"],
        "pre_params_full": pre_params_full,
        "post_params_full": post_params_full,
        "pre_params_backbone": pre_params_backbone,
        "post_params_backbone": post_params_backbone,
        "param_reduction_pct_full": 100 * (1 - post_params_full / pre_params_full),
        "param_reduction_pct_backbone": 100 * (1 - post_params_backbone / pre_params_backbone),
        "actual_pct_backbone": actual_pct,
        # MACs scope: feature extractor + head convs at 320 (the traceable wrapper); the full SSD adds only pre/post.
        "pre_macs": pre_macs, "post_macs": post_macs,
        "macs_scope": "backbone+heads (SSDLitePerLevelHeads wrapper)",
        "macs_reduction_pct": 100 * (1 - post_macs / pre_macs) if pre_macs > 0 else 0,
        "n_pruning_steps": n_steps,
        "duration_s": {"pruning": t_prune, "finetune": t_ft, "total": t_prune + t_ft},
        "output_file": str(out_path),
        "layer_structure_backbone_post": layer_structure_backbone,
    }

    effective_recipe = {**RECIPE_FT, **cli_recipe}
    full_log = {
        **result,
        "recipe_ft": effective_recipe,
        "pruning_protocol": PRUNING_PROTOCOL,
        "seed": args.seed, "device": str(device),
        "image_size": args.image_size,
        "num_classes": args.num_classes,
        "ft_history": ft_history,
    }
    with open(log_path, "w") as f:
        json.dump(full_log, f, indent=2, default=str)

    print(f"  ╔══ DONE ══")
    print(f"  ║ mAP       : {ref_eval['mAP']:.4f} → {final_eval['mAP']:.4f} "
          f"(Δ={result['mAP_delta']:+.4f})")
    print(f"  ║ Backbone  : {pre_params_backbone:,} → {post_params_backbone:,} "
          f"({result['param_reduction_pct_backbone']:.1f}% reduced)")
    print(f"  ║ Total     : {pre_params_full:,} → {post_params_full:,} "
          f"({result['param_reduction_pct_full']:.1f}% reduced)")
    if pre_macs > 0:
        print(f"  ║ MACs      : {pre_macs:,.0f} → {post_macs:,.0f} "
              f"({result['macs_reduction_pct']:.1f}% reduced)")
    print(f"  ║ Duration  : pruning={t_prune:.0f}s, FT={t_ft:.0f}s")
    print(f"  ║ → {out_path.name} ({out_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"  ║ → {log_path.name}")
    print(f"  ╚{'═' * 50}", flush=True)

    return result


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Pruning SSDLite320-MobileNetV3-Large backbone "
                    "on COCO (PruningBench backbone-only)")

    parser.add_argument("--coco_root", type=str, required=True,
                        help="COCO 2017 root (contains `annotations/`, "
                             "`train2017/`, `val2017/`)")
    parser.add_argument("--coco_subset", type=str, default="train2017",
                        choices=list(COCO_TRAIN_SUBSETS),
                        help="Which COCO split to train on. "
                             "'train2017'=full 118k (default, slow), "
                             "'minitrain'=25k stratified subset (~5x faster, "
                             "needs instances_minitrain2017.json in annotations/), "
                             "'val2017'=5k smoke-test set (leaky, eval also uses val2017). "
                             "Val mAP is always reported on val2017.")
    parser.add_argument("--baseline_name", type=str, default=BASELINE_NAME_DEFAULT,
                        help=f"Baseline prefix in models/ "
                             f"(default: {BASELINE_NAME_DEFAULT})")
    parser.add_argument("--baseline_path", type=Path, default=None,
                        help="Override the baseline .pt path. By default: "
                             "MODELS_DIR / f'{baseline_name}.pt'. Use this to decouple "
                             "the input file from the output prefix.")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES_DEFAULT,
                        help=f"Number of head columns incl. background "
                             f"(default: {NUM_CLASSES_DEFAULT} = COCO-91)")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE_DEFAULT,
                        help=f"Input size for MACs/pruner tracing (default: "
                             f"{IMAGE_SIZE_DEFAULT}; the model's internal "
                             f"transform is fixed at 320)")
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--checkpoints", nargs="+", type=int, required=True,
                        help="BACKBONE pruning levels in %% "
                             "(e.g. 10 20 30 40 50 60 70 80 90)")
    parser.add_argument("--importance", nargs="+", default=None,
                        choices=ALL_IMPORTANCES,
                        help=f"Subset of importances "
                             f"(default: all = {ALL_IMPORTANCES})")

    parser.add_argument("--final_epochs", type=int, default=None,
                        help="Override the number of final FT epochs (default: 100)")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate mAP every N epochs during FT "
                             "(default: 5, to amortize COCO eval cost)")

    # Recovery FT hyperparameters (same defaults as the effdet worker)
    parser.add_argument("--lr", type=float, default=RECIPE_FT["lr"],
                        help=f"Peak learning rate (default: {RECIPE_FT['lr']:g})")
    parser.add_argument("--weight_decay", type=float, default=RECIPE_FT["weight_decay"],
                        help=f"L2 weight decay (default: {RECIPE_FT['weight_decay']:g})")
    parser.add_argument("--scheduler", type=str, default=RECIPE_FT["scheduler"],
                        choices=["cosine", "multistep"],
                        help=f"LR schedule shape after warmup "
                             f"(default: {RECIPE_FT['scheduler']}).")
    parser.add_argument("--warmup_epochs", type=int, default=RECIPE_FT["warmup_epochs"],
                        help=f"Linear-warmup epochs from lr*"
                             f"{RECIPE_FT['warmup_start_factor']:g} to lr "
                             f"(default: {RECIPE_FT['warmup_epochs']}).")
    parser.add_argument("--lr_decay_epochs", type=int, nargs="+",
                        default=RECIPE_FT["lr_decay_epochs"],
                        help=f"MultiStep milestones (used only when "
                             f"--scheduler multistep). Milestones past "
                             f"--final_epochs are dropped. "
                             f"Default: {RECIPE_FT['lr_decay_epochs']}.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if the .pt exists")
    args = parser.parse_args()

    if args.batch_size < 2:
        parser.error("--batch_size must be >= 2: the SSD extra blocks "
                     "BatchNorm over [1, C, 1, 1] crashes on batches of 1.")

    # parents[3] = repo root; see the same note in families/detection/effdet/prune.py.
    OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "outputs"
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
              f"Run scripts/download_and_finetune_models.sh to download the "
              f"torchvision SSDLite COCO weights into {MODELS_DIR}/")
        return

    n_runs = len(importance_list) * len(checkpoints)

    print("=" * 70)
    print("PRUNING SSDLite320-MobileNetV3-Large (backbone-only, PruningBench)")
    print("=" * 70)
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Baseline     : {args.baseline_name} ({baseline_path.name})")
    print(f"  Image size   : {args.image_size}")
    print(f"  Num classes  : {args.num_classes} (incl. background=0)")
    print(f"  COCO root    : {args.coco_root}")
    print(f"  COCO subset  : {args.coco_subset} (train); val=val2017")
    print(f"  Importances  : {importance_list}")
    print(f"  Levels       : {checkpoints}% (backbone)")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Seed         : {args.seed}")
    print(f"  Total runs   : {n_runs}  ({len(importance_list)} × {len(checkpoints)})")
    if args.final_epochs is not None:
        print(f"  ⚠ Override final_epochs={args.final_epochs}")
    print(f"  Eval every   : {args.eval_every} ep")
    sched_extra = (f" ms={args.lr_decay_epochs}"
                   if args.scheduler == "multistep" else "")
    print(f"  FT recipe    : SGD lr={args.lr:g} wd={args.weight_decay:g}, "
          f"{args.scheduler} warmup={args.warmup_epochs}ep{sched_extra}")
    print("=" * 70, flush=True)

    all_results = []
    t_start = time.time()

    for imp_name in importance_list:
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
              f"mAP {r['ref_eval']['mAP']:.4f} → {r['final_eval']['mAP']:.4f}  "
              f"(backbone {r['param_reduction_pct_backbone']:.1f}% reduced, "
              f"ΔmAP={r['mAP_delta']:+.4f})")
    if all_results:
        summary_path = LOG_DIR / f"{args.baseline_name}_coco-{args.coco_subset}_pruning_summary.json"
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
