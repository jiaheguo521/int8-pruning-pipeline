#!/usr/bin/env python3
"""Structured pruning of the EfficientDet-Lite1/Lite2 backbone on COCO 2017.

Structured pruning of the **EfficientNet-Lite backbone of EfficientDet-Lite1/Lite2**
on **COCO 2017** following the **PruningBench** protocol (Li et al. 2024).

⚠️  Task = **object detection** (mAP), not classification. The entire
pipeline (loss, eval, dataloader, ignored_layers) is different from
`families/classification/imagenet_backbones/prune.py` (and from the CIFAR-100 pruning study this
line of work started from).

Strategy: **backbone-only pruning**.
  - We do not touch the BiFPN (`model.fpn`) nor the heads (`model.class_net`,
    `model.box_net`) -- they are placed in `ignored_layers`.
  - torch_pruning traces the full graph (backbone -> BiFPN -> heads) and
    automatically detects that the backbone layers feeding BiFPN cannot
    change their `out_channels` (otherwise it would break the ignored
    BiFPN). So the pruner only touches the convolutions **internal to
    the backbone** (between stages).
  - `--checkpoints` expresses the % of params pruned **in the backbone
    only** (not the entire model), since the rest (BiFPN + heads) is
    frozen. This makes comparisons between runs interpretable.

Importances: the 5 *data-free* subset (identical to `families/classification/imagenet_backbones/prune.py`)
    magnitude_l1, magnitude_l2, fpgm, random, lamp

Recovery FT: PruningBench recipe (SGD lr=0.01 m=0.9 wd=5e-4, MultiStep
[60,80] γ=0.1, 100 ep, batch=16). ⚠️  The detection loss (focal + box
regression) can diverge under lr=0.01 -- a warning is printed if
train_loss > 50 in the first 5 epochs. If that happens, lower to
lr=1e-3 or switch to AdamW (to be coded by hand for now).

Dependencies:
    pip install effdet pycocotools torch_pruning

Baseline contract:
    outputs/models/<baseline_name>.pt = complete EfficientDet nn.Module
    (or DetBenchTrain / DetBenchPredict -- we unwrap automatically).

Usage:
    python families/detection/effdet/prune.py \\
        --coco_root <path>/coco2017 \\
        --baseline_name efficientdet_lite1 \\
        --num_classes 90 --image_size 384 --batch_size 16 \\
        --checkpoints 10 20 30 40 50 60 70 80 90 \\
        --importance magnitude_l1 magnitude_l2 fpgm random lamp

    # Smoke (5 ep FT, 1 importance, 1 checkpoint):
    python families/detection/effdet/prune.py \\
        --coco_root <path>/coco2017 \\
        --checkpoints 50 --importance magnitude_l2 \\
        --final_epochs 5 --eval_every 1 --force
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from int8_pruning.data.coco import COCO_TRAIN_SUBSETS
from int8_pruning.prune.core import (
    ALL_IMPORTANCES, count_macs, count_params, extract_layer_structure,
    get_importance, get_num_workers, progressive_pruning_to_target, seed_everything,
)
from int8_pruning.prune.detection import build_ft_scheduler


IMAGE_SIZE_DEFAULT = 384  # EfficientDet-Lite1 input (Lite2 is 448 -- pass --image_size)
NUM_CLASSES_DEFAULT = 90  # COCO 2017 detection
BASELINE_NAME_DEFAULT = "efficientdet_lite1"


def effdet_cfg_name(baseline_name):
    """`efficientdet_lite<N>` -> the effdet config name carrying its mean/std.

    The baseline `.pt` is a complete nn.Module, so the architecture never comes
    from this config -- only the normalization does. Lite1 and Lite2 both use
    0.5/0.5, but reading it per-rung keeps the rung a data point rather than an
    assumption (see the normalization warning in `get_dataloaders_coco`).
    """
    return f"tf_{baseline_name}"


# Recovery FT recipe. Every field is overridable from the CLI (--lr, --weight_decay,
# --scheduler, --warmup_epochs, --lr_decay_epochs). lr=1e-3 is ~1/10 of the paper's
# scaled scratch lr (0.16 @ bs=128), which is where the literature puts FT of a
# pretrained detector at bs=16; wd=4e-5 is Tan et al. 2020 and rwightman's value.
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
    "warning": ("Recovery FT recipe adapted from EfficientDet paper (Tan et al. 2020). "
                "train_loss is monitored; warning emitted if >50 in the first 5 epochs."),
    "ref": "EfficientDet (Tan et al., CVPR 2020) + PruningBench (Li et al. 2024) protocol",
}

PRUNING_PROTOCOL = {
    "iterative_steps": 400,
    "global_pruning": True,
    "max_pruning_ratio": 0.9,
    "ref": "PruningBench progressive_pruning + 10% protection (backbone-only)",
}


# Reproducibility




# EfficientDet model — load / unwrap / wrap
def unwrap_effdet(model):
    """If `model` is a DetBenchTrain or DetBenchPredict, returns the raw
    EfficientDet. Otherwise returns `model` as-is.

    The benches expose the EfficientDet via `.model`.
    """
    if hasattr(model, "model") and hasattr(model.model, "backbone"):
        return model.model
    return model


def load_baseline_effdet(baseline_path, num_classes, image_size):
    """Loads the `.pt` baseline and returns the raw EfficientDet (no bench).

    Tolerates 3 input formats:
      - raw EfficientDet (ideal)
      - DetBenchTrain(EfficientDet)
      - DetBenchPredict(EfficientDet)
    Checks that `.backbone`, `.fpn`, `.class_net`, `.box_net` exist.
    """
    obj = torch.load(str(baseline_path), map_location="cpu", weights_only=False)
    model = unwrap_effdet(obj)
    for attr in ("backbone", "fpn", "class_net", "box_net"):
        if not hasattr(model, attr):
            raise RuntimeError(
                f"Baseline {baseline_path}: loaded object lacks the `{attr}` "
                f"attribute expected on an EfficientDet. Type={type(model).__name__}"
            )
    return model


def wrap_train_bench(model):
    """Wraps in DetBenchTrain for FT."""
    from effdet import DetBenchTrain
    return DetBenchTrain(model, create_labeler=True)


def wrap_predict_bench(model):
    """Wraps in DetBenchPredict for mAP eval."""
    from effdet import DetBenchPredict
    return DetBenchPredict(model)


# Training subset -> (ann_filename, img_dir), both relative to coco_root; the val
# loader always reads val2017. minitrain is the 25k stratified sample of Samet et al.,
# BMVC 2020 (github.com/giddyyupp/coco-minitrain) over the same train2017/ images.
# val2017 as a training set is smoke-test only: the eval loader reads it too, so it leaks.


def _build_coco_detection_dataset(coco_root, ann_relpath, img_relname):
    """Build an effdet DetectionDatset from an explicit (ann_file, img_dir).

    Used to inject custom COCO-format annotation files (e.g. minitrain)
    that the bundled `Coco2017Cfg` does not know about. Mirrors the body
    of `effdet.data.dataset_factory.create_dataset` for the COCO branch.
    """
    from pathlib import Path as _Path
    from effdet.data.dataset import DetectionDatset
    from effdet.data.parsers import create_parser
    from effdet.data.parsers.parser_config import CocoParserCfg

    coco_root = _Path(coco_root)
    ann_file = coco_root / ann_relpath
    if not ann_file.exists():
        raise FileNotFoundError(
            f"COCO annotation file not found: {ann_file}\n"
            f"For 'minitrain', download via scripts/download_datasets.sh "
            f"(it fetches instances_minitrain2017.json into annotations/)."
        )
    parser_cfg = CocoParserCfg(ann_filename=ann_file, has_labels=True)
    return DetectionDatset(
        data_dir=coco_root / img_relname,
        parser=create_parser("coco", cfg=parser_cfg),
    )


def get_dataloaders_coco(coco_root, batch_size, seed, image_size=384,
                         train_subset="train2017",
                         baseline_name=BASELINE_NAME_DEFAULT):
    """Creates COCO train/val DataLoaders via effdet 0.4.x utilities.

    effdet 0.4.x no longer exports `CocoDetection` -- we go through
    `create_dataset("coco", root, splits=...)` which returns instances of
    `DetectionDataset` configured with `Coco2017Cfg` (paths
    `annotations/instances_<split>2017.json` + `<split>2017/`).

    `train_subset` picks which COCO split feeds training:
      - "train2017" (default): full COCO train (118k images, ~11 h/ep here)
      - "minitrain"          : 25k stratified subset, drops train time ~5x
      - "val2017"             : 5k smoke-test set (also used for eval -> leaky)
    Val loader always uses standard val2017 so the mAP numbers stay
    comparable across subsets.

    The transforms (Resize, Normalize, RandomFlip) are applied by
    `create_loader`. ⚠️  effdet 0.4.x requires `use_prefetcher=True` in
    `transforms_coco_{train,eval}` (hard assert in `transforms.py`); so we
    enable it -- the consequence is that images are already on GPU and
    normalized (mean/std*255) at loader output, which makes the
    `.to(device, non_blocking=True)` on the pruning script side a no-op
    but harmless. Batch format remains `(images_tensor, target_dict)`
    where target_dict contains `bbox`, `cls`, `img_size`, `img_scale`, `img_id`
    (variable length packed).

    The COCO GT object (for `loadRes` / mAP) is exposed as
    `val_loader.dataset.parser.coco`; we return it separately to stay
    compatible with the old signature (`val_ds.coco`).

    ⚠️  Normalization: `create_loader` defaults to ImageNet mean/std, but the
    `tf_efficientdet_lite*` weights were trained with mean=std=0.5 (see the
    model's own `get_efficientdet_config(...)['mean'/'std']`). Feeding
    ImageNet-normalized inputs silently degrades mAP (~0.291 vs ~0.322 on the
    dense Lite1 baseline). We pull mean/std from the config and pass them
    explicitly so train + eval match the pretrained init.
    """
    from effdet.data import create_loader
    from effdet.config import get_efficientdet_config

    _cfg = get_efficientdet_config(effdet_cfg_name(baseline_name))
    _mean, _std = tuple(_cfg["mean"]), tuple(_cfg["std"])

    if train_subset not in COCO_TRAIN_SUBSETS:
        raise ValueError(
            f"Unknown COCO train_subset='{train_subset}'. "
            f"Valid choices: {list(COCO_TRAIN_SUBSETS)}"
        )
    train_ann, train_img = COCO_TRAIN_SUBSETS[train_subset]
    val_ann,   val_img   = COCO_TRAIN_SUBSETS["val2017"]

    train_ds = _build_coco_detection_dataset(coco_root, train_ann, train_img)
    val_ds   = _build_coco_detection_dataset(coco_root, val_ann,   val_img)

    nw = get_num_workers()
    train_loader = create_loader(
        train_ds,
        input_size=(3, image_size, image_size),
        batch_size=batch_size,
        is_training=True,
        use_prefetcher=True,
        num_workers=nw,
        pin_mem=True,
        mean=_mean,
        std=_std,
    )
    val_loader = create_loader(
        val_ds,
        input_size=(3, image_size, image_size),
        batch_size=batch_size,
        is_training=False,
        use_prefetcher=True,
        num_workers=nw,
        pin_mem=True,
        mean=_mean,
        std=_std,
    )
    print(f"  [Dataset COCO] subset={train_subset}  "
          f"train: {len(train_ds)} images, val: {len(val_ds)} images, "
          f"input={image_size}, batch={batch_size}, workers={nw}", flush=True)
    return train_loader, val_loader, val_ds


# Training (detection)
def _targets_to_device(target, device):
    """Move target tensors to device. effdet target is a dict of Tensors."""
    if isinstance(target, dict):
        return {k: (v.to(device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v)
                for k, v in target.items()}
    if isinstance(target, (list, tuple)):
        return [_targets_to_device(t, device) for t in target]
    return target


def train_epoch_det(bench_train, train_loader, optimizer, device,
                    warn_threshold=50.0, desc="train"):
    """One detection FT epoch. Returns (mean loss, max loss seen).

    `bench_train` is a `DetBenchTrain(model)`. Forward returns a dict
    {'loss', 'class_loss', 'box_loss'}; we backprop on 'loss'.
    """
    bench_train.train()
    running_loss, max_loss, n_batches = 0.0, 0.0, 0
    pbar = tqdm(train_loader, desc=desc, leave=False, dynamic_ncols=True,
                mininterval=1.0)
    for batch in pbar:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, targets = batch
        else:
            raise RuntimeError(
                f"Unexpected batch format: {type(batch).__name__}. "
                f"Expected (images, targets)."
            )
        images = images.to(device, non_blocking=True)
        targets = _targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        output = bench_train(images, targets)
        loss = output["loss"]
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
def evaluate_coco_map(bench_predict, val_loader, val_ds, device):
    """Runs DetBenchPredict on val_loader, collects detections in COCO
    submission format, and computes mAP@[0.5:0.95] via pycocotools.

    `val_ds` must be the CocoDetection instance (has `.coco` = COCO GT).

    DetBenchPredict.forward(input, img_info=target) returns a Tensor
    (B, max_det, 6) with [x1, y1, x2, y2, score, class_id]. **class_id is
    already the native COCO category_id (1..90)**: effdet/anchors.py applies
    `classes + 1` in post-processing (background = 0), and its
    CocoDetection parser runs by default with `cat_ids_as_labels=True`,
    so the head has 90 channels indexed by (category_id - 1) and the `+1`
    at the output puts it directly back in COCO ids. No remapping needed here.
    Ref: rwightman/efficientdet-pytorch effdet/anchors.py + parser_coco.py.
    """
    from pycocotools.cocoeval import COCOeval

    # effdet 0.4.x `DetectionDataset.__getitem__` returns `img_idx`, not the COCO
    # `image_id`; `parser.img_ids[dataset_idx]` is the remap. Without it `coco.loadRes`
    # rejects everything with "Results do not correspond to current coco set".
    id_table = val_ds.parser.img_ids

    bench_predict.eval()
    detections = []
    for batch in tqdm(val_loader, desc="eval", leave=False,
                      dynamic_ncols=True, mininterval=1.0):
        if not (isinstance(batch, (list, tuple)) and len(batch) == 2):
            raise RuntimeError(f"Unexpected eval batch format: {type(batch).__name__}")
        images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = _targets_to_device(targets, device)
        # DetBenchPredict exposes `img_info` via the 2nd argument
        det = bench_predict(images, img_info=targets)
        # det: (B, max_det, 6) -- [x1,y1,x2,y2,score,class]
        idxs = targets.get("img_idx", targets.get("img_id", None))
        if idxs is None:
            raise RuntimeError("No img_idx / img_id in the eval batch.")
        det_np = det.cpu().numpy()
        idxs_np = idxs.cpu().numpy() if isinstance(idxs, torch.Tensor) else np.asarray(idxs)
        ids_np = np.asarray([id_table[int(i)] for i in idxs_np], dtype=np.int64)
        for b in range(det_np.shape[0]):
            iid = int(ids_np[b])
            for k in range(det_np.shape[1]):
                x1, y1, x2, y2, score, cls = det_np[b, k]
                if score <= 0:
                    continue
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))
                if w <= 0 or h <= 0:
                    continue
                detections.append({
                    "image_id": iid,
                    "category_id": int(cls),
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(score),
                })

    if not detections:
        print("  [COCO eval] No detection returned -- mAP=0", flush=True)
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0,
                "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0,
                "n_det": 0}

    # effdet 0.4.x keeps COCO GT on `dataset.parser.coco`; fall back in case an older version aliases it.
    coco_gt = getattr(val_ds, "coco", None) or val_ds.parser.coco
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




# Recovery FT (detection, PruningBench recipe)


def finetune_pruningbench_det(bench_train, bench_predict, val_ds,
                              train_loader, val_loader, device,
                              total_epochs, eval_every, label,
                              recipe=None):
    """Detection FT with the EfficientDet paper recipe.

    `recipe` overrides any field of the module-level RECIPE_FT dict
    (typically the CLI-resolved values: lr, weight_decay, scheduler,
    warmup_epochs, lr_decay_epochs).

    Monitor:
      - train_loss every epoch
      - WARN if max_loss > 50 in the first 5 epochs (lr likely too high)
      - mAP every `eval_every` epochs (and always on the last one)
      - Keeps the best (mAP) state_dict.
    """
    if total_epochs <= 0:
        return 0.0, []

    # Merge CLI overrides into the module-level defaults.
    eff_recipe = dict(RECIPE_FT)
    if recipe:
        eff_recipe.update(recipe)

    optimizer = optim.SGD(bench_train.parameters(),
                          lr=eff_recipe["lr"],
                          momentum=eff_recipe["momentum"],
                          weight_decay=eff_recipe["weight_decay"])
    scheduler, sched_desc = build_ft_scheduler(optimizer, total_epochs, eff_recipe)

    print(f"  [{label}] {total_epochs} ep, SGD lr={eff_recipe['lr']:g} "
          f"m={eff_recipe['momentum']} wd={eff_recipe['weight_decay']:g}, "
          f"{sched_desc}, eval_every={eval_every} ep", flush=True)

    best_map, best_state, best_epoch = -1.0, None, -1
    history = []
    raw_model = unwrap_effdet(bench_train)

    for epoch in range(total_epochs):
        t0 = time.time()
        train_loss, max_loss = train_epoch_det(
            bench_train, train_loader, optimizer, device,
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
            eval_dict = evaluate_coco_map(bench_predict, val_loader, val_ds, device)
            dt_eval = time.time() - t_eval
            improved = eval_dict["mAP"] > best_map
            if improved:
                best_map, best_epoch = eval_dict["mAP"], epoch + 1
                best_state = {k: v.cpu().clone()
                              for k, v in raw_model.state_dict().items()}
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
        raw_model.load_state_dict(best_state); raw_model.to(device)
    print(f"  [{label}] Best mAP : {best_map:.4f} @ ep {best_epoch}/{len(history)}",
          flush=True)
    return best_map, history


# Pruning utilities






def get_ignored_layers_effdet(model):
    """Backbone-only pruning: ignore everything except the inner backbone.

    Add to `ignored_layers`:
      - `model.fpn`        (entire BiFPN)
      - `model.class_net`  (classification head)
      - `model.box_net`    (box regression head)
      - `model.backbone.conv_stem` if present (protects in_channels=3)

    ⚠️  Measured 2026-08-17, correcting what this docstring used to claim:
    DepGraph freezes **zero** backbone convolutions here. The BiFPN's entry
    convolutions are pointwise, so a multi-scale tap can change out_channels
    freely and the whole backbone stays prunable (C3/C4/C5 40/112/320 prune
    all the way to 3/9/30). The transitive-freeze mechanism is real for
    *ssdlite*, whose head entry is depthwise `Conv2d(672,672,groups=672)` and
    welds 2,304 backbone channels (14.63% of the model) shut -- see
    `families/detection/ssdlite/prune.py:get_ignored_layers_ssd`. Consequence here:
    unfreezing `model.fpn` raises the reachable ceiling by exactly 0.00 pp,
    because EfficientDet reuses one class_net/box_net conv set across all
    five levels (only the BatchNorms are per-level).
    """
    ignored = []
    if hasattr(model, "fpn"):
        ignored.append(model.fpn)
    if hasattr(model, "class_net"):
        ignored.append(model.class_net)
    if hasattr(model, "box_net"):
        ignored.append(model.box_net)
    if hasattr(model, "backbone") and hasattr(model.backbone, "conv_stem"):
        ignored.append(model.backbone.conv_stem)
    return ignored




# Single run: (importance, checkpoint_pct)
def run_one(imp_name, checkpoint_pct, args, device,
            models_dir, pruned_dir, log_dir):
    """Run for a triplet (fixed baseline EfficientDet-Lite rung, importance, %)."""
    import torch_pruning as tp

    seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    ds_tag = f"coco-{args.coco_subset}"
    # `_full` keeps the two scopes from sharing a filename: `pruned50pct` means 50% of
    # BACKBONE params by default and 50% of FULL-MODEL params under --scope full. It still
    # matches family.yaml's `_pruned\d+pct(?:_\w+)?` because `\w` covers the underscore.
    scope_sfx = "_full" if args.scope == "full" else ""
    out_name = (f"{args.baseline_name}_{ds_tag}_pruned{checkpoint_pct}pct_"
                f"{imp_name}{scope_sfx}{seed_suffix}")
    out_path = pruned_dir / f"{out_name}.pt"
    log_path = (log_dir / f"{args.baseline_name}_{ds_tag}_{imp_name}_"
                          f"{checkpoint_pct}pct{scope_sfx}{seed_suffix}.json")

    if out_path.exists() and not args.force:
        print(f"  [{out_name}] Already present, skip.")
        return None

    print(f"\n{'━' * 70}")
    print(f"  {args.baseline_name.upper()} — {imp_name.upper()} — "
          f"{checkpoint_pct}% (backbone)  (seed={args.seed})")
    print(f"{'━' * 70}", flush=True)

    baseline_path = args.baseline_path or (models_dir / f"{args.baseline_name}.pt")
    seed_everything(args.seed)
    model = load_baseline_effdet(baseline_path, args.num_classes, args.image_size)
    model = model.to(device)

    # Data
    train_loader, val_loader, val_ds = get_dataloaders_coco(
        args.coco_root, args.batch_size, args.seed,
        image_size=args.image_size, train_subset=args.coco_subset,
        baseline_name=args.baseline_name)

    # DetBenchPredict builds anchor_boxes at creation and they stay on CPU without an
    # explicit `.to(device)`, even when the model is already on GPU: without it
    # `effdet.anchors.generate_detections` crashes with "indices should be either on cpu
    # or on the same device as the indexed tensor".
    bench_predict_init = wrap_predict_bench(model).to(device)
    print(f"  [Pre-pruning eval] mAP on COCO val...", flush=True)
    t0 = time.time()
    ref_eval = evaluate_coco_map(bench_predict_init, val_loader, val_ds, device)
    print(f"  Pre-pruning: mAP={ref_eval['mAP']:.4f} mAP50={ref_eval['mAP_50']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del bench_predict_init

    pre_params_full = count_params(model)
    pre_params_backbone = count_params(model.backbone)
    pre_macs = count_macs(model, args.image_size)
    print(f"  Pre-pruning: {pre_params_full:,} params total, "
          f"{pre_params_backbone:,} params backbone, {pre_macs:,.0f} MACs "
          f"(loaded from {baseline_path.name})", flush=True)

    # Build pruner -- MetaPruner on the full model, but with ignored_layers
    # protecting FPN + heads.
    importance = get_importance(imp_name)
    ignored_layers = get_ignored_layers_effdet(model)
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

    print(f"  [Pruning] target={checkpoint_pct}% ({args.scope}), "
          f"iterative_steps={PRUNING_PROTOCOL['iterative_steps']}, "
          f"global={PRUNING_PROTOCOL['global_pruning']}, "
          f"max_per_layer={PRUNING_PROTOCOL['max_pruning_ratio']}, "
          f"ignored={[type(m).__name__ for m in ignored_layers]}", flush=True)
    t_prune_start = time.time()
    # scope=backbone: the ratio is measured over the backbone only (BiFPN +
    # heads are in ignored_layers and must not count toward the target).
    n_steps, actual_pct, _post_params_scoped = progressive_pruning_to_target(
        model, pruner, checkpoint_pct,
        scope=None if args.scope == "full" else (lambda m: m.backbone))
    # `_post_params_scoped` is counted in whatever scope the target used, so it is
    # the full-model count under --scope full. Re-measure the backbone explicitly
    # rather than reusing it, or every "Backbone:" figure below is the total.
    post_params_backbone = count_params(model.backbone)
    t_prune = time.time() - t_prune_start
    post_params_full = count_params(model)
    post_macs = count_macs(model, args.image_size)

    # Post-pruning eval (before FT) -- same `.to(device)` constraint as above
    bench_predict_post = wrap_predict_bench(model).to(device)
    print(f"  [Post-pruning eval] mAP before FT...", flush=True)
    post_prune_eval = evaluate_coco_map(bench_predict_post, val_loader, val_ds, device)
    # `actual_pct` is measured in the TARGET's scope, so under --scope full it is
    # the full-model figure and does NOT describe the backbone transition beside
    # it. Print each with its own denominator.
    bb_pct = 100 * (1 - post_params_backbone / pre_params_backbone)
    full_pct = 100 * (1 - post_params_full / pre_params_full)
    print(f"  [Pruning] {n_steps} steps in {t_prune:.0f}s -- "
          f"backbone {pre_params_backbone:,} → {post_params_backbone:,} ({bb_pct:.1f}%), "
          f"full {pre_params_full:,} → {post_params_full:,} ({full_pct:.1f}%), "
          f"target={actual_pct:.1f}% ({args.scope}), "
          f"mAP={post_prune_eval['mAP']:.4f}", flush=True)
    del bench_predict_post

    # Wrap for FT and eval (both benches share the same `model`)
    bench_train = wrap_train_bench(model).to(device)
    bench_predict = wrap_predict_bench(model).to(device)

    final_epochs = args.final_epochs if args.final_epochs is not None else RECIPE_FT["epochs"]
    cli_recipe = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "lr_decay_epochs": args.lr_decay_epochs,
    }
    t_ft_start = time.time()
    final_map, ft_history = finetune_pruningbench_det(
        bench_train, bench_predict, val_ds,
        train_loader, val_loader, device,
        total_epochs=final_epochs,
        eval_every=args.eval_every,
        label=f"Final FT @{checkpoint_pct}%",
        recipe=cli_recipe)
    t_ft = time.time() - t_ft_start

    # Save the pruned model (raw EfficientDet, no bench)
    raw_pruned = unwrap_effdet(bench_train)
    torch.save(raw_pruned, str(out_path))

    # Final eval (reuses bench_predict, raw_pruned shares the weights)
    print(f"  [Final eval] mAP after FT...", flush=True)
    final_eval = evaluate_coco_map(bench_predict, val_loader, val_ds, device)
    print(f"  [Final eval] mAP={final_eval['mAP']:.4f} mAP50={final_eval['mAP_50']:.4f} "
          f"mAP75={final_eval['mAP_75']:.4f}  (n_det={final_eval['n_det']})",
          flush=True)

    layer_structure_backbone = extract_layer_structure(raw_pruned.backbone)
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
        "pre_macs": pre_macs, "post_macs": post_macs,
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
        description="Pruning an EfficientDet-Lite backbone on COCO "
                    "(PruningBench backbone-only)")

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
                        help=f"Number of COCO classes (default: {NUM_CLASSES_DEFAULT})")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE_DEFAULT,
                        help=f"Input size (default: {IMAGE_SIZE_DEFAULT} for Lite1)")
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

    # Recovery FT hyperparameters (EfficientDet paper defaults; see RECIPE_FT)
    parser.add_argument("--lr", type=float, default=RECIPE_FT["lr"],
                        help=f"Peak learning rate (default: {RECIPE_FT['lr']:g}; "
                             "EfficientDet paper FT-adapted for bs=16). "
                             "Old PruningBench value was 0.01 -- too high for det.")
    parser.add_argument("--weight_decay", type=float, default=RECIPE_FT["weight_decay"],
                        help=f"L2 weight decay (default: {RECIPE_FT['weight_decay']:g}; "
                             "matches Tan et al. 2020).")
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
    parser.add_argument("--scope", choices=["backbone", "full"], default="backbone",
                        help="What --checkpoints is measured against. 'backbone' "
                             "(default) keeps the existing convention: the ratio counts "
                             "only backbone parameters, since the BiFPN and both heads "
                             "sit in ignored_layers. 'full' measures against the whole "
                             "model, which is what the paper means by a reduction in "
                             "remaining parameters, and writes to a '_full' suffixed stem "
                             "so the two ladders never collide. Measured ceilings under "
                             "'full': lite1 88.7%%, lite2 84.5%% -- a 90 target lands short "
                             "there, which the paper's protocol explicitly allows so long "
                             "as the realized value is reported.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if the .pt exists")
    args = parser.parse_args()

    # parents[3] because this worker sits at families/<task>/<family>/. One level short
    # resolves to families/<task>/outputs/, which holds no baseline, so the run dies later
    # on "Baseline not found" rather than at the path. Miscounted twice already.
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
              f"Place your EfficientDet-Lite .pt (raw EfficientDet, DetBenchTrain "
              f"or DetBenchPredict accepted) in {MODELS_DIR}/")
        # Non-zero, not `return`: a bare return made Slurm record a 16-second
        # sweep as COMPLETED/0:0, so a job that produced nothing looked like a
        # successful one in sacct and in any log-watching monitor.
        raise SystemExit(1)

    n_runs = len(importance_list) * len(checkpoints)

    print("=" * 70)
    print(f"PRUNING {args.baseline_name} (backbone-only, PruningBench)")
    print("=" * 70)
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Baseline     : {args.baseline_name} ({baseline_path.name})")
    print(f"  Image size   : {args.image_size}")
    print(f"  Num classes  : {args.num_classes}")
    print(f"  COCO root    : {args.coco_root}")
    print(f"  COCO subset  : {args.coco_subset} (train); val=val2017")
    print(f"  Importances  : {importance_list}")
    print(f"  Levels       : {checkpoints}% ({args.scope} params)")
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
