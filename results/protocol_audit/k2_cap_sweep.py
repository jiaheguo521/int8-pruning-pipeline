"""K2: does max_pruning_ratio actually preserve mAP, or only min-survival?

Prior evidence is allocation-only (min surv 0.083/0.375/0.563/0.760 for caps
0.9/0.5/0.3/0.1 at a 10% backbone target, at zero ratio cost). No mAP was ever
measured for any cap variant. This closes that.

Writes nothing to disk. Prints only.
"""
import sys, time, copy
# Repo root, from this file's location (results/protocol_audit/<this>). It was a hard-coded absolute
# path when this ran out of outputs/; deriving it is what lets the script run from a clone.
import os as _os
from pathlib import Path as _Path
REPO = _os.environ.get("ETPU_REPO") or str(_Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO + "/src")
sys.path.insert(0, REPO + "/families/detection/effdet")

import torch, torch_pruning as tp
from int8_pruning.prune.core import (seed_everything, count_params, get_importance,
                             progressive_pruning_to_target)
from prune import (load_baseline_effdet, wrap_predict_bench,
                   get_dataloaders_coco, evaluate_coco_map,
                   get_ignored_layers_effdet)

DEV = "cuda"
BASE = REPO + "/outputs/models/efficientdet_lite1.pt"
IMG, NC, TARGET = 384, 90, 10

print("[setup] building val2017 loader (5000 imgs)...", flush=True)
_, val_loader, val_ds = get_dataloaders_coco(
    REPO + "/data/datasets/coco", 16, 42, image_size=IMG,
    train_subset="val2017", baseline_name="efficientdet_lite1")

def min_survival(model, init):
    worst, rows = (None, 1.0), []
    for n, m in model.backbone.named_modules():
        if isinstance(m, torch.nn.Conv2d) and n in init:
            r = m.out_channels / init[n]
            rows.append((r, n, init[n], m.out_channels))
            if r < worst[1]:
                worst = (n, r)
    rows.sort()
    return worst, rows[:5]

results = []
for cap in (0.9, 0.5, 0.3, 0.1):
    seed_everything(42)
    model = load_baseline_effdet(BASE, NC, IMG).to(DEV)
    init = {n: m.out_channels for n, m in model.backbone.named_modules()
            if isinstance(m, torch.nn.Conv2d)}
    p0_full, p0_bb = count_params(model), count_params(model.backbone)

    pruner = tp.pruner.MetaPruner(
        model=model, example_inputs=torch.randn(1, 3, IMG, IMG).to(DEV),
        importance=get_importance("magnitude_l2"),
        pruning_ratio=1.0, iterative_steps=400,
        global_pruning=True, max_pruning_ratio=cap,
        ignored_layers=get_ignored_layers_effdet(model))

    t0 = time.time()
    steps, bb_pct, _ = progressive_pruning_to_target(
        model, pruner, TARGET, scope=lambda m: m.backbone)
    full_pct = 100 * (1 - count_params(model) / p0_full)
    worst, w5 = min_survival(model, init)

    t1 = time.time()
    ev = evaluate_coco_map(wrap_predict_bench(model).to(DEV), val_loader, val_ds, DEV)
    results.append((cap, steps, bb_pct, full_pct, worst, ev["mAP"], ev["mAP_50"]))

    print(f"\n{'='*78}\ncap={cap}  steps={steps} ({time.time()-t1:.0f}s eval, {t1-t0:.0f}s prune)")
    print(f"  backbone -{bb_pct:.2f}%   full-model -{full_pct:.2f}%")
    print(f"  min survival: {worst[0]}  {worst[1]*100:.1f}%")
    for r, n, a, b in w5:
        print(f"      {n:38} {a:4d} -> {b:4d}   {r*100:5.1f}%")
    print(f"  ==> mAP={ev['mAP']:.4f}  mAP50={ev['mAP_50']:.4f}", flush=True)
    del model, pruner
    torch.cuda.empty_cache()

print(f"\n\n{'#'*78}\nK2 SUMMARY  (efficientdet_lite1, COCO val2017 5000, global=True, magnitude_l2,")
print(f"             backbone target {TARGET}%, no finetune.  Baseline mAP 0.3220)")
print(f"{'#'*78}")
print(f"{'cap':>5} {'steps':>6} {'bb%':>7} {'full%':>7} {'minsurv':>8}  {'mAP':>7} {'mAP50':>7}")
for cap, st, bb, fu, w, m, m50 in results:
    print(f"{cap:>5} {st:>6} {bb:>7.2f} {fu:>7.2f} {w[1]*100:>7.1f}%  {m:>7.4f} {m50:>7.4f}")
print("\nreference points from the audit (same target, no finetune):")
print("  global=True  lamp      minsurv 80.5%   mAP 0.1092")
print("  global=False magnitude minsurv 91.7%   mAP 0.0820")
print("  global=True  random    minsurv 87.5%   mAP 0.0674")
