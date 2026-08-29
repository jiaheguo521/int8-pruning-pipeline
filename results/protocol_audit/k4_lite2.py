"""K4: does the collapse transfer to efficientdet_lite2 -- the rung actually
deployed downstream? Everything measured so far is lite1.

Mirrors the lite1 four-way control at a matched backbone target.
Writes nothing to disk. Prints only.
"""
import sys, time
# Repo root, from this file's location (results/protocol_audit/<this>). It was a hard-coded absolute
# path when this ran out of outputs/; deriving it is what lets the script run from a clone.
import os as _os
from pathlib import Path as _Path
REPO = _os.environ.get("ETPU_REPO") or str(_Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO + "/src"); sys.path.insert(0, REPO + "/families/detection/effdet")

import torch, torch_pruning as tp
from int8_pruning.prune.core import (seed_everything, count_params, get_importance,
                             progressive_pruning_to_target)
from prune import (load_baseline_effdet, wrap_predict_bench, get_dataloaders_coco,
                   evaluate_coco_map, get_ignored_layers_effdet)

DEV, IMG, NC, TARGET = "cuda", 448, 90, 10
BASE = REPO + "/outputs/models/efficientdet_lite2.pt"

print("[setup] lite2, image_size=448, building val2017 loader...", flush=True)
_, val_loader, val_ds = get_dataloaders_coco(
    REPO + "/data/datasets/coco", 16, 42, image_size=IMG,
    train_subset="val2017", baseline_name="efficientdet_lite2")

# --- baseline mAP: NOT recorded anywhere on disk (0.3588 was only ever pasted). Establish it first-hand.
seed_everything(42)
m0 = load_baseline_effdet(BASE, NC, IMG).to(DEV)
t = time.time()
base = evaluate_coco_map(wrap_predict_bench(m0).to(DEV), val_loader, val_ds, DEV)
print(f"\n[BASELINE] lite2 mAP={base['mAP']:.4f} mAP50={base['mAP_50']:.4f} "
      f"({time.time()-t:.0f}s)  <-- first on-disk-reproducible value", flush=True)
print(f"[BASELINE] params full={count_params(m0):,} backbone={count_params(m0.backbone):,}")
del m0; torch.cuda.empty_cache()

rows = []
for imp, glob in (("magnitude_l2", True), ("magnitude_l2", False), ("lamp", True)):
    seed_everything(42)
    model = load_baseline_effdet(BASE, NC, IMG).to(DEV)
    init = {n: mm.out_channels for n, mm in model.backbone.named_modules()
            if isinstance(mm, torch.nn.Conv2d)}
    p0 = count_params(model)
    pruner = tp.pruner.MetaPruner(
        model=model, example_inputs=torch.randn(1, 3, IMG, IMG).to(DEV),
        importance=get_importance(imp), pruning_ratio=1.0, iterative_steps=400,
        global_pruning=glob, max_pruning_ratio=0.9,
        ignored_layers=get_ignored_layers_effdet(model))
    steps, bb, _ = progressive_pruning_to_target(model, pruner, TARGET,
                                                 scope=lambda m: m.backbone)
    full = 100 * (1 - count_params(model) / p0)
    surv = sorted((mm.out_channels / init[n], n, init[n], mm.out_channels)
                  for n, mm in model.backbone.named_modules()
                  if isinstance(mm, torch.nn.Conv2d) and n in init)
    ev = evaluate_coco_map(wrap_predict_bench(model).to(DEV), val_loader, val_ds, DEV)
    rows.append((imp, glob, steps, bb, full, surv[0], ev["mAP"]))
    print(f"\n{'='*78}\n{imp}  global={glob}  steps={steps}")
    print(f"  backbone -{bb:.2f}%  full -{full:.2f}%")
    for r, n, a, b in surv[:4]:
        print(f"      {n:38} {a:4d} -> {b:4d}  {r*100:5.1f}%")
    print(f"  ==> mAP={ev['mAP']:.4f}", flush=True)
    del model, pruner; torch.cuda.empty_cache()

print(f"\n\n{'#'*78}\nK4 SUMMARY — efficientdet_LITE2 (the deployed rung), "
      f"COCO val2017 5000, bb target {TARGET}%, no finetune")
print(f"{'#'*78}\nbaseline mAP {base['mAP']:.4f}")
print(f"{'criterion':>14} {'global':>7} {'bb%':>7} {'full%':>7} {'minsurv':>26} {'mAP':>8}")
for imp, g, st, bb, fu, w, m in rows:
    print(f"{imp:>14} {str(g):>7} {bb:>7.2f} {fu:>7.2f} "
          f"{w[1][:20]:>20} {w[0]*100:5.1f}% {m:>8.4f}")
print("\nlite1 reference (same protocol): base 0.3220 | g=T mag 0.0003 (8.3%) | "
      "g=F mag 0.0820 (91.7%) | g=T lamp 0.1092 (80.5%)")
