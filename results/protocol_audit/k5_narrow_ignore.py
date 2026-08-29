"""K5: does NARROWING ignored_layers preserve accuracy, not just output shape?

Today effdet freezes model.fpn + class_net + box_net + conv_stem, so only the
backbone is prunable. The audit measured that freezing ONLY the two output
convolutions lifts the reachable ceiling from 88.74% to 98.29% -- but no mAP was
ever run on that variant, and the stated risk is real: the BiFPN and head
internals then enter the SAME global importance pool that gutted the backbone.

Output contract (verified from the module tree, lite1):
    class_net.predict.conv_pw   88 -> 810   = 9 anchors x 90 classes   LOCKED
    box_net.predict.conv_pw     88 ->  36   = 9 anchors x 4 coords     LOCKED
Their INPUT width (88, shared with the whole BiFPN) is free to shrink.

Design: one arm only. Match the EXISTING bb50 LAMP point at full-model -43.97%
(post-FT 0.2555) by targeting the same FULL-MODEL reduction with scope=None, so
the two differ ONLY in where the cut is allowed to land. Same 15ep budget.

K1/Pareto proved pre-FT mAP has no predictive value, so this MUST include
recovery fine-tuning to mean anything.
"""
import sys, os, json, time, gc
# Repo root, from this file's location (results/protocol_audit/<this>). It was a hard-coded absolute
# path when this ran out of outputs/; deriving it is what lets the script run from a clone.
import os as _os
from pathlib import Path as _Path
REPO = _os.environ.get("ETPU_REPO") or str(_Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO + "/src"); sys.path.insert(0, REPO + "/families/detection/effdet")

import torch, torch.nn as nn, torch_pruning as tp
from int8_pruning.prune.core import (seed_everything, count_params, get_importance,
                             progressive_pruning_to_target)
from prune import (load_baseline_effdet, wrap_train_bench, wrap_predict_bench,
                   get_dataloaders_coco, evaluate_coco_map, unwrap_effdet,
                   finetune_pruningbench_det, RECIPE_FT)

DEV, IMG, NC = "cuda", 384, 90
BS, EPOCHS, EVAL_EVERY, LR = 8, 15, 5, 5e-4
TARGET_FULL = 43.97          # match the existing bb50 LAMP point exactly
OUT = REPO + "/outputs/k5_narrow_ignore"; os.makedirs(OUT, exist_ok=True)

def narrowed_ignored(model):
    """Freeze ONLY what the output contract requires, plus the RGB stem."""
    ig = [model.class_net.predict.conv_pw,   # out=810, locked
          model.box_net.predict.conv_pw]     # out=36,  locked
    if hasattr(model.backbone, "conv_stem"):
        ig.append(model.backbone.conv_stem)
    return ig

seed_everything(42)
model = load_baseline_effdet(REPO + "/outputs/models/efficientdet_lite1.pt", NC, IMG).to(DEV)
p0 = count_params(model)
p0_parts = {k: count_params(getattr(model, k)) for k in ("backbone", "fpn", "class_net", "box_net")}
print(f"[base] total {p0:,} | " + " ".join(f"{k} {v:,}" for k, v in p0_parts.items()), flush=True)

pruner = tp.pruner.MetaPruner(
    model=model, example_inputs=torch.randn(1, 3, IMG, IMG).to(DEV),
    importance=get_importance("lamp"), pruning_ratio=1.0, iterative_steps=400,
    global_pruning=True, max_pruning_ratio=0.9,
    ignored_layers=narrowed_ignored(model))
steps, full_pct, _ = progressive_pruning_to_target(model, pruner, TARGET_FULL, scope=None)
del pruner; gc.collect(); torch.cuda.empty_cache()

p1_parts = {k: count_params(getattr(model, k)) for k in ("backbone", "fpn", "class_net", "box_net")}
print(f"\n[pruned] {steps} steps, full -{full_pct:.2f}%   (target {TARGET_FULL}%)")
print(f"  {'part':<12} {'before':>10} {'after':>10} {'cut':>8}")
for k in p0_parts:
    print(f"  {k:<12} {p0_parts[k]:>10,} {p1_parts[k]:>10,} "
          f"{100*(1-p1_parts[k]/p0_parts[k]):>7.1f}%")

# ---- output contract check: must fail loudly, not silently ship a broken model
cls_out = model.class_net.predict.conv_pw.out_channels
box_out = model.box_net.predict.conv_pw.out_channels
fpn_w   = model.class_net.predict.conv_pw.in_channels
print(f"\n[contract] class_net out={cls_out} (need 810)  box_net out={box_out} (need 36)  "
      f"shared width 88 -> {fpn_w}", flush=True)
assert cls_out == 810 and box_out == 36, "OUTPUT CONTRACT BROKEN -- refusing to continue"

bench_p = wrap_predict_bench(model).to(DEV)
with torch.no_grad():
    raw = unwrap_effdet(bench_p)(torch.randn(1, 3, IMG, IMG).to(DEV))
shapes = [tuple(t.shape) for t in (raw if isinstance(raw, (list, tuple)) else [raw])]
print(f"[contract] forward OK, {len(shapes)} head tensors: {shapes[:3]} ...", flush=True)

_, val_loader, val_ds = get_dataloaders_coco(REPO + "/data/datasets/coco", BS, 42,
    image_size=IMG, train_subset="val2017", baseline_name="efficientdet_lite1")
pre = evaluate_coco_map(bench_p, val_loader, val_ds, DEV)
print(f"[pre-FT] mAP = {pre['mAP']:.4f}  (expect ~0; not predictive)", flush=True)
del bench_p; gc.collect(); torch.cuda.empty_cache()

train_loader, val_loader, val_ds = get_dataloaders_coco(REPO + "/data/datasets/coco", BS, 42,
    image_size=IMG, train_subset="minitrain", baseline_name="efficientdet_lite1")
bt = wrap_train_bench(model).to(DEV); bp = wrap_predict_bench(model).to(DEV)
t0 = time.time()
best, hist = finetune_pruningbench_det(bt, bp, val_ds, train_loader, val_loader, DEV,
    total_epochs=EPOCHS, eval_every=EVAL_EVERY, label="K5 narrow-ignore",
    recipe={"lr": LR, "weight_decay": RECIPE_FT["weight_decay"], "scheduler": "cosine",
            "warmup_epochs": 1, "lr_decay_epochs": RECIPE_FT["lr_decay_epochs"]})
final = evaluate_coco_map(bp, val_loader, val_ds, DEV)
torch.save(unwrap_effdet(bt), f"{OUT}/efficientdet_lite1_narrowignore_full44pct_lamp_ft15ep.pt")

res = {"variant": "narrowed ignored_layers (only the two output convs + stem)",
       "criterion": "lamp", "global_pruning": True, "target_full_pct": TARGET_FULL,
       "realized_full_pct": round(full_pct, 2), "steps": steps,
       "params_before": p0_parts, "params_after": p1_parts,
       "shared_fpn_width": {"before": 88, "after": fpn_w},
       "contract": {"class_net_out": cls_out, "box_net_out": box_out,
                    "n_head_tensors": len(shapes), "forward_ok": True},
       "pre_ft_mAP": pre["mAP"], "post_ft_mAP": final["mAP"], "best_ft_mAP": best,
       "ft_history": hist, "wall_min": round((time.time()-t0)/60, 1),
       "baseline_mAP": 0.3220,
       "comparison_point": {"variant": "current ignored_layers (backbone-only), bb50 lamp",
                            "full_pct": -43.97, "post_ft_mAP": 0.2555, "frac_of_base": 0.793}}
json.dump(res, open(f"{OUT}/k5_summary.json", "w"), indent=2)

print(f"\n{'#'*80}\nK5 — narrowed ignored_layers vs current, matched full-model reduction")
print(f"{'#'*80}")
print(f"  current  (backbone-only)  full -43.97%  post-FT 0.2555  ({100*0.2555/0.3220:.1f}% of base)")
print(f"  narrowed (fpn+heads too)  full -{full_pct:.2f}%  post-FT {final['mAP']:.4f}  "
      f"({100*final['mAP']/0.3220:.1f}% of base)")
d = final["mAP"] - 0.2555
print(f"  -> narrowing is {'BETTER' if d>0 else 'WORSE'} by {abs(d):.4f} mAP ({100*d/0.2555:+.1f}%)")
