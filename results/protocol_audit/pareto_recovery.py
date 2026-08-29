"""Pareto: recovery-finetuned mAP at DEEP backbone ratios on efficientdet_lite1.

K1 answered recoverability at ONE ratio (bb 10%) and simultaneously proved that
post-prune / pre-finetune mAP does not predict recoverability. So the whole
"how deep can detection be pruned" question is open: there is currently not a
single post-finetune point beyond bb 10%.

Arms, run in this order so a partial run still yields a contiguous curve:
    lamp   g=T  bb 20, 30, 50     -- the candidate deployment ladder
    mag_l2 g=F  bb 50             -- crossover check at the deep end.
                                     reid's measured table shows global+LAMP can
                                     RELOCATE starvation and lose to local at high
                                     ratios, so a LAMP-only ladder could recommend
                                     the wrong criterion where it matters most.

Same finetune budget as K1 so points are comparable: 15 ep, minitrain-25k,
bs8, lr 5e-4, cosine + 1ep warmup. Saves one .pt per arm + an incremental JSON.
"""
import sys, os, json, time, gc
# Repo root, from this file's location (results/protocol_audit/<this>). It was a hard-coded absolute
# path when this ran out of outputs/; deriving it is what lets the script run from a clone.
import os as _os
from pathlib import Path as _Path
REPO = _os.environ.get("ETPU_REPO") or str(_Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO + "/src"); sys.path.insert(0, REPO + "/families/detection/effdet")

import torch, torch_pruning as tp
from int8_pruning.prune.core import (seed_everything, count_params, get_importance,
                             progressive_pruning_to_target)
from prune import (load_baseline_effdet, wrap_train_bench, wrap_predict_bench,
                   get_dataloaders_coco, evaluate_coco_map, unwrap_effdet,
                   get_ignored_layers_effdet, finetune_pruningbench_det, RECIPE_FT)

DEV, IMG, NC = "cuda", 384, 90
BS, EPOCHS, EVAL_EVERY, LR = 8, 15, 5, 5e-4
BASE = REPO + "/outputs/models/efficientdet_lite1.pt"
OUT = REPO + "/outputs/pareto_recovery"
os.makedirs(OUT, exist_ok=True)

ARMS = [("lamp", True, 20), ("lamp", True, 30), ("lamp", True, 50),
        ("magnitude_l2", False, 50)]

print(f"[setup] lite1 {IMG}px | minitrain bs={BS} | {EPOCHS} ep | lr={LR} | "
      f"{len(ARMS)} arms", flush=True)
train_loader, val_loader, val_ds = get_dataloaders_coco(
    REPO + "/data/datasets/coco", BS, 42, image_size=IMG,
    train_subset="minitrain", baseline_name="efficientdet_lite1")

summary = {"baseline_mAP": 0.3220, "model": "efficientdet_lite1", "image_size": IMG,
           "finetune": {"epochs": EPOCHS, "batch_size": BS, "lr": LR,
                        "subset": "minitrain", "eval_every": EVAL_EVERY,
                        "deviations": ["bs 16->8 (4070 OOM at 16)",
                                       "lr 1e-3->5e-4 (linear scaling for bs8)",
                                       "epochs 100->15 (time budget)"],
                        "comparable_to": "outputs/k1_recovery (same budget, bb 10%)"},
           "arms": []}

for imp, glob, target in ARMS:
    tag = f"{imp}_global{'T' if glob else 'F'}_bb{target}"
    print(f"\n{'#'*78}\n#  ARM {tag}\n{'#'*78}", flush=True)
    t_arm = time.time()
    try:
        seed_everything(42)
        model = load_baseline_effdet(BASE, NC, IMG).to(DEV)
        init = {n: m.out_channels for n, m in model.backbone.named_modules()
                if isinstance(m, torch.nn.Conv2d)}
        p0 = count_params(model)
        pruner = tp.pruner.MetaPruner(
            model=model, example_inputs=torch.randn(1, 3, IMG, IMG).to(DEV),
            importance=get_importance(imp), pruning_ratio=1.0, iterative_steps=400,
            global_pruning=glob, max_pruning_ratio=0.9,
            ignored_layers=get_ignored_layers_effdet(model))
        steps, bb_pct, p1_bb = progressive_pruning_to_target(
            model, pruner, target, scope=lambda m: m.backbone)
        del pruner; gc.collect(); torch.cuda.empty_cache()
        p1 = count_params(model)
        full_pct = 100 * (1 - p1 / p0)
        surv = sorted((m.out_channels / init[n], n, init[n], m.out_channels)
                      for n, m in model.backbone.named_modules()
                      if isinstance(m, torch.nn.Conv2d) and n in init)
        print(f"  {steps} steps | backbone -{bb_pct:.2f}% | full -{full_pct:.2f}% | "
              f"minsurv {surv[0][1]} {surv[0][0]*100:.1f}%", flush=True)

        bp = wrap_predict_bench(model).to(DEV)
        pre = evaluate_coco_map(bp, val_loader, val_ds, DEV)
        print(f"  pre-FT mAP = {pre['mAP']:.4f}", flush=True)
        del bp; gc.collect(); torch.cuda.empty_cache()

        bt = wrap_train_bench(model).to(DEV); bp = wrap_predict_bench(model).to(DEV)
        best, hist = finetune_pruningbench_det(
            bt, bp, val_ds, train_loader, val_loader, DEV,
            total_epochs=EPOCHS, eval_every=EVAL_EVERY, label=f"pareto {tag}",
            recipe={"lr": LR, "weight_decay": RECIPE_FT["weight_decay"],
                    "scheduler": "cosine", "warmup_epochs": 1,
                    "lr_decay_epochs": RECIPE_FT["lr_decay_epochs"]})
        final = evaluate_coco_map(bp, val_loader, val_ds, DEV)
        pt = f"{OUT}/efficientdet_lite1_bb{target}pct_{imp}_global{'T' if glob else 'F'}_ft{EPOCHS}ep.pt"
        torch.save(unwrap_effdet(bt), pt)

        summary["arms"].append({
            "importance": imp, "global_pruning": glob, "backbone_target_pct": target,
            "steps": steps, "backbone_reduction_pct": round(bb_pct, 2),
            "full_model_reduction_pct": round(full_pct, 2),
            "params_full": p1, "params_backbone": p1_bb,
            "min_survival": {"layer": surv[0][1], "ratio": round(surv[0][0], 4),
                             "from": surv[0][2], "to": surv[0][3]},
            "worst5": [{"layer": n, "from": a, "to": b, "ratio": round(r, 4)}
                       for r, n, a, b in surv[:5]],
            "pre_ft_mAP": pre["mAP"], "post_ft_mAP": final["mAP"],
            "post_ft_mAP50": final["mAP_50"], "best_ft_mAP": best,
            "ft_history": hist, "artifact": os.path.basename(pt),
            "wall_min": round((time.time() - t_arm) / 60, 1)})
        print(f"\n  ==> {tag}: pre-FT {pre['mAP']:.4f} -> post-FT {final['mAP']:.4f} "
              f"({100*final['mAP']/0.3220:.1f}% of base) [{summary['arms'][-1]['wall_min']:.0f} min]",
              flush=True)
        del bt, bp, model
    except Exception as e:
        import traceback; traceback.print_exc()
        summary["arms"].append({"importance": imp, "global_pruning": glob,
                                "backbone_target_pct": target,
                                "error": f"{type(e).__name__}: {e}"})
    gc.collect(); torch.cuda.empty_cache()
    with open(f"{OUT}/pareto_recovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

print(f"\n\n{'#'*90}\nPARETO — recovery-finetuned mAP vs depth (lite1, {EPOCHS}ep minitrain bs{BS})")
print(f"{'#'*90}\nbaseline 0.3220 | bb10 reference: lamp 0.3027 / local 0.2952 / global-mag 0.2043")
print(f"{'arm':>26} {'bb%':>7} {'full%':>7} {'minsurv':>8} {'pre-FT':>8} {'post-FT':>8} {'%base':>7}")
for a in summary["arms"]:
    if "error" in a:
        print(f"{a['importance']+' bb'+str(a['backbone_target_pct']):>26}  ERROR {a['error'][:44]}"); continue
    t = f"{a['importance']} g={'T' if a['global_pruning'] else 'F'} bb{a['backbone_target_pct']}"
    print(f"{t:>26} {a['backbone_reduction_pct']:>7.2f} {a['full_model_reduction_pct']:>7.2f} "
          f"{a['min_survival']['ratio']*100:>7.1f}% {a['pre_ft_mAP']:>8.4f} "
          f"{a['post_ft_mAP']:>8.4f} {100*a['post_ft_mAP']/0.3220:>6.1f}%")
