"""K1: does the detection collapse survive recovery fine-tuning?

Every detection mAP measured so far is PRE-finetune, while the paper's protocol
explicitly includes a recovery stage. This is the blocking unknown.

Three arms at a matched backbone-10% target on efficientdet_lite1:
    g=True  lamp          -- the candidate recommended criterion (pre-FT 0.1092)
    g=True  magnitude_l2  -- the defective status quo            (pre-FT 0.0003)
    g=False magnitude_l2  -- the local-ranking control           (pre-FT 0.0820)

Local RTX 4070 cannot fit bs=16 (measured OOM), so bs=8 with lr linearly
scaled 1e-3 -> 5e-4. Both deviations are recorded in the output JSON.
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

DEV, IMG, NC, TARGET = "cuda", 384, 90, 10
BS, EPOCHS, EVAL_EVERY, LR = 8, 15, 5, 5e-4
BASE = REPO + "/outputs/models/efficientdet_lite1.pt"
OUT = REPO + "/outputs/k1_recovery"
os.makedirs(OUT, exist_ok=True)

ARMS = [
    ("lamp",         True,  0.1092),
    ("magnitude_l2", True,  0.0003),
    ("magnitude_l2", False, 0.0820),
]

print(f"[setup] lite1 {IMG}px | minitrain bs={BS} | {EPOCHS} ep | lr={LR} "
      f"(scaled from {RECIPE_FT['lr']:g} @bs16) | backbone target {TARGET}%", flush=True)
train_loader, val_loader, val_ds = get_dataloaders_coco(
    REPO + "/data/datasets/coco", BS, 42, image_size=IMG,
    train_subset="minitrain", baseline_name="efficientdet_lite1")
print(f"[data] {len(train_loader)} train batches/ep", flush=True)

summary = {"baseline_mAP": 0.3220, "model": "efficientdet_lite1",
           "image_size": IMG, "backbone_target_pct": TARGET,
           "finetune": {"epochs": EPOCHS, "batch_size": BS, "lr": LR,
                        "eval_every": EVAL_EVERY, "subset": "minitrain",
                        "deviations": ["bs 16->8 (4070 OOM at 16)",
                                       "lr 1e-3->5e-4 (linear scaling for bs8)",
                                       "epochs 100->15 (time budget)"]},
           "arms": []}

for imp, glob, ref in ARMS:
    tag = f"{imp}_global{'T' if glob else 'F'}"
    print(f"\n{'#'*78}\n#  ARM {tag}   (pre-FT reference {ref:.4f})\n{'#'*78}", flush=True)
    t_arm = time.time()
    try:
        seed_everything(42)
        model = load_baseline_effdet(BASE, NC, IMG).to(DEV)
        init = {n: m.out_channels for n, m in model.backbone.named_modules()
                if isinstance(m, torch.nn.Conv2d)}
        p0_full, p0_bb = count_params(model), count_params(model.backbone)

        pruner = tp.pruner.MetaPruner(
            model=model, example_inputs=torch.randn(1, 3, IMG, IMG).to(DEV),
            importance=get_importance(imp), pruning_ratio=1.0, iterative_steps=400,
            global_pruning=glob, max_pruning_ratio=0.9,
            ignored_layers=get_ignored_layers_effdet(model))
        steps, bb_pct, p1_bb = progressive_pruning_to_target(
            model, pruner, TARGET, scope=lambda m: m.backbone)
        del pruner; gc.collect(); torch.cuda.empty_cache()

        p1_full = count_params(model)
        surv = sorted((m.out_channels / init[n], n, init[n], m.out_channels)
                      for n, m in model.backbone.named_modules()
                      if isinstance(m, torch.nn.Conv2d) and n in init)
        print(f"  pruned in {steps} steps: backbone -{bb_pct:.2f}%  "
              f"full -{100*(1-p1_full/p0_full):.2f}%  minsurv {surv[0][1]} "
              f"{surv[0][0]*100:.1f}%", flush=True)

        bp = wrap_predict_bench(model).to(DEV)
        pre = evaluate_coco_map(bp, val_loader, val_ds, DEV)
        print(f"  post-prune / pre-FT mAP = {pre['mAP']:.4f}", flush=True)
        del bp; gc.collect(); torch.cuda.empty_cache()

        bt = wrap_train_bench(model).to(DEV)
        bp = wrap_predict_bench(model).to(DEV)
        best, hist = finetune_pruningbench_det(
            bt, bp, val_ds, train_loader, val_loader, DEV,
            total_epochs=EPOCHS, eval_every=EVAL_EVERY, label=f"K1 {tag}",
            recipe={"lr": LR, "weight_decay": RECIPE_FT["weight_decay"],
                    "scheduler": "cosine", "warmup_epochs": 1,
                    "lr_decay_epochs": RECIPE_FT["lr_decay_epochs"]})
        final = evaluate_coco_map(bp, val_loader, val_ds, DEV)

        raw = unwrap_effdet(bt)
        pt = f"{OUT}/efficientdet_lite1_bb{TARGET}pct_{tag}_ft{EPOCHS}ep.pt"
        torch.save(raw, pt)

        rec = {"importance": imp, "global_pruning": glob, "steps": steps,
               "backbone_reduction_pct": round(bb_pct, 2),
               "full_model_reduction_pct": round(100*(1-p1_full/p0_full), 2),
               "params_full": p1_full, "params_backbone": p1_bb,
               "min_survival": {"layer": surv[0][1], "ratio": round(surv[0][0], 4),
                                "from": surv[0][2], "to": surv[0][3]},
               "worst5": [{"layer": n, "from": a, "to": b, "ratio": round(r, 4)}
                          for r, n, a, b in surv[:5]],
               "pre_ft_mAP": pre["mAP"], "post_ft_mAP": final["mAP"],
               "post_ft_mAP50": final["mAP_50"], "best_ft_mAP": best,
               "ft_history": hist, "artifact": os.path.basename(pt),
               "wall_min": round((time.time()-t_arm)/60, 1)}
        summary["arms"].append(rec)
        print(f"\n  ==> {tag}: pre-FT {pre['mAP']:.4f} -> post-FT "
              f"{final['mAP']:.4f} (best {best:.4f})  [{rec['wall_min']:.0f} min]", flush=True)
        del bt, bp, model, raw
    except Exception as e:
        import traceback; traceback.print_exc()
        summary["arms"].append({"importance": imp, "global_pruning": glob,
                                "error": f"{type(e).__name__}: {e}"})
    gc.collect(); torch.cuda.empty_cache()
    with open(f"{OUT}/k1_recovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

print(f"\n\n{'#'*78}\nK1 SUMMARY — does the collapse survive recovery fine-tuning?")
print(f"{'#'*78}\nbaseline mAP 0.3220 | lite1 | backbone {TARGET}% | "
      f"{EPOCHS} ep minitrain bs{BS} lr{LR}")
print(f"{'arm':>26} {'minsurv':>8} {'pre-FT':>8} {'post-FT':>8} {'best':>8} {'% of base':>10}")
for a in summary["arms"]:
    if "error" in a:
        print(f"{a['importance']+' g='+str(a['global_pruning']):>26}  ERROR {a['error'][:40]}")
        continue
    tag = f"{a['importance']} g={'T' if a['global_pruning'] else 'F'}"
    print(f"{tag:>26} {a['min_survival']['ratio']*100:>7.1f}% {a['pre_ft_mAP']:>8.4f} "
          f"{a['post_ft_mAP']:>8.4f} {a['best_ft_mAP']:>8.4f} "
          f"{100*a['post_ft_mAP']/0.3220:>9.1f}%")
print(f"\nwritten: {OUT}/k1_recovery_summary.json")
