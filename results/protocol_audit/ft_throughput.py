"""Measure real finetune throughput so A5's cost stops being an estimate.
Times N optimizer steps on the actual pruned model + real minitrain loader.
Writes nothing."""
import sys, time, os
# Repo root, from this file's location (results/protocol_audit/<this>). It was a hard-coded absolute
# path when this ran out of outputs/; deriving it is what lets the script run from a clone.
import os as _os
from pathlib import Path as _Path
REPO = _os.environ.get("ETPU_REPO") or str(_Path(__file__).resolve().parents[2])
sys.path.insert(0,REPO+"/src"); sys.path.insert(0,REPO+"/families/detection/effdet")
import torch, torch_pruning as tp
from int8_pruning.prune.core import seed_everything, count_params, get_importance, progressive_pruning_to_target
from prune import (load_baseline_effdet, wrap_train_bench, get_dataloaders_coco,
                   get_ignored_layers_effdet, RECIPE_FT)

DEV, IMG, NC = "cuda", 384, 90
N_WARM, N_TIME = 10, 60

seed_everything(42)
model = load_baseline_effdet(REPO+"/outputs/models/efficientdet_lite1.pt", NC, IMG).to(DEV)
pruner = tp.pruner.MetaPruner(
    model=model, example_inputs=torch.randn(1,3,IMG,IMG).to(DEV),
    importance=get_importance("magnitude_l2"), pruning_ratio=1.0,
    iterative_steps=400, global_pruning=False, max_pruning_ratio=0.9,
    ignored_layers=get_ignored_layers_effdet(model))
progressive_pruning_to_target(model, pruner, 10, scope=lambda m: m.backbone)
print(f"[model] pruned to {count_params(model):,} params", flush=True)
del pruner
import gc; gc.collect(); torch.cuda.empty_cache()

BS = int(os.environ.get('BS','8'))
train_loader, _, _ = get_dataloaders_coco(REPO+"/data/datasets/coco", BS, 42,
    image_size=IMG, train_subset="minitrain", baseline_name="efficientdet_lite1")
n_batches = len(train_loader)
print(f"[data] minitrain: {n_batches} batches/epoch @ bs={BS}", flush=True)

bench = wrap_train_bench(model).to(DEV); bench.train()
opt = torch.optim.SGD(bench.parameters(), lr=RECIPE_FT["lr"],
                      momentum=0.9, weight_decay=RECIPE_FT["weight_decay"])
it = iter(train_loader); t0 = None
for i in range(N_WARM + N_TIME):
    try: batch = next(it)
    except StopIteration: it = iter(train_loader); batch = next(it)
    images, targets = batch if isinstance(batch,(list,tuple)) and len(batch)==2 else (batch[0],batch[1])
    if isinstance(targets, dict):
        targets = {k:(v.to(DEV) if torch.is_tensor(v) else v) for k,v in targets.items()}
    out = bench(images.to(DEV) if torch.is_tensor(images) else images, targets)
    loss = out["loss"] if isinstance(out, dict) else out
    opt.zero_grad(); loss.backward(); opt.step()
    if i == N_WARM - 1:
        torch.cuda.synchronize(); t0 = time.time()
torch.cuda.synchronize()
sec_per_it = (time.time()-t0)/N_TIME
ep_min = sec_per_it*n_batches/60
print(f"\n{'#'*70}\nMEASURED FINETUNE COST (RTX 4070, effdet_lite1 384px, bs16)")
print(f"{'#'*70}")
print(f"  {sec_per_it*1000:.0f} ms/iter   ->   {ep_min:.1f} min/epoch  ({n_batches} iters)")
for ep in (15, 40):
    h = ep_min*ep/60
    print(f"  {ep:2d} epochs = {h:5.2f} h/run   ->  2 runs {2*h:5.2f} h   4 runs {4*h:5.2f} h")
print(f"  (+ {40/60:.1f} min per mAP eval, {(15//5)+1} evals/run at eval_every=5)")
