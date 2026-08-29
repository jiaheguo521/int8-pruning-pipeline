"""Recovery after pruning — the two methods, and the shared loop for one of them.

A structurally pruned network has holes in it, and something has to close them.
This repo uses two methods, and which one a family can use is decided by whether
it has labels, not by preference:

* **dataset** — fine-tune on the labelled training set against the task loss.
  `families/classification/imagenet_backbones/prune.py:finetune_pruningbench` (CrossEntropy, the
  PruningBench ImageNet recipe) and the detection workers' own FT loops are this
  method. It needs labels, and its ceiling is the label set it trains on.

* **distill** — fine-tune against a FROZEN copy of the model as it was *before*
  pruning, on UNLABELLED images. The teacher supplies a dense target per image,
  so no labels are needed and the objective is exactly "reproduce what the dense
  model computed". This is the only method available to the embedding families,
  where there is no label space to train against: `clip_rn50` (open-vocabulary
  zero-shot is image-embedding x text-embedding cosine), `relu_clip` (same), and
  `reid` (a re-identification embedder whose four-source teacher has no shared
  label space at all).

This module holds the distillation loop, which was three near-identical copies
before — same AdamW + cosine schedule, same per-epoch history, differing only in
the loss and in which metric they printed each epoch. Those two are parameters
here; everything else the families share.

What deliberately does NOT live here, matching `int8_pruning.prune.core`:

* `RECIPE_FT` — each family keeps its own. The optimizer settings are shared
  (AdamW 1e-4, wd 1e-4, cosine) but the epoch and batch budgets are not, and
  they are a per-family measurement, not a default.
* The teacher itself. It is always `copy.deepcopy(model)` taken before
  `pruner.step()`, but *where* that copy is taken is part of each worker's
  ordering and is better read in place.
* The dataset-method loops, which share nothing beyond `nn.CrossEntropyLoss` —
  different evaluators, different best-state policies, different schedules.
"""

import time

import torch
import torch.nn.functional as F
import torch.optim as optim


def cosine_loss(se, te):
    """1 - cos(student, teacher). Pins the embedding DIRECTION, which is what a
    cosine-scored model is read by. Requires matching dimensions."""
    se = F.normalize(se, dim=-1)
    te = F.normalize(te, dim=-1)
    return (1.0 - (se * te).sum(-1)).mean()


def simmat_loss(se, te):
    """MSE between the within-batch cosine-similarity matrices.

    Dimension-AGNOSTIC (required when the student's embedding width differs from
    the teacher's) and constrains cosine structure in the student space directly.
    NB: a learnable projector + cosine would be wrong here — preserving cosine
    AFTER a linear map does not preserve cosine in the student space.
    """
    ss = F.normalize(se, dim=-1) @ F.normalize(se, dim=-1).t()
    ts = F.normalize(te, dim=-1) @ F.normalize(te, dim=-1).t()
    return F.mse_loss(ss, ts)


DISTILL_LOSSES = {"cosine": cosine_loss, "simmat": simmat_loss}


def distill_finetune(student, teacher, loader, device, epochs, recipe,
                     loss_fn=cosine_loss, metric_fn=None,
                     metric_name="metric", loss_name="distill"):
    """Recover `student` against a frozen `teacher` on unlabelled images.

    `loader` may yield either bare image tensors or `(images, ...)` tuples; the
    labels of a labelled loader are ignored, which is what lets an ImageFolder
    stand in for an unlabelled corpus.

    `metric_fn` is called once per epoch and returns `(value, log_suffix)`.
    `value` lands in the history under `metric_name`; `log_suffix` is appended to
    the epoch line (e.g. the eval count). Pass None to skip.

    `student.train()` is deliberate: after removing channels the surviving BN
    statistics no longer describe the activations flowing through them, and the
    training-mode forward is what re-estimates them. Confirm before reusing this
    that the family's forward returns the same thing in train mode as in eval —
    it does for an embedder that always returns the embedding, and does not for
    a torchreid model with a training-only classifier head.

    Returns the per-epoch history list.
    """
    if epochs <= 0:
        return []
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = optim.AdamW(student.parameters(), lr=recipe["lr"],
                      weight_decay=recipe["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    history = []
    for ep in range(epochs):
        student.train()
        t0 = time.time()
        running, nb = 0.0, 0
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                te = teacher(images).float()
            se = student(images).float()
            loss = loss_fn(se, te)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        sched.step()
        value, suffix = metric_fn() if metric_fn else (None, "")
        dt = time.time() - t0
        history.append({"epoch": ep + 1, "distill_loss": running / max(1, nb),
                        metric_name: value,
                        "lr": opt.param_groups[0]["lr"], "duration_s": dt})
        mstr = f"  {metric_name}={value:.4f}{suffix}" if value is not None else ""
        print(f"    Ep {ep + 1:3d}/{epochs}  {loss_name}_loss={running / max(1, nb):.4f}"
              f"{mstr}  lr={opt.param_groups[0]['lr']:.2e}  ({dt:.0f}s)", flush=True)
    return history
