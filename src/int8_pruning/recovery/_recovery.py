"""Both entry points build a `step(batch) -> loss` and hand it to one training loop,
`_train`, which steps the schedule per epoch and keeps the best state when asked."""
import logging
import time
import warnings
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

__all__ = ["finetune", "distill", "EpochResult"]

logger = logging.getLogger(__name__)

# A batch loss above this in the first epochs means the learning rate, not a hard batch.
_WARN_EPOCHS, _WARN_LOSS = 5, 50.0


class EpochResult(NamedTuple):
    """One epoch of `finetune` or `distill`.

    Attributes
    ----------
    epoch : int
        Counted from 1.
    train_loss, max_loss : float
        Mean and largest batch loss.
    eval : dict or None
        What `evaluate()` returned, or None in an epoch it did not run.
    lr : float
        Learning rate after the epoch's scheduler step.
    duration_s : float
    """
    epoch: int
    train_loss: float
    max_loss: float
    eval: dict
    lr: float
    duration_s: float


def finetune(model, loader, epochs, *, optimizer, scheduler=None, train_step=None,
             evaluate=None, select=None, eval_every=1, amp=False, device=None,
             label="finetune"):
    """Fine-tune `model` in place for `epochs`.

    Parameters
    ----------
    model : torch.nn.Module
    loader : iterable
        Batches, ``(images, labels)`` for the default CrossEntropy loss.
    epochs : int
    optimizer : torch.optim.Optimizer
        Over `model.parameters()`, built after pruning.
    scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        Stepped once per epoch; None keeps the lr.
    train_step : callable, optional
        ``(model, batch) -> loss``, replacing the default loss.
    evaluate : callable, optional
        ``() -> dict``, run every `eval_every` epochs and on the last one.
    select : str, optional
        Key of `evaluate()`'s dict whose best epoch is restored at the end.
    eval_every : int, default 1
    amp : bool, default False
        Mixed precision under CUDA autocast.
    device : str or torch.device, optional
        Where the default step moves batches; default the model's device.
    label : str, default "finetune"
        Prefix of the progress lines.

    Returns
    -------
    list of EpochResult

    Raises
    ------
    ValueError
        If `optimizer` holds parameters `model` no longer has (built before pruning).
    """
    device = _device(model, device)
    if train_step is None:
        criterion = nn.CrossEntropyLoss()

        def step(batch):
            images, labels = batch
            return criterion(model(images.to(device, non_blocking=True)),
                             labels.to(device, non_blocking=True))
    else:
        def step(batch):
            return train_step(model, batch)

    return _train(model, step, loader, epochs, optimizer, scheduler, evaluate=evaluate,
                  select=select, eval_every=eval_every, amp=amp, device=device, label=label)


def distill(student, teacher, loader, epochs, *, optimizer, scheduler=None, train_step=None,
            loss="cosine", evaluate=None, select=None, eval_every=1, amp=False, device=None,
            label="distill"):
    """Train `student` in place to reproduce a frozen `teacher`; no labels needed.

    Parameters
    ----------
    student : torch.nn.Module
    teacher : torch.nn.Module
        Usually a `copy.deepcopy` taken before pruning; put in eval mode and frozen.
    loader : iterable
        Batches of images, or tuples whose first element is the images.
    train_step : callable, optional
        ``(student, teacher, batch) -> loss``, replacing the default step.
    loss : {"cosine", "simmat"} or callable, default "cosine"
        For the default step: 1 - cos per image, or the batch cosine-similarity matrix
        when the widths differ, or ``(student_out, teacher_out) -> loss``.
    label : str, default "distill"
    epochs, optimizer, scheduler, evaluate, select, eval_every, amp, device
        As in `finetune`.

    Returns
    -------
    list of EpochResult

    Raises
    ------
    ValueError
        If `optimizer` holds parameters `student` no longer has.
    """
    device = _device(student, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if train_step is None:
        loss_fn = {"cosine": _cosine_loss, "simmat": _simmat_loss}.get(loss, loss)

        def step(batch):
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                te = teacher(images).float()
            return loss_fn(student(images).float(), te)
    else:
        def step(batch):
            return train_step(student, teacher, batch)

    return _train(student, step, loader, epochs, optimizer, scheduler, evaluate=evaluate,
                  select=select, eval_every=eval_every, amp=amp, device=device, label=label)


def _device(model, device):
    return torch.device(device) if device is not None else next(model.parameters()).device


def _train(model, step, loader, epochs, optimizer, scheduler, *, evaluate, select,
           eval_every, amp, device, label):
    """The loop behind `finetune` and `distill`; returns the history."""
    if epochs <= 0:
        return []
    own = {id(p) for p in model.parameters()}
    if any(id(p) not in own for g in optimizer.param_groups for p in g["params"]):
        raise ValueError("the optimizer holds parameters the model no longer has: build it "
                         "after pruning, from model.parameters()")
    g = optimizer.param_groups[0]
    logger.info(f"  [{label}] {epochs} ep, {type(optimizer).__name__} "
                f"lr={g.get('initial_lr', g['lr']):g}, "
                + (type(scheduler).__name__ if scheduler is not None else "constant lr")
                + (f", eval_every={eval_every} ep" if evaluate and eval_every > 1 else ""))

    scaler = torch.amp.GradScaler("cuda") if amp else None
    best, best_state, best_epoch = None, None, -1
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, max_loss = _epoch(model, loader, step, optimizer, scaler, epoch,
                                      f"{label} ep{epoch + 1}/{epochs}")
        if scheduler is not None:
            scheduler.step()

        ev, star = None, ""
        if evaluate is not None and ((epoch + 1) % eval_every == 0
                                     or epoch == epochs - 1):
            ev = evaluate()
            if select is not None and (best is None or ev[select] > best):
                best, best_epoch, star = ev[select], epoch + 1, " *"
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        row = EpochResult(epoch=epoch + 1, train_loss=train_loss, max_loss=max_loss, eval=ev,
                          lr=optimizer.param_groups[0]["lr"], duration_s=time.time() - t0)
        history.append(row)

        shown = {"train_loss": train_loss, "max_loss": max_loss, **(ev or {})}
        logger.info(f"    Ep {epoch + 1:3d}/{epochs}  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in shown.items() if isinstance(v, float))
                    + f"  lr={row.lr:.2e}  ({row.duration_s:.0f}s){star}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        logger.info(f"  [{label}] Best {select} : {best:.4f} @ ep {best_epoch}/{len(history)}")
    return history


def _epoch(model, loader, step, optimizer, scaler, epoch, desc):
    """One pass calling `step(batch) -> loss`; returns `(mean, max)` of the batch losses.

    The progress bar counts batches rather than wrapping the loader: `tqdm(loader)` creates
    an extra iterator, which draws from the loader's generator, so a seeded shuffle would
    visit the batches in a different order than a bare loop over the same loader.
    """
    model.train()
    running, worst, n = 0.0, 0.0, 0
    bar = tqdm(total=len(loader), desc=desc, leave=False, dynamic_ncols=True, mininterval=1.0)
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=scaler is not None):
            loss = step(batch)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        value = float(loss.item())
        running, worst, n = running + value, max(worst, value), n + 1
        bar.update(1)
        bar.set_postfix(loss=f"{running / n:.3f}", mx=f"{worst:.1f}")
    bar.close()
    if epoch < _WARN_EPOCHS and worst > _WARN_LOSS:
        warnings.warn(f"epoch {epoch + 1}: max batch loss={worst:.1f} > {_WARN_LOSS:g}; "
                      f"try a smaller lr or a longer warmup")
    return running / max(1, n), worst


def _cosine_loss(se, te):
    """1 - cos(student, teacher). Pins the embedding DIRECTION, which is what a
    cosine-scored model is read by. Requires matching dimensions."""
    se = F.normalize(se, dim=-1)
    te = F.normalize(te, dim=-1)
    return (1.0 - (se * te).sum(-1)).mean()


def _simmat_loss(se, te):
    """MSE between the within-batch cosine-similarity matrices.

    Works across embedding widths (required when the student's differs from the
    teacher's) and constrains cosine structure in the student space directly; a
    learnable projector followed by cosine would constrain it only after the projection.
    """
    ss = F.normalize(se, dim=-1) @ F.normalize(se, dim=-1).t()
    ts = F.normalize(te, dim=-1) @ F.normalize(te, dim=-1).t()
    return F.mse_loss(ss, ts)
