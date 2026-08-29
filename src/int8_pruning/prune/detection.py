"""Recovery fine-tuning helpers shared by the detection families.

`build_ft_scheduler` was defined in the effdet worker and imported by the
ssdlite one — a cross-family import that is exactly the coupling the
family-per-directory layout removes. It is detection-shaped (PruningBench
recipe keys), so it sits here rather than in `core`.
"""

import torch.optim as optim


def build_ft_scheduler(optimizer, total_epochs, recipe):
    """Build the LR scheduler for recovery FT.

    Two modes (`recipe['scheduler']`):
      - 'cosine'    : optional N-epoch linear warmup -> cosine decay to 0.
      - 'multistep' : optional N-epoch linear warmup -> step decays at the
                       given milestones with γ = recipe['lr_decay_rate'].

    For both modes, milestones that fall past `total_epochs` are dropped
    (so a 30-ep FT with default [60,80] silently behaves like cosine /
    no-decay rather than crashing).
    """
    warmup_ep = max(0, int(recipe.get("warmup_epochs", 0)))
    warmup_ep = min(warmup_ep, max(0, total_epochs - 1))  # need >=1 ep post-warmup

    sched_name = recipe["scheduler"]
    if sched_name == "cosine":
        main = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_ep), eta_min=0.0)
        sched_desc = f"cosine T_max={total_epochs - warmup_ep}"
    elif sched_name == "multistep":
        ms = [m for m in recipe["lr_decay_epochs"] if m < total_epochs]
        if not ms:
            ms = [max(1, total_epochs - 1)]
        # MultiStepLR sees epoch counted AFTER warmup, so shift milestones.
        ms_shifted = [max(1, m - warmup_ep) for m in ms]
        main = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=ms_shifted, gamma=recipe["lr_decay_rate"])
        sched_desc = f"multistep {ms} γ={recipe['lr_decay_rate']}"
    else:
        raise ValueError(f"Unknown scheduler='{sched_name}'. Use 'cosine' or 'multistep'.")

    if warmup_ep == 0:
        return main, sched_desc

    start_factor = float(recipe.get("warmup_start_factor", 0.01))
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0,
        total_iters=warmup_ep)
    chained = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, main], milestones=[warmup_ep])
    return chained, f"warmup({warmup_ep}ep, start={start_factor}) -> {sched_desc}"
