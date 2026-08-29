"""Family-agnostic structured-pruning helpers.

These were duplicated across the six per-family pruning workers — five copies of
`ALL_IMPORTANCES`, five of `seed_everything` / `count_params`, `get_importance`
in two mutually incompatible dialects (if/elif vs dict-of-lambdas), and three
near-identical MACs counters. The canonical versions live here.

What deliberately does NOT live here:

* `PRUNING_PROTOCOL` — each family keeps its own. The `global_pruning` value
  genuinely forks (True for classification/effdet/ssdlite/clip, False for
  reid/line_seg, the latter measured: global allocation leaves the Re-ID stem
  with 1/64 channels and drops recovered cosine 0.766 -> 0.390). Sharing the
  helpers while keeping the protocol per-family is what makes that fork
  visible instead of hiding it in six copy-pasted dicts.
* `get_ignored_layers*` — six genuinely different implementations.
* Eval metrics, data loaders, and line_seg's channel-ratio semantics.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn

ALL_IMPORTANCES = ["magnitude_l1", "magnitude_l2", "fpgm", "random", "lamp"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_workers():
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return 4


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def count_macs(model, input_size):
    """MACs via torch_pruning. `input_size` is an int (square) or (H, W).

    The (H, W) form exists for Re-ID, whose input is 256x128 — the old
    square-only signature is why the Re-ID and CLIP workers had no MACs
    counting at all.
    """
    try:
        import torch_pruning as tp
        # example MUST be on the model's device: on GPU the old CPU-tensor version raised a
        # dtype/device RuntimeError that the bare `except` swallowed, silently logging macs=0.
        device = next(model.parameters()).device
        h, w = (input_size, input_size) if isinstance(input_size, int) else input_size
        example = torch.randn(1, 3, h, w, device=device)
        macs, _ = tp.utils.count_ops_and_params(model, example)
        return macs
    except Exception as e:
        print(f"  [warn] count_macs failed ({type(e).__name__}: {e}); "
              f"recording macs=0", flush=True)
        return 0


def get_importance(name):
    import torch_pruning as tp
    if name == "magnitude_l1":
        return tp.importance.MagnitudeImportance(p=1)
    if name == "magnitude_l2":
        return tp.importance.MagnitudeImportance(p=2)
    if name == "fpgm":
        return tp.importance.FPGMImportance()
    if name == "random":
        return tp.importance.RandomImportance()
    if name == "lamp":
        return tp.importance.LAMPImportance(p=2)
    raise ValueError(f"Unknown importance: {name}. Available: {ALL_IMPORTANCES}")


def extract_layer_structure(module):
    """Lists Conv2d / Linear / BN2d of a module with out_channels."""
    out = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            out.append({"layer": name, "kind": "Conv2d",
                        "out_channels": int(m.out_channels)})
        elif isinstance(m, nn.Linear):
            out.append({"layer": name, "kind": "Linear",
                        "out_channels": int(m.out_features)})
        elif isinstance(m, nn.BatchNorm2d):
            out.append({"layer": name, "kind": "BatchNorm2d",
                        "out_channels": int(m.num_features)})
    return out


def progressive_pruning_to_target(model, pruner, target_pct, scope=None,
                                  initial_params=None):
    """Data-free incremental pruning loop until `target_pct` param reduction.

    All 5 supported importances (magnitude_l1/l2, fpgm, random, lamp) are
    computed from the weights — no forward+backward is needed before
    `pruner.step()`.

    Two orthogonal knobs, both load-bearing:

    * `scope` selects what the ratio is measured against. The detection
      workers pass `lambda m: m.backbone` (BiFPN / heads are in
      ignored_layers and must not count toward the target); everything else
      leaves it None to measure the whole model.
    * `initial_params` expresses `target_pct` against a fixed reference (the
      ORIGINAL dense count) rather than the model's current count. The
      classification worker's iterative/LTH mode passes the dense count so
      "40%" still means 40% of the original even when the model is already
      pruned to 30%. Independent mode leaves it None.

    Returns (n_steps, actual_pct, actual_params) — actual_params is in the
    same scope as the target.
    """
    measured = (lambda m: m) if scope is None else scope
    initial = count_params(measured(model)) if initial_params is None else initial_params
    target = initial * (1 - target_pct / 100)

    n_steps = 0
    last_progress = 0
    while count_params(measured(model)) > target:
        if pruner.current_step >= pruner.iterative_steps:
            actual = 100 * (1 - count_params(measured(model)) / initial)
            stalled = n_steps - last_progress
            if stalled >= 25:
                # Measured 2026-08-17: `max_pruning_ratio` is a PER-GROUP pre-step admission gate
                # (torch_pruning base_pruner.py:351-374), so once every group sits at its floor the remaining
                # steps are no-ops. Spending more of them cannot help; raising the cap can.
                print(f"  [WARN] max_pruning_ratio="
                      f"{getattr(pruner, 'max_pruning_ratio', '?')} saturated at "
                      f"{actual:.1f}% < target {target_pct}% -- nothing has shrunk "
                      f"in the last {stalled} steps; raise the cap, not the budget",
                      flush=True)
            else:
                print(f"  [WARN] iterative_steps budget exhausted "
                      f"({pruner.iterative_steps}) at {actual:.1f}% < target "
                      f"{target_pct}% -- still shrinking; raise iterative_steps",
                      flush=True)
            break
        before = count_params(measured(model))
        pruner.step()
        n_steps += 1
        if count_params(measured(model)) < before:
            last_progress = n_steps

    actual_params = count_params(measured(model))
    actual_pct = 100 * (1 - actual_params / initial)
    return n_steps, actual_pct, actual_params
