"""Walking a checkpoint ladder: how the rungs are named, and where a job resumes.

A worker prunes one baseline to several ratios. There are two ways to do that,
and they produce DIFFERENT models at the same nominal ratio:

  independent   each rung is pruned from the dense baseline on its own, then
                recovered. This is the PruningBench protocol and the default
                everywhere in this repo.
  iterative     ONE trajectory: prune to the smallest ratio, recover, save, then
                keep pruning the RECOVERED model to the next ratio. LTH-style
                iterative magnitude pruning (Frankle & Carbin, ICLR 2019), which
                the literature reports recovers high ratios better -- not a claim
                this repo has tested.

`mode_suffix` is what keeps the two from colliding on disk, and `resume_point`
is the ladder-walk half of iterative mode. `imagenet_backbones`, `clip_rn50` and
`relu_clip` implement iterative; the rest are independent-only and never need either.
Every ladder committed under `results/` was run in independent mode -- the trajectory's
mechanism is tested, its effect on retained accuracy is not.
That is why the per-rung body (prune -> recover -> save -> eval) is deliberately
NOT abstracted here: the recovery objective, the evaluator and the log schema are
family property, and a callback soup would hide the parts a reader must check.

Three contracts a worker adding iterative mode has to honour. None of them is
enforceable from here, and getting any of them wrong fails SILENTLY:

1. Fixed dense reference. Pass `initial_params=<dense count>` to
   `progressive_pruning_to_target`. Without it the target is measured against the
   CURRENT model, so entering the 40% rung on a model already at 30% leaves
   0.7 x 0.6 = 42% of dense -- a 58% cut reported as 40%.
2. Ascending order. `progressive_pruning_to_target` loops `while count > target`,
   so a rung below the model's current size does nothing at all and silently
   emits a duplicate checkpoint. `resume_point` rejects unsorted input for this
   reason.
3. One pruner for the whole trajectory. `iterative_steps` is then a cumulative
   budget (torch_pruning increments `current_step` on every `step()`), and
   `max_pruning_ratio` stays anchored to the trajectory-start widths. Rebuilding
   the pruner per rung re-anchors that gate to the already-pruned widths, which
   loosens the absolute floor rung by rung.
"""
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

# independent keeps the historical name so every artifact, every `filename_pattern` in
# families/*/family.yaml and all 202 hashes in results/deliverables.sha256 stay valid. `_iter`
# survives the convert patterns for the same reason effdet's `_full` does: `\w` covers `_`.
MODE_SUFFIX = {"independent": "", "iterative": "_iter"}


def mode_suffix(prune_mode: str) -> str:
    """Filename marker for a pruning mode. Empty for the default mode.

    Without this the two modes write the same `.pt` and the same `.json` for the
    same (model, ratio, importance), so an iterative run silently overwrites a
    published independent ladder and nothing in the artifact says which made it.
    """
    try:
        return MODE_SUFFIX[prune_mode]
    except KeyError:
        raise ValueError(
            f"unknown prune_mode {prune_mode!r} (known: {' '.join(sorted(MODE_SUFFIX))})"
        ) from None


def resume_point(checkpoints: Sequence[int],
                 path_for: Callable[[int], Path],
                 force: bool = False) -> Tuple[int, Optional[Path]]:
    """Where an ascending ladder restarts. Returns (start_idx, resume_path).

    `start_idx` advances only over a LEADING run of finished rungs and stops at
    the first gap, so a hole in the middle (10, 20 done, 30 missing, 40 done)
    resumes from 20 and rewrites 40 -- rung 40's file was produced from a
    different trajectory and cannot be trusted to continue this one.

    `resume_path` is the checkpoint to load and keep pruning: the last finished
    rung, or None when starting from the dense baseline. `start_idx ==
    len(checkpoints)` means everything is already present.

    A resumed trajectory is not bit-identical to an uninterrupted one: the pruner
    gets rebuilt on the already-pruned model, so `max_pruning_ratio` re-anchors to
    those widths. Say so wherever the run is reported.
    """
    ladder: List[int] = list(checkpoints)
    if any(b <= a for a, b in zip(ladder, ladder[1:])):
        raise ValueError(f"checkpoints must be strictly ascending, got {ladder}")

    if force:
        return 0, None

    start_idx = 0
    for i, cp in enumerate(ladder):
        if path_for(cp).exists():
            start_idx = i + 1
        else:
            break

    if start_idx == 0 or start_idx >= len(ladder):
        return start_idx, None
    return start_idx, path_for(ladder[start_idx - 1])
