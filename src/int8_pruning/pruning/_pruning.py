"""One MetaPruner, built to reach 100% over `iterative_steps`, stepped until each target
ratio is met; the target is therefore met to within one step."""
import warnings
from typing import NamedTuple

import torch_pruning as tp

__all__ = ["prune", "prune_trajectory", "count_params", "PruneResult"]

_IMPORTANCES = ["magnitude_l1", "magnitude_l2", "fpgm", "random", "lamp"]


class PruneResult(NamedTuple):
    """What `prune` and each step of `prune_trajectory` reached.

    Attributes
    ----------
    params : int
        Parameters left in the whole model.
    actual_ratio : float
        Fraction removed from `scope(model)`, relative to `dense_params` when given.
    steps : int
        Pruner steps taken.
    """
    params: int
    actual_ratio: float
    steps: int


def count_params(module):
    """Number of parameters in `module`, the unit every ratio is measured in."""
    return sum(p.numel() for p in module.parameters())


def prune(model, example_inputs, ratio, *, importance="magnitude_l2", ignored_layers=(),
          scope=None, traced=None, dense_params=None, **pruner_kwargs):
    """Remove `ratio` of the parameters of `scope(model)` from `model`, in place.

    Parameters
    ----------
    model : torch.nn.Module
    example_inputs : torch.Tensor or tuple of torch.Tensor
        Only traced; the values are not used.
    ratio : float
        Fraction of the parameters of `scope(model)` to remove.
    importance : str or torch_pruning importance, default "magnitude_l2"
        "magnitude_l1", "magnitude_l2", "fpgm", "random", "lamp", or a data-free
        torch_pruning importance object (not a gradient-based one).
    ignored_layers : sequence of torch.nn.Module, default ()
        Modules left untouched, e.g. the classifier head.
    scope : callable, optional
        ``model -> module`` the ratio is measured over; default the whole model.
    traced : torch.nn.Module, optional
        A wrapper to trace instead of `model`, sharing its submodules.
    dense_params : int, optional
        Count to measure the ratio against; default the current `scope(model)`.
    **pruner_kwargs
        Passed to torch_pruning's MetaPruner, e.g. global_pruning (default False),
        max_pruning_ratio (0.9), iterative_steps (200), round_to.

    Returns
    -------
    PruneResult

    Raises
    ------
    ValueError
        If `importance` is an unknown name.
    """
    return next(prune_trajectory(
        model, example_inputs, [ratio], importance=importance, ignored_layers=ignored_layers,
        scope=scope, traced=traced, dense_params=dense_params, **pruner_kwargs))[1]


def prune_trajectory(model, example_inputs, ratios, *, importance="magnitude_l2",
                     ignored_layers=(), scope=None, traced=None, dense_params=None,
                     **pruner_kwargs):
    """Prune `model` in place to each of `ratios` in turn, yielding after each.

    Every ratio is measured against the dense model, and one pruner serves the whole
    trajectory, so `iterative_steps` is a budget for all of it.

    Parameters
    ----------
    ratios : sequence of float
        Strictly ascending fractions of the dense parameter count to remove.
    dense_params : int, optional
        The dense count over `scope(model)`, for resuming an interrupted trajectory.
    model, example_inputs, importance, ignored_layers, scope, traced, **pruner_kwargs
        As in `prune`.

    Yields
    ------
    ratio : float
    info : PruneResult

    Raises
    ------
    ValueError
        If `ratios` is not strictly ascending; raised by the call, before any `next()`.

    Examples
    --------
    >>> for ratio, info in prune_trajectory(model, example, [0.3, 0.5, 0.7]):
    ...     finetune(model, loader, 10, optimizer=torch.optim.SGD(model.parameters(), 0.01))
    ...     torch.save(model, f"model_{ratio}.pt")  # the next ratio continues from here
    """
    ratios = list(ratios)
    if any(b <= a for a, b in zip(ratios, ratios[1:])):
        raise ValueError(f"ratios must be strictly ascending, got {ratios}")
    measured = model if scope is None else scope(model)
    dense = count_params(measured) if dense_params is None else dense_params
    pruner = _pruner(model, example_inputs, importance, ignored_layers, traced, pruner_kwargs)
    return _walk(model, pruner, ratios, scope, dense)


def _importance(name):
    """The criterion named `name`; a torch_pruning importance object is returned as is."""
    if not isinstance(name, str):
        return name
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
    raise ValueError(f"Unknown importance: {name}. Available: {_IMPORTANCES}")


# What the pruner gets unless the caller says otherwise; any other MetaPruner argument
# passes through untouched.
_DEFAULTS = {"global_pruning": False, "max_pruning_ratio": 0.9, "iterative_steps": 200}

def _pruner(model, example_inputs, importance, ignored_layers, traced, pruner_kwargs):
    return tp.pruner.MetaPruner(
        model=model if traced is None else traced, example_inputs=example_inputs,
        importance=_importance(importance), pruning_ratio=1.0,
        ignored_layers=list(ignored_layers), **{**_DEFAULTS, **pruner_kwargs})


def _walk(model, pruner, ratios, scope, dense):
    """The generator behind `prune_trajectory`, apart so that the ratios are checked when
    it is called rather than on the first `next()`."""
    for ratio in ratios:
        steps, actual = _prune_to(model, pruner, ratio, scope, dense)
        yield ratio, PruneResult(params=count_params(model), actual_ratio=actual, steps=steps)


def _prune_to(model, pruner, ratio, scope, dense):
    """Step `pruner` until `ratio` of `dense` parameters is gone from `scope(model)`.

    Returns `(steps, actual_ratio)`.
    """
    measured = (lambda m: m) if scope is None else scope
    target = dense * (1 - ratio)

    n_steps = 0
    last_progress = 0
    while count_params(measured(model)) > target:
        if pruner.current_step >= pruner.iterative_steps:
            actual = 1 - count_params(measured(model)) / dense
            stalled = n_steps - last_progress
            if stalled >= 25:
                # `max_pruning_ratio` is a PER-GROUP pre-step admission gate in torch_pruning's
                # base pruner, so once every group sits at its floor the remaining steps are
                # no-ops. Spending more of them cannot help; raising the cap can.
                warnings.warn(f"max_pruning_ratio="
                              f"{getattr(pruner, 'max_pruning_ratio', '?')} saturated at "
                              f"{actual:.1%} < target {ratio:.1%} -- nothing has shrunk "
                              f"in the last {stalled} steps; raise the cap, not the budget")
            else:
                warnings.warn(f"iterative_steps budget exhausted "
                              f"({pruner.iterative_steps}) at {actual:.1%} < target "
                              f"{ratio:.1%} -- still shrinking; raise iterative_steps")
            break
        before = count_params(measured(model))
        pruner.step()
        n_steps += 1
        if count_params(measured(model)) < before:
            last_progress = n_steps

    return n_steps, 1 - count_params(measured(model)) / dense
