"""Structured pruning of any nn.Module to a parameter-reduction target, with torch_pruning.

    result = int8_pruning.prune(model, example_inputs, 0.5, ignored_layers=[model.classifier])
    for ratio, result in int8_pruning.prune_trajectory(model, example_inputs, [0.3, 0.5]):
        ...                                     # recover, save; the next ratio continues

A ratio is the fraction of parameters removed, measured over `scope(model)` (the whole
model by default). Pruning is data-free: every criterion is computed from the weights, so
`example_inputs` is only traced.
"""
from int8_pruning.pruning._pruning import PruneResult, count_params, prune, prune_trajectory

__all__ = ["prune", "prune_trajectory", "count_params", "PruneResult"]

# Report the public path, not the private module, in help() and tracebacks.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
