"""Structured pruning, recovery and int8 export for Edge TPU deployment, as plain functions.

    import int8_pruning

    int8_pruning.prune(model, example_inputs, 0.5, ignored_layers=[model.classifier])

The public API is what `int8_pruning.` reaches, listed in `__all__`: the functions, and
the stage packages that group them (`pruning`, `recovery`, `export`, `compilation`, then
`evaluation` for running what was exported). Each stage's `__init__.py` only re-exports;
modules whose names start with an underscore are internal and may change.

Every name is imported on first use, so the export toolchains load only when an export
runs. torch_pruning, and with it torch, is the exception, imported here when installed:
its first import draws from Python's `random`, so that draw has to come before the
caller seeds.
"""
import importlib

try:
    import torch_pruning  # noqa: F401
except ImportError:
    pass

_API = {
    "prune": "int8_pruning.pruning",
    "prune_trajectory": "int8_pruning.pruning",
    "count_params": "int8_pruning.pruning",
    "finetune": "int8_pruning.recovery",
    "distill": "int8_pruning.recovery",
    "export_int8": "int8_pruning.export",
    "export_onnx": "int8_pruning.export",
    "compile_edgetpu": "int8_pruning.compilation",
    "make_interpreter": "int8_pruning.evaluation",
    "run_tflite": "int8_pruning.evaluation",
    "latency": "int8_pruning.evaluation",
    "seed_everything": "int8_pruning._seed",
}
# Public modules, one per stage. None shares a name with a function above: importing a
# submodule binds it on the package, which would shadow a function of the same name.
_MODULES = {
    "pruning": "int8_pruning.pruning",
    "recovery": "int8_pruning.recovery",
    "export": "int8_pruning.export",
    "compilation": "int8_pruning.compilation",
    "evaluation": "int8_pruning.evaluation",
}


__all__ = sorted([*_API, *_MODULES])


def __dir__():
    return __all__


def __getattr__(name):
    if name in _MODULES:
        return importlib.import_module(_MODULES[name])
    if name in _API:
        return getattr(importlib.import_module(_API[name]), name)
    raise AttributeError(f"module 'int8_pruning' has no attribute {name!r}")
