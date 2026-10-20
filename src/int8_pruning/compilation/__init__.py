"""Compiling an int8 .tflite for the Edge TPU, with Coral's `edgetpu_compiler`.

    int8_pruning.compile_edgetpu("model_int8.tflite")        # -> model_int8_edgetpu.tflite
    int8_pruning.compile_edgetpu(["a_int8.tflite", "b_int8.tflite"], "out/")  # co-compiled

Needs only the standard library and the compiler, installed from Coral's apt repository
(https://coral.ai/docs/edgetpu/compiler/).
"""
from int8_pruning.compilation._compile import compile_edgetpu

__all__ = ["compile_edgetpu"]

# Report the public path, not the private module, in help() and tracebacks.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
