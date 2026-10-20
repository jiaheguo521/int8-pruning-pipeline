"""Running an exported .tflite: one image through it on the CPU or a Coral, and its latency.

    it = int8_pruning.make_interpreter("model_int8.tflite")    # CPU
    outputs = int8_pruning.run_tflite(it, hwc_image)          # dequantized float32
    int8_pruning.latency("model_int8_edgetpu.tflite")         # on a Coral, in ms

The implementation, `_evaluation.py`, imports nothing from this package, so it also runs
by path inside a docker image that ships a TFLite runtime and libedgetpu; its header has
the `docker run` line.
"""
from int8_pruning.evaluation._evaluation import (LatencyResult, latency, make_interpreter,
                                                  run_tflite)

__all__ = ["make_interpreter", "run_tflite", "latency", "LatencyResult"]

# Report the public path, not the private module, in help() and tracebacks.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
