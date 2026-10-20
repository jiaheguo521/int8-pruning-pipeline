"""Exporting a recovered model: to int8 TFLite for the Edge TPU, or to fp32 ONNX.

    calib = int8_pruning.export.image_folder("data/val", 100, 224, mean, std)
    result = int8_pruning.export_int8(model, calib, "model_int8.tflite")
    int8_pruning.compile_edgetpu(result.path)                # -> model_int8_edgetpu.tflite

`export_int8` quantizes through litert-torch + ai-edge-quantizer (the default) or onnx2tf,
calibrated on a set `image_folder` builds; `graph_patches` such as `se_mean_pool` keep
constructs litert-torch lowers badly on the Edge TPU. `export_onnx` writes fp32 ONNX for
the other vendor toolchains. Compiling the int8 file for the Edge TPU is the next stage,
`int8_pruning.compilation`.
"""
from int8_pruning.export._calibration import image_folder, normalize_nhwc
from int8_pruning.export._int8 import ExportInt8Result, export_int8
from int8_pruning.export._onnx import ExportOnnxResult, export_onnx
from int8_pruning.export._patches import adaptive_max_pool, se_mean_pool

__all__ = ["export_int8", "export_onnx", "image_folder", "normalize_nhwc", "se_mean_pool",
           "adaptive_max_pool", "ExportInt8Result", "ExportOnnxResult"]

# Report the public path, not the private module, in help() and tracebacks.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
