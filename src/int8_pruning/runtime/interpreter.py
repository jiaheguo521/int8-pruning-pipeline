"""TFLite / Edge TPU interpreter factory and input quantization.

Extracted from `families/classification/imagenet_backbones/eval.py`, which four eval scripts were
already importing from — it was the de-facto shared runtime layer living inside
a family-specific file. Moving it here makes that explicit and lets the
classification eval be a plain family script like every other one.

`_edgetpu.tflite` inputs go through pycoral; everything else runs on CPU via
tflite_runtime, falling back to the full TensorFlow build.
"""

import numpy as np


def _cpu_interpreter_cls():
    try:
        from tflite_runtime.interpreter import Interpreter   # the docker bench image
        return Interpreter
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter   # the convert venv
        return Interpreter
    except ImportError:
        import tensorflow as tf                              # neither: older env
        return tf.lite.Interpreter


def make_interpreter(tflite_path):
    p = str(tflite_path)
    if p.endswith("_edgetpu.tflite"):
        from pycoral.utils.edgetpu import make_interpreter as _mk
        it = _mk(p)
    else:
        it = _cpu_interpreter_cls()(model_path=p)
    it.allocate_tensors()
    return it


def _input_quant(inp):
    """(scale, zero_point) from either the flat or the nested quant fields."""
    scale, zp = inp.get("quantization", (0.0, 0))
    if not scale:
        qp = inp.get("quantization_parameters", {})
        scales, zps = qp.get("scales", []), qp.get("zero_points", [])
        scale = float(scales[0]) if len(scales) else 0.0
        zp = int(zps[0]) if len(zps) else 0
    return scale, zp


def quantize_input(arr_hwc, inp):
    """Normalized HWC float -> NHWC tensor in the tflite's input dtype."""
    dtype = inp["dtype"]
    scale, zp = _input_quant(inp)
    if dtype in (np.int8, np.uint8) and scale:
        q = np.round(arr_hwc / scale + zp)
        info = np.iinfo(dtype)
        q = np.clip(q, info.min, info.max)
        return np.expand_dims(q.astype(dtype), 0)
    return np.expand_dims(arr_hwc.astype(dtype), 0)  # float model fallback


def embed_tflite(it, hwc_norm):
    """One normalized HWC image -> dequantized embedding [D].

    Was duplicated byte-for-byte between the Re-ID and CLIP eval paths (the
    Re-ID copy's docstring even said "identical to the CLIP eval path").
    """
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    it.set_tensor(inp["index"], quantize_input(hwc_norm, inp))
    it.invoke()
    emb = it.get_tensor(out["index"]).flatten().astype(np.float32)
    scale, zp = _input_quant(out)
    if scale:
        emb = (emb - zp) * scale
    return emb
