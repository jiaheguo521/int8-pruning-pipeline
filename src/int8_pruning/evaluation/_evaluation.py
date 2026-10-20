#!/usr/bin/env python3
"""Imports nothing from the package, so it also runs by path where the package is not
installed, e.g. to time compiled models inside the `edgetpu_ros` docker image (ships
tflite_runtime 2.16 + /usr/lib/x86_64-linux-gnu/libedgetpu.so.1) with the Coral USB
passed through:

    docker run --rm --privileged -v /dev/bus/usb:/dev/bus/usb \
        -v <repo>/src/int8_pruning/evaluation/_evaluation.py:/src/evaluation.py:ro \
        -v <repo>/outputs/edgetpu:/models:ro \
        edgetpu_ros:latest \
        bash -c 'python3 /src/evaluation.py /models/*_edgetpu.tflite'

The TPU invoke() is input-independent, so `latency` feeds a random tensor of the model's
own dtype/shape. The first invocations load the Edge TPU firmware (the USB re-enumerates
1a6e:089a -> 18d1:9302); the warmup loop absorbs that one-off cost.
"""
import sys
import time
from typing import NamedTuple

import numpy as np

__all__ = ["make_interpreter", "run_tflite", "latency", "LatencyResult"]


class LatencyResult(NamedTuple):
    """Edge TPU invoke() times from `latency`, in ms.

    Attributes
    ----------
    mean, median, std, min : float
    """
    mean: float
    median: float
    std: float
    min: float


def make_interpreter(tflite_path):
    """Load `tflite_path` into an allocated interpreter.

    Parameters
    ----------
    tflite_path : str or pathlib.Path
        An ``_edgetpu.tflite`` runs on the Edge TPU; any other .tflite on the CPU.

    Returns
    -------
    Interpreter
        From tflite_runtime, ai_edge_litert or tensorflow, the first one installed.

    Raises
    ------
    ImportError
        If none of the three is installed.
    """
    p = str(tflite_path)
    interpreter, load_delegate = _runtime()
    delegates = [load_delegate("libedgetpu.so.1")] if p.endswith("_edgetpu.tflite") else None
    it = interpreter(model_path=p, experimental_delegates=delegates)
    it.allocate_tensors()
    return it


def run_tflite(it, hwc_norm):
    """Run one normalized image through `it` and return every output as float32.

    Parameters
    ----------
    it : Interpreter
        As returned by `make_interpreter`.
    hwc_norm : numpy.ndarray
        (H, W, 3) float, normalized as for the float model; quantized here.

    Returns
    -------
    list of numpy.ndarray
        Every output, dequantized, in the interpreter's output order.
    """
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], _quantize_input(hwc_norm, inp))
    it.invoke()
    outs = []
    for o in it.get_output_details():
        arr = it.get_tensor(o["index"]).astype(np.float32)
        scale, zp = _quant(o)
        outs.append((arr - zp) * scale if scale else arr)
    return outs


def latency(path, warmup=15, runs=100):
    """Time the Edge TPU invoke() of `path` over `runs` passes after `warmup`.

    Parameters
    ----------
    path : str or pathlib.Path
        An ``_edgetpu.tflite``.
    warmup : int, default 15
        Untimed passes first, which absorb the firmware load.
    runs : int, default 100

    Returns
    -------
    LatencyResult
    """
    interpreter, load_delegate = _runtime()
    it = interpreter(model_path=str(path),
                     experimental_delegates=[load_delegate("libedgetpu.so.1")])
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    shape, dtype = inp["shape"], inp["dtype"]
    if dtype == np.int8:
        d = np.random.randint(-128, 128, shape, dtype=np.int8)
    elif dtype == np.uint8:
        d = np.random.randint(0, 256, shape, dtype=np.uint8)
    else:
        d = np.random.randn(*shape).astype(np.float32)
    for _ in range(warmup):
        it.set_tensor(inp["index"], d)
        it.invoke()
    lats = []
    for _ in range(runs):
        it.set_tensor(inp["index"], d)
        t0 = time.perf_counter()
        it.invoke()
        lats.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(lats)
    return LatencyResult(mean=float(a.mean()), median=float(np.median(a)),
                         std=float(a.std()), min=float(a.min()))


def main(argv):
    warmup, runs = 15, 100
    paths = [a for a in argv if a.endswith(".tflite")]
    if not paths:
        print("usage: evaluation.py <*_edgetpu.tflite> ...", file=sys.stderr)
        return 1
    print(f"{'model':<50}{'mean':>9}{'median':>9}{'std':>8}{'min':>8}   (ms)")
    print("-" * 92)
    for p in paths:
        try:
            r = latency(p, warmup, runs)
            print(f"{p.split('/')[-1]:<50}{r.mean:>9.3f}{r.median:>9.3f}"
                  f"{r.std:>8.3f}{r.min:>8.3f}")
        except Exception as e:
            print(f"{p.split('/')[-1]:<50}ERROR: {type(e).__name__}: {e}")
    return 0


def _runtime():
    """(Interpreter, load_delegate) from whichever runtime this machine has."""
    # Imported here: each machine has one of the three runtimes, tried in this order.
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate   # the docker bench image
        return Interpreter, load_delegate
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate   # the convert venv
        return Interpreter, load_delegate
    except ImportError:
        import tensorflow as tf                                             # neither: older env
        return tf.lite.Interpreter, tf.lite.experimental.load_delegate


def _quant(detail):
    """(scale, zero_point) from either the flat or the nested quant fields."""
    scale, zp = detail.get("quantization", (0.0, 0))
    if not scale:
        qp = detail.get("quantization_parameters", {})
        scales, zps = qp.get("scales", []), qp.get("zero_points", [])
        scale = float(scales[0]) if len(scales) else 0.0
        zp = int(zps[0]) if len(zps) else 0
    return scale, zp


def _quantize_input(arr_hwc, inp):
    """Normalized HWC float -> NHWC tensor in the tflite's input dtype."""
    dtype = inp["dtype"]
    scale, zp = _quant(inp)
    if dtype in (np.int8, np.uint8) and scale:
        q = np.round(arr_hwc / scale + zp)
        info = np.iinfo(dtype)
        q = np.clip(q, info.min, info.max)
        return np.expand_dims(q.astype(dtype), 0)
    return np.expand_dims(arr_hwc.astype(dtype), 0)  # float model fallback


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
