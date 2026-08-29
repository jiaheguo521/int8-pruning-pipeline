#!/usr/bin/env python3
"""Single-inference latency of any int8 model on the physical Coral USB Edge TPU (tflite_runtime + libedgetpu delegate, no pycoral).

Runs INSIDE the `edgetpu_ros` docker image (ships tflite_runtime 2.16 +
/usr/lib/x86_64-linux-gnu/libedgetpu.so.1) with the Coral USB passed through:

    docker run --rm --privileged -v /dev/bus/usb:/dev/bus/usb \
        -v <repo>/src/int8_pruning/backends/edgetpu:/src:ro \
        -v <repo>/outputs/edgetpu:/models:ro \
        edgetpu_ros:latest \
        bash -c 'python3 /src/bench.py /models/single_*/reid_*_edgetpu.tflite'

The TPU invoke() is input-independent, so a random tensor of the model's own
dtype/shape is fine. The first invocations load the Edge TPU firmware (the USB
re-enumerates 1a6e:089a -> 18d1:9302); the warmup loop absorbs that one-off cost.
This is the real-device counterpart to families/re-identification/reid/report.py's CPU-latency leg.
"""
import sys
import time

import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate


def bench(path, warmup, runs):
    it = Interpreter(model_path=path,
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
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std()), "min": float(a.min())}


def main(argv):
    warmup, runs = 15, 100
    paths = [a for a in argv if a.endswith(".tflite")]
    if not paths:
        print("usage: bench.py <*_edgetpu.tflite> ...", file=sys.stderr)
        return 1
    print(f"{'model':<50}{'mean':>9}{'median':>9}{'std':>8}{'min':>8}   (ms)")
    print("-" * 92)
    for p in paths:
        try:
            r = bench(p, warmup, runs)
            print(f"{p.split('/')[-1]:<50}{r['mean']:>9.3f}{r['median']:>9.3f}"
                  f"{r['std']:>8.3f}{r['min']:>8.3f}")
        except Exception as e:
            print(f"{p.split('/')[-1]:<50}ERROR: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
