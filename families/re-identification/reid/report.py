#!/usr/bin/env python3
"""Post-compile report for the reid_* Edge-TPU embedders: on-chip / off-chip SRAM per model + single-inference CPU int8 latency.

SRAM is the hardware-free predictor of the "fits in ~8 MiB or streams over USB"
question that decides whether embed() is a few ms or tens of ms. For each
reid_*_int8.tflite this:
  * runs `edgetpu_compiler -o <out> <tflite>` and parses its stdout (reusing
    int8_pruning.backends.edgetpu.compiler_log.parse_compiler_stdout) into on-chip / off-chip MiB;
    off-chip > 0 => the model does NOT fully fit the SRAM and pages over USB.
  * times single-inference latency on CPU (int8 tflite via tflite_runtime /
    ai_edge_litert / tf.lite): warmup then N timed invoke()s -> mean/median/std
    ms. This is a proxy — real Coral latency needs the device + libedgetpu.

Usage:
    python families/re-identification/reid/report.py \
        --tflite-dir outputs/tflite_int8 \
        --glob 'reid_*_market1501_int8.tflite' \
        --out-json outputs/pruning_logs/reid_edgetpu_report.json
"""

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from int8_pruning.backends.edgetpu.compiler_log import parse_compiler_stdout

CORAL_SRAM_MIB = 8.0  # Coral USB on-chip parameter cache is ~8 MiB


def _cpu_interpreter(tflite_path):
    """int8 CPU interpreter (never the Edge TPU delegate)."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(tflite_path))
    it.allocate_tensors()
    return it


def measure_cpu_latency(tflite_path, warmup, runs):
    it = _cpu_interpreter(tflite_path)
    inp = it.get_input_details()[0]
    shape, dtype = inp["shape"], inp["dtype"]
    if dtype == np.uint8:
        d = np.random.randint(0, 256, shape, dtype=np.uint8)
    elif dtype == np.int8:
        d = np.random.randint(-128, 128, shape, dtype=np.int8)
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
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "std_ms": float(a.std()), "runs": runs}


def compile_sram(compiler, tflite_path, out_dir):
    """Compile one int8 tflite for the Edge TPU and parse its SRAM usage.

    The Edge TPU is an optional backend, so a missing compiler is a skip, not a
    crash. Checking up front rather than catching FileNotFoundError around the
    subprocess also keeps the empty out_dir from being created: `check=False`
    below only covers a non-zero exit code, never an absent executable.
    """
    if shutil.which(compiler) is None:
        return {"skipped": f"{compiler} not found — Edge TPU compile leg not run"}
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([compiler, "-o", str(out_dir), str(tflite_path)],
                          capture_output=True, text=True, check=False)
    captured = proc.stdout + proc.stderr
    (out_dir / f"{tflite_path.stem}_compile.txt").write_text(captured)
    if proc.returncode != 0:
        return {"error": f"edgetpu_compiler rc={proc.returncode}", "log": captured[-800:]}
    mems = parse_compiler_stdout(captured)
    if not mems:
        return {"error": "no memory lines parsed", "log": captured[-800:]}
    m = mems[0]
    return {"on_chip_mib": round(m.on_chip_mib, 3),
            "off_chip_mib": round(m.off_chip_mib, 3),
            "fits_sram": m.off_chip_mib == 0.0}


def run(args):
    tdir = Path(args.tflite_dir)
    inputs = sorted(tdir.glob(args.glob))
    if not inputs:
        raise SystemExit(f"no tflite matched {args.glob!r} under {tdir}")
    out_dir = (Path(args.out_dir) if args.out_dir else
               Path("outputs/edgetpu") /
               f"reid_report_{datetime.now():%Y%m%d_%H%M%S}")

    rows = []
    for p in inputs:
        row = {"model": p.stem, "size_mib": round(p.stat().st_size / 1024**2, 3)}
        if not args.no_compile:
            row.update(compile_sram(args.edgetpu_compiler, p, out_dir))
        if not args.no_latency:
            try:
                row["cpu_latency"] = measure_cpu_latency(p, args.warmup, args.runs)
            except Exception as e:
                row["cpu_latency"] = {"error": f"{type(e).__name__}: {e}"}
        rows.append(row)

    print(f"\n{'model':<44} {'size':>7} {'on-chip':>9} {'off-chip':>9} {'fits':>5} {'cpu ms':>8}")
    print("-" * 90)
    for r in rows:
        on, off, fits = r.get("on_chip_mib"), r.get("off_chip_mib"), r.get("fits_sram")
        ms = r.get("cpu_latency", {}).get("median_ms")
        print(f"{r['model']:<44} {r['size_mib']:>6.2f}M "
              f"{('%.2f' % on) if on is not None else '-':>9} "
              f"{('%.2f' % off) if off is not None else '-':>9} "
              f"{('yes' if fits else 'NO') if fits is not None else '-':>5} "
              f"{('%.2f' % ms) if ms is not None else '-':>8}")
    for r in rows:
        off = r.get("off_chip_mib")
        if off and off > 0:
            print(f"[warn] {r['model']}: off-chip={off:.2f} MiB > 0 — does NOT fully fit "
                  f"the ~{CORAL_SRAM_MIB:.0f} MiB SRAM; params stream over USB (slow).")

    summary = {"out_dir": str(out_dir), "coral_sram_mib": CORAL_SRAM_MIB, "models": rows}
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"\n[reid-report] wrote {args.out_json}")
    # out_dir exists only when the compile leg actually ran, so do not announce an empty directory.
    if out_dir.is_dir():
        print(f"[reid-report] edgetpu artifacts + compile logs in {out_dir}")
    else:
        print("[reid-report] Edge TPU compile leg skipped — no compiler, no edgetpu/ output")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Stays on the retired onnx2tf dir on purpose: the three backbone comparison rows (osnet_x0_5,
    # osnet_x0_75, mobilenetv2_x1_0) and the e256/simmat variants were never re-converted and exist
    # only there. Pass --tflite-dir outputs/tflite_int8_litert to report the current ladder instead.
    p.add_argument("--tflite-dir", default="outputs/tflite_int8")
    p.add_argument("--glob", default="reid_*_market1501_int8.tflite")
    p.add_argument("--edgetpu-compiler", default="edgetpu_compiler")
    p.add_argument("--out-dir", default=None,
                   help="edgetpu artifacts + compile logs "
                        "(default: outputs/edgetpu/reid_report_<TS>)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--no-compile", action="store_true",
                   help="skip the SRAM report (no edgetpu_compiler available)")
    p.add_argument("--no-latency", action="store_true", help="skip CPU latency")
    p.add_argument("--out-json", default=None)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
