#!/usr/bin/env python3
"""Results table for the line_seg_* families.

Merge every artifact the line-seg pipeline produced into one results table:

  outputs/pruning_logs/<variant>_<imp>_<P>pct.json  params + float held-out metrics
  outputs/tflite_int8/<stem>_int8.size.json         int8 flatbuffer bytes + i/o dtypes
  outputs/pruning_logs/compile_edgetpu_*.log        `edgetpu_compiler -s` reports
  outputs/line_seg_eval_param/results_*.json        upstream int8 / edgetpu accuracy
  <latency json>                                    on-device latency (optional)

The `-s` report is parsed with the UPSTREAM parser (training/common.parse_report),
not a local reimplementation: the acceptance gate is "one subgraph, zero CPU ops",
and both sides must agree on what that means. The compiler prints those lines to
stdout only (the per-model .log holds just the operator table), so the input here
is the run-level log that compile_edgetpu.sh tees, split per `Input model:` block.

⚠ `pruned<P>pct` in this family is a per-layer CHANNEL ratio, not a parameter
ratio — see families/segmentation/line_seg/prune.py. The table prints both.

Usage:
    python families/segmentation/line_seg/report.py --upstream ../MicroROS-Pi5_Coral_TPU/training \
        --out-json outputs/line_seg_eval_param/report.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

CORAL_SRAM_MIB = 8.0


def parse_compile_logs(paths, parse_report):
    """stem -> `edgetpu_compiler -s` report dict, newest log wins."""
    out = {}
    for p in sorted(paths, key=lambda q: q.stat().st_mtime):
        text = p.read_text(errors="replace")
        # Each model's report starts at its own "Input model:" line.
        blocks = re.split(r"(?=^Input model:)", text, flags=re.M)
        for b in blocks:
            m = re.search(r"^Input model:\s*(.+?)\s*$", b, flags=re.M)
            if not m:
                continue
            stem = Path(m.group(1)).name.replace("_int8.tflite", "")
            rep = parse_report(b)
            if rep.get("subgraphs") is not None:
                out[stem] = rep
    return out


def mib(b):
    return None if b is None else round(b / 1024 ** 2, 3)


def collect(args):
    sys.path.insert(0, str(Path(args.upstream).resolve()))
    from common import parse_report

    project = Path(args.project_dir)
    logs = project / "outputs" / "pruning_logs"

    compiles = parse_compile_logs(sorted(logs.glob("compile_edgetpu_*.log")), parse_report)

    prune = {}
    for p in logs.glob("line_seg_*pct.json"):
        d = json.loads(p.read_text())
        prune[Path(d["output_file"]).stem] = d

    sizes = {}
    for d in args.tflite_dir:
        for p in Path(d).glob("line_seg_*_int8.size.json"):
            s = json.loads(p.read_text())
            sizes[s["name"]] = s

    evals = {}
    # One file per variant, the parameter ladder having been swept per variant. The single results.json
    # this used to read belonged to the round-1 channel ladder, withdrawn 2026-08-22 (results/_deleted/).
    for ev in sorted((project / "outputs" / "line_seg_eval_param").glob("results_*.json")):
        for row in json.loads(ev.read_text()):
            evals[row["stem"]] = row

    lat = {}
    if args.latency_json and Path(args.latency_json).is_file():
        for row in json.loads(Path(args.latency_json).read_text()):
            lat[row["stem"]] = row

    # compile_edgetpu_*.log is shared with every family here, so restrict to this one's stems.
    stems = {s for s in set(sizes) | set(compiles) | set(prune)
             if s.startswith("line_seg_")}
    rows = []
    for stem in sorted(stems):
        p, c = prune.get(stem, {}), compiles.get(stem, {})
        s, e = sizes.get(stem, {}), evals.get(stem, {})
        cpu_eval = (e.get("int8_cpu") or {})
        tpu_eval = (e.get("edgetpu") or {})
        rows.append({
            "stem": stem,
            "variant": stem.split("_pruned")[0],
            "channel_ratio_pct": p.get("channel_ratio_pct"),
            "param_reduction_pct": (round(p["param_reduction_pct"], 2)
                                    if "param_reduction_pct" in p else None),
            "params": p.get("post_params"),
            "finetuned": p.get("finetuned"),
            "tflite_mib": s.get("size_mib"),
            "subgraphs": c.get("subgraphs"),
            "cpu_ops": c.get("cpu_ops"),
            "cpu_op_names": c.get("cpu_op_names") or [],
            "total_ops": c.get("total_ops"),
            "on_chip_mib": mib(c.get("on_chip_bytes")),
            "off_chip_mib": mib(c.get("off_chip_bytes")),
            "fits_sram": (None if c.get("off_chip_bytes") is None
                          else c["off_chip_bytes"] == 0),
            "float_val_iou": p.get("val_iou"),
            "float_floor_fire": p.get("floor_fire"),
            "int8_val_iou": cpu_eval.get("val_iou"),
            "int8_floor_fire": cpu_eval.get("floor_fire"),
            "edgetpu_val_iou": tpu_eval.get("val_iou"),
            "edgetpu_floor_fire": tpu_eval.get("floor_fire"),
            "tpu_latency_ms": (lat.get(stem) or {}).get("median"),
        })
    return rows


def _num(v, width, prec=2):
    return f"{v:>{width}.{prec}f}" if isinstance(v, (int, float)) else f"{'-':>{width}}"


def _int(v, width):
    return f"{v:>{width},}" if isinstance(v, int) else f"{'-':>{width}}"


def show(rows):
    hdr = (f"{'model':<44}{'ch%':>5}{'par%':>7}{'params':>11}{'on-chip':>9}"
           f"{'off-chip':>9}{'sub':>4}{'cpu':>4}{'IoU':>8}{'fire':>7}{'ms':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['stem']:<44}"
              f"{_int(r['channel_ratio_pct'], 5)}"
              f"{_num(r['param_reduction_pct'], 7, 1)}"
              f"{_int(r['params'], 11)}"
              f"{_num(r['on_chip_mib'], 9)}"
              f"{_num(r['off_chip_mib'], 9)}"
              f"{_int(r['subgraphs'], 4)}"
              f"{_int(r['cpu_ops'], 4)}"
              f"{_num(r['int8_val_iou'], 8, 4)}"
              f"{_num(r['int8_floor_fire'], 7, 4)}"
              f"{_num(r['tpu_latency_ms'], 8, 3)}")
    bad = [r for r in rows if r["subgraphs"] not in (None, 1) or (r["cpu_ops"] or 0) > 0]
    for r in bad:
        print(f"[FAIL] {r['stem']}: subgraphs={r['subgraphs']} cpu_ops={r['cpu_ops']} "
              f"{r['cpu_op_names']}")
    for r in rows:
        if r["int8_floor_fire"]:
            print(f"[warn] {r['stem']}: floor_fire={r['int8_floor_fire']:.4f} > 0 "
                  f"(seam misfire — the failure mode for this task)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # families/segmentation/line_seg/report.py -> line_seg -> families -> project root
    project_default = Path(__file__).resolve().parents[3]
    p.add_argument("--project-dir", type=Path, default=project_default)
    p.add_argument("--upstream", type=Path, required=True,
                   help="upstream training/ dir (supplies common.parse_report)")
    # First leg stays on the retired onnx2tf dir on purpose: the deep tail (w96 pruned90 magnitude_l2
    # and the 92/95/97/99pct _param rungs) was never re-converted and exists only there. It is what
    # results/line_seg/w96_deep_rungs.json is built on.
    p.add_argument("--tflite-dir", nargs="+",
                   default=[project_default / "outputs" / "tflite_int8"])
    p.add_argument("--latency-json", default=None,
                   help="on-device latency rows: [{stem, median}, ...]")
    p.add_argument("--out-json", default=None)
    args = p.parse_args(argv)

    rows = collect(args)
    show(rows)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(
            {"coral_sram_mib": CORAL_SRAM_MIB,
             "ratio_semantics": "channel ratio per layer, NOT parameter ratio",
             "models": rows}, indent=2))
        print(f"\n[line-seg-report] wrote {args.out_json}")


if __name__ == "__main__":
    main()
