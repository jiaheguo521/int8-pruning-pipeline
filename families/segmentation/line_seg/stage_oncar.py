#!/usr/bin/env python3
"""Stage one line_seg ratio ladder into the shape the car's follow_lane_tpu wants.

    <out>/<variant>_p<NN>/{model_edgetpu.tflite, meta.json}

`meta.json` carries the geometry/normalization the node reads plus the provenance
the on-car analysis needs. Two fields are load-bearing and were the two traps in
the upstream MicroROS-Pi5_Coral_TPU repo's docs/Pruning_OnCar_Round1.md section 2:

  * roi_top — EdgeTPUSeg defaults it to 0.0 (whole frame) and does NOT warn.
  * ratio_semantics — this ladder is a PARAMETER ratio. The round-1 ladder was a
    CHANNEL ratio, where params fall as r^2. Reading a param rung with the channel
    rule misreports the reduction by an order of magnitude, so the semantics ship
    inside the file rather than living in a doc.

mean255/std255 are copied from the pruning log and asserted against the dataset
meta: line_seg_link_r34 uses ImageNet normalization while the other two use 127.5,
and swapping them produces a garbage mask with no error.

Usage:
    python families/segmentation/line_seg/stage_oncar.py line_seg_w96 --out outputs/oncar_ladder
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def newest_edgetpu(stem):
    hits = subprocess.run(
        ["find", os.path.join(ROOT, "outputs/edgetpu"), "-name", f"{stem}_int8_edgetpu.tflite",
         "-printf", "%T@ %p\n"], capture_output=True, text=True).stdout.split()
    return sorted(zip(hits[0::2], hits[1::2]))[-1][1] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", help="line_seg_w96 | line_seg_base128 | line_seg_link_r34")
    ap.add_argument("--out", default="outputs/oncar_ladder")
    ap.add_argument("--importance", default="magnitude_l2")
    ap.add_argument("--latency", default="outputs/line_seg_eval/latency_w96_param.json")
    ap.add_argument("--eval", nargs="*", default=[], help="results*.json from publish.sh")
    args = ap.parse_args()

    dataset_meta = json.load(open(f"{ROOT}/data/datasets/line_seg/{args.variant}/meta.json"))
    lat = {}
    if os.path.exists(os.path.join(ROOT, args.latency)):
        lat = {r["stem"]: r["mean"] for r in json.load(open(os.path.join(ROOT, args.latency)))}
    ev = {}
    for f in args.eval:
        for r in json.load(open(os.path.join(ROOT, f))):
            if r["int8_cpu"]:
                ev[r["stem"]] = r["int8_cpu"]

    out = os.path.join(ROOT, args.out, args.variant)
    os.makedirs(out, exist_ok=True)
    n = 0
    for pct in range(0, 100, 10):
        if pct == 0:
            stem, log = f"{args.variant}_pruned0pct", None
        else:
            stem = f"{args.variant}_pruned{pct}pct_{args.importance}_param"
            log = f"{ROOT}/outputs/pruning_logs/{args.variant}_{args.importance}_{pct}pct_param.json"
        edge = newest_edgetpu(stem)
        if edge is None:
            print(f"  [SKIP] {stem}: not compiled", file=sys.stderr)
            continue

        meta = dict(dataset_meta)
        meta["pruned_from"] = args.variant
        meta["ratio_semantics"] = "param"
        if log:
            d = json.load(open(log))
            assert d["mean255"] == dataset_meta["mean255"], f"{stem}: mean255 disagrees with dataset meta"
            assert d["std255"] == dataset_meta["std255"], f"{stem}: std255 disagrees with dataset meta"
            assert d["input_size"] == dataset_meta["input_size"], f"{stem}: input_size disagrees"
            meta.update(param_target_pct=d["param_target_pct"],
                        realized_param_pct=round(d["realized_param_pct"], 2),
                        params_pre=d["pre_params"], params_post=d["post_params"],
                        importance=d["importance"])
        else:
            meta.update(param_target_pct=0, realized_param_pct=0.0, importance=None)

        meta["source_stem"] = stem
        if stem in lat:
            meta["tpu_ms"] = round(lat[stem], 3)
        if stem in ev:
            meta["int8_val_iou"] = ev[stem]["val_iou"]
            meta["int8_floor_fire"] = ev[stem]["floor_fire"]

        tag = os.path.join(out, f"{args.variant.replace('line_seg_', '')}_p{pct:02d}")
        os.makedirs(tag, exist_ok=True)
        shutil.copy(edge, os.path.join(tag, "model_edgetpu.tflite"))
        with open(os.path.join(tag, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
            fh.write("\n")
        n += 1
        print(f"  ✓ {os.path.basename(tag)}  {meta.get('realized_param_pct')}%  "
              f"{meta.get('tpu_ms', '?')} ms  IoU={meta.get('int8_val_iou', '?')}")
    print(f"{n} rungs -> {out}")


if __name__ == "__main__":
    main()
