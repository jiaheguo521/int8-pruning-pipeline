#!/usr/bin/env python3
"""Count what has actually been put on the Coral, and check the READMEs still say so.

The first screen of both READMEs publishes how much Edge TPU work this repository
holds. This recomputes it, and fails if either README no longer prints what it counts.

The counting convention, which the READMEs state alongside the numbers:

    A timed artifact is one (model, export path) pair carrying a measured
    Coral latency in a committed file under results/. Re-measuring the same
    artifact on a later date counts once.

So a model converted through both export paths is two artifacts, because it is two
compiles and two bench sessions on the device, and the comparison between them is
what finding 3 rests on. The three synthetic I/O controls are counted apart from the
models: they are probes built to isolate the fixed I/O cost, not pruned networks.

Unlike criterion_census.py this reads only tracked files, so it runs on a clean
clone and sits in tier 1 of scripts/check.sh.

    python3 results/protocol_audit/edgetpu_census.py
"""
import collections
import json
import os as _os
import sys
from pathlib import Path as _Path

REPO = _Path(_os.environ.get("ETPU_REPO") or _Path(__file__).resolve().parents[2])
OUT = REPO / "results" / "edgetpu_census.json"
DOCS = ["README.md", "README.zh-CN.md"]

LITERT = "litert-torch + ai-edge-quantizer"
ONNX2TF = "onnx2tf"

# One entry per file holding a measured Coral latency. `latency_fields` maps a column to
# the export path that produced the artifact it times; several rows carry two columns
# because the same network was converted twice. The path is never inferred from a date:
# each one comes from the file's own metadata, quoted in the comment beside it.
SOURCES = [
    {"file": "line_seg/param_ladder_oncar.json", "rows": "rows", "id": "stem",
     "family": "line_seg",
     # export_path is litert on every row; tpu_ms_mean_onnx2tf keeps the retired path's same-rung number.
     "latency_fields": {"tpu_ms_mean": LITERT, "tpu_ms_mean_onnx2tf": ONNX2TF}},

    {"file": "line_seg/w96_deep_rungs.json", "rows": "rows", "id": "stem",
     "family": "line_seg",
     # Benched 2026-08-20, before the repo switched paths. int8_iou_by_export_path says -92 and -97 "still
     # have onnx2tf builds only"; the -95/-99 re-quantization of 2026-08-28 was scored on CPU, never re-benched.
     "latency_fields": {"tpu_ms_mean": ONNX2TF}},

    {"file": "reid/coral_latency.json", "rows": "backbones", "id": "model",
     "family": "reid",
     # backbones_measured_on: "onnx2tf (retired 2026-08-22). The three backbone rows were NOT re-converted".
     # The two date columns are one artifact benched twice (backbone_reproducibility: "Re-measured
     # 2026-08-22 on the same device"), so they collapse to one entry under the convention above.
     "latency_fields": {"tpu_ms_2026_07_10": ONNX2TF, "tpu_ms_2026_08_22": ONNX2TF}},

    {"file": "reid/coral_latency.json", "rows": "pruning_ladder", "id": "stem",
     "family": "reid",
     # export_path is litert on every row; export_path_migration records onnx2tf's values for the same six rungs.
     "latency_fields": {"tpu_ms_mean": LITERT, "tpu_ms_mean_onnx2tf": ONNX2TF}},

    {"file": "detection/full_mapping_ladder_lite1.json", "rows": "rows",
     "id": ("importance", "target_pct"), "id_prefix": "efficientdet_lite1",
     "family": "effdet", "latency_fields": {"tpu_ms_mean": LITERT}},

    {"file": "detection/full_mapping_ladder_lite2.json", "rows": "rows",
     "id": ("importance", "target_pct"), "id_prefix": "efficientdet_lite2",
     "family": "effdet", "latency_fields": {"tpu_ms_mean": LITERT}},

    {"file": "detection/coral_latency.json", "rows": "rows", "id": "stem",
     "family": "effdet + ssdlite (retired path)",
     # measured_on: "onnx2tf (retired 2026-08-22)". The lite1 rows here are superseded by
     # full_mapping_ladder_lite1.json but are a different artifact, the partially mapped build, benched separately.
     "latency_fields": {"tpu_ms_mean": ONNX2TF}},

    {"file": "clip_rn50/coral_latency.json", "rows": "rows", "id": "config",
     "id_prefix": "clip_rn50", "family": "clip_rn50",
     "latency_fields": {"tpu_ms_mean": LITERT, "tpu_ms_mean_onnx2tf": ONNX2TF}},

    {"file": "detection/io_floor_control.json", "rows": "models", "id": "name",
     "id_prefix": "io_floor_control", "family": "io control", "control": True,
     "latency_fields": {"tpu_ms_mean": LITERT}},
]

# One artifact whose latency lives in a metadata object rather than a row: coral_latency.json's
# ssdlite_litert_migration re-benched the one ssdlite rung still on disk through the current path
# ("20.106 ms (std 0.354, n=100)"). Same model as a row above, different export path, second artifact.
EXTRA = [{"file": "detection/coral_latency.json", "at": "ssdlite_litert_migration",
          "stem": "ssdlite_mobilenetv3_coco-val2017_pruned0pct_magnitude_l2",
          "path": LITERT, "family": "effdet + ssdlite (retired path)"}]


def collect():
    seen, entries = set(), []
    for src in SOURCES:
        doc = json.loads((REPO / "results" / src["file"]).read_text())
        for row in doc[src["rows"]]:
            if isinstance(src["id"], tuple):
                ident = "_".join([src["id_prefix"]] + [str(row[k]) for k in src["id"]])
            else:
                ident = row[src["id"]]
                if src.get("id_prefix"):
                    ident = f"{src['id_prefix']}_{ident}"
            for field, path in src["latency_fields"].items():
                if row.get(field) is None:
                    continue
                key = (ident, path)
                if key in seen:      # a re-measurement of the same artifact
                    continue
                seen.add(key)
                entries.append({"model": ident, "export_path": path,
                                "family": src["family"], "control": src.get("control", False),
                                "source": f"results/{src['file']}#{src['rows']}"})
    for e in EXTRA:
        key = (e["stem"], e["path"])
        assert key not in seen, f"{key} already counted from a row"
        seen.add(key)
        entries.append({"model": e["stem"], "export_path": e["path"],
                        "family": e["family"], "control": False,
                        "source": f"results/{e['file']}#{e['at']}"})
    return entries


def main():
    entries = collect()
    models = {e["model"] for e in entries if not e["control"]}
    controls = {e["model"] for e in entries if e["control"]}
    by_path = collections.Counter(e["export_path"] for e in entries)

    table = []
    for fam in sorted({e["family"] for e in entries}):
        rows = [e for e in entries if e["family"] == fam]
        table.append({"family": fam, "models": len({e["model"] for e in rows}),
                      "artifacts": len(rows),
                      "export_paths": sorted({e["export_path"] for e in rows})})

    out = {
        "produced_by": "results/protocol_audit/edgetpu_census.py",
        "what": ("How many models and compiled artifacts this repository has timed on a "
                 "Coral USB Accelerator. Backs the Edge TPU half of the scale sentence on "
                 "the first screen of both READMEs."),
        "convention": ("A timed artifact is one (model, export path) pair carrying a measured "
                       "Coral latency in a committed file under results/. Re-measuring the "
                       "same artifact on a later date counts once. A model converted through "
                       "both export paths is two artifacts: two compiles, two bench sessions. "
                       "The synthetic I/O controls are counted apart from the models."),
        "counted_over": sorted({f"results/{s['file']}" for s in SOURCES}),
        "note": ("Every source is tracked in git, so this recounts on a clean clone. Export "
                 "path is taken from each file's own metadata, never inferred from a date; "
                 "the attribution is commented per source in the script."),
        "artifacts_timed": len(entries),
        "models_timed": len(models),
        "io_controls_timed": len(controls),
        "by_export_path": dict(sorted(by_path.items())),
        "by_family": table,
        "entries": sorted(entries, key=lambda e: (e["family"], e["model"], e["export_path"])),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    print(f"{len(models)} models, {len(entries)} compiled artifacts timed on the Coral "
          f"(plus {len(controls)} I/O controls, included in the artifact count)")
    print(f"\n{'family':<34}{'models':>8}{'artifacts':>11}")
    for r in table:
        print(f"{r['family']:<34}{r['models']:>8}{r['artifacts']:>11}")
    print(f"\n{'export path':<34}{'artifacts':>19}")
    for p, n in sorted(by_path.items()):
        print(f"{p:<34}{n:>19}")

    # The failure this file exists to prevent: a ladder lands, the JSON recounts, the first screen does not.
    bad = []
    for doc in DOCS:
        text = (REPO / doc).read_text()
        for label, n in (("models_timed", len(models)), ("artifacts_timed", len(entries))):
            if f"**{n}**" not in text:
                bad.append(f"{doc} does not print {label} = {n}")
    print()
    if bad:
        print("STALE — the documents no longer print what this census counts:")
        for b in bad:
            print("   " + b)
        return 1
    print(f"both READMEs print {len(models)} and {len(entries)}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
