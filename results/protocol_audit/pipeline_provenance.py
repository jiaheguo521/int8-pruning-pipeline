#!/usr/bin/env python3
"""Say, per results file, which branches of the pipeline produced it.

The pipeline is a set of branches -- recovery labelled or distilled, ratio scope
full-model or backbone, five importance criteria, three export paths -- and until
now no file said which ones any given measurement came through. Three partial
censuses existed, each covering one axis and none of them per-file:
capabilities.json says what a family DECLARES, criterion_census.json says which
criterion was actually run, edgetpu_census.json says which export path was timed.

This is the per-file view. Read it as a map from a number to the run behind it.

Where a value is read from the file, `source` says which key it came from. Where
the file records nothing, the cell is null and the file is listed under
`unrecorded`: the gaps are the point, not a defect in the table. Two axes are
derived rather than read, and both are properties the pipeline declares:

  recovery   from capabilities.json, which is generated from families/*/family.yaml.
             The README states the rule this rests on: "Which arm a family uses is
             a property of that family, not a knob."
  scope      from the family, where the file does not say. effdet and ssdlite target
             a BACKBONE parameter reduction; every other family targets the full
             model. Marked "by family" so a derived value is never mistaken for a
             recorded one.

Reads only tracked files, so it runs on a clean clone.

    python3 results/protocol_audit/pipeline_provenance.py
"""
import json
import os as _os
import re
import sys
from pathlib import Path as _Path

REPO = _Path(_os.environ.get("ETPU_REPO") or _Path(__file__).resolve().parents[2])
OUT = REPO / "results" / "pipeline_provenance.json"
CAPS = REPO / "results" / "capabilities.json"
CENSUS = REPO / "results" / "criterion_census.json"

CRITERIA = ["magnitude_l1", "magnitude_l2", "fpgm", "random", "lamp"]
BACKBONE_SCOPE_FAMILIES = {"effdet", "ssdlite"}

# The pipeline products: files holding measurements of models that went through prune ->
# recover -> convert. Analyses of those files are not listed, each one naming its own
# inputs; nor is pruning_matrix.json, an audit record whose rows results/README.md says
# are unrecorded.
FILES = [
    ("detection/full_mapping_ladder_lite1.json", "effdet"),
    ("detection/full_mapping_ladder_lite2.json", "effdet"),
    ("detection/full_scope_grid.json", "effdet"),
    ("detection/coral_latency.json", "effdet + ssdlite"),
    ("detection/compiled_size.json", "effdet"),
    ("detection/target_reachability.json", "effdet"),
    ("line_seg/param_ladder_oncar.json", "line_seg"),
    ("line_seg/w96_deep_rungs.json", "line_seg"),
    ("line_seg/prune_sweep_10_90pct.json", "line_seg"),
    ("reid/coral_latency.json", "reid"),
    ("reid/sweep_results.json", "reid"),
    ("clip_rn50/coral_latency.json", "clip_rn50"),
    ("clip_rn50/zero_shot.json", "clip_rn50"),
    ("classification/litert_stage0.json", "imagenet_backbones"),
]

SCOPE_KEYS = ["scope", "ratio_semantics", "scope_note"]

# Read from STRUCTURED fields only, never from prose. Prose matching produced two false
# positives worth remembering: "random" is also a criterion name and appears in every
# latency protocol ("100 invokes, random input"), and `export.why_not_onnx2tf` names the
# path it explains NOT using. A file stating its path only in prose is therefore reported
# as unrecorded, which is true of it.
CRIT_VALUE_KEYS = ("importance", "criterion")
CRIT_NAME_KEYS = ("stem", "model", "name", "config", "tflite", "pt_source", "variant", "src")
PATH_VALUE_KEYS = ("export_path", "measured_on", "backbones_measured_on")

# "not recorded" and "does not apply" are different answers and the table must not merge them. A file
# whose rows carry no int8/Edge TPU column never reached the export stage, so it has no path to record;
# one whose rows carry no pruning column measures the dense model only, so it has no criterion.
EXPORTED_COL = re.compile(r"int8|tflite|edgetpu|onchip|offchip|tpu_ms|compiled")
PRUNED_COL = re.compile(r"pruned|reduction_pct|target_pct|channel_ratio")


def _rows_of(doc):
    for v in doc.values():
        if isinstance(v, list):
            for r in v:
                if isinstance(r, dict):
                    yield r
        elif isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, list):
                    for r in vv:
                        if isinstance(r, dict):
                            yield r


def criteria_in(doc):
    """Every importance criterion the file names in a field value or a model name."""
    found = set()

    def take(text):
        for c in CRITERIA:
            if c in text:
                found.add(c)

    # Top-level keys of the form "<model>|<criterion>" (target_reachability).
    for k in doc:
        if "|" in k:
            take(k)
    # A file whose rows carry no stem declares the criterion at the top level instead.
    for k in CRIT_VALUE_KEYS:
        if isinstance(doc.get(k), str):
            found.update(c for c in CRITERIA if c == doc[k])
    for row in _rows_of(doc):
        for k, v in row.items():
            if not isinstance(v, str):
                continue
            if k in CRIT_VALUE_KEYS:
                found.update(c for c in CRITERIA if c == v)
            elif k in CRIT_NAME_KEYS:
                take(v)
    return sorted(found)


def scope_in(doc):
    """(label, source) for the ratio the file's `pruned<P>pct` targets.

    Precedence: an explicit scope field, then the row columns, then the file's own
    `what`, then the family. The row rule matters: both full_mapping ladders are
    full-scope and say so only in `what`, so a family default would have labelled
    the two headline ladders backbone.
    """
    for k in SCOPE_KEYS:
        v = doc.get(k)
        if isinstance(v, str) and _scope_of(v):
            return _scope_of(v), k

    fields = set()
    for v in doc.values():
        if isinstance(v, list):
            for r in v:
                if isinstance(r, dict):
                    fields |= set(r)
    if "realized_backbone_pct" in fields or "backbone_pct" in fields:
        return "backbone", "rows"
    if "realized_full_pct" in fields:
        return "full-model", "rows"

    if isinstance(doc.get("what"), str) and _scope_of(doc["what"]):
        return _scope_of(doc["what"]), "what"
    return None, None


def _scope_of(text):
    up = text.upper()
    if "CHANNEL RATIO" in up or "PER-LAYER CHANNEL" in up:
        return "channel"
    if "BACKBONE-SCOPE" in up or "BACKBONE PARAMETER" in up or "BACKBONE-SCOPE" in up:
        return "backbone"
    if "FULL MODEL" in up or "FULL-MODEL" in up or "FULL-SCOPE" in up:
        return "full-model"
    return None


def _path_of(text):
    low = text.lower()
    if "onnx2tf" in low:
        return "onnx2tf"
    if "litert" in low or "ai-edge-quantizer" in low or "ai_edge_quantizer" in low:
        return "litert"
    return None


def paths_in(doc):
    """Which export paths the file names, in a structured field, and where."""
    found, where = set(), []

    def note(path, key):
        if path:
            found.add(path)
            if key not in where:
                where.append(key)

    for k in PATH_VALUE_KEYS:
        if isinstance(doc.get(k), str):
            note(_path_of(doc[k]), k)
    # `export` is a dict whose `path` is the chain; its sibling keys explain what was rejected.
    if isinstance(doc.get("export"), dict) and isinstance(doc["export"].get("path"), str):
        note(_path_of(doc["export"]["path"]), "export.path")
    # A `*_migration` block means the file carries both paths on the same rows.
    for k, v in doc.items():
        if k.endswith("_migration") and isinstance(v, dict):
            note("litert", k)
            note("onnx2tf", k)
    for row in _rows_of(doc):
        for k, v in row.items():
            if k in PATH_VALUE_KEYS and isinstance(v, str):
                note(_path_of(v), k)
            elif k.endswith("_onnx2tf") and v is not None:
                note("onnx2tf", k)
    return sorted(found), where[:2]


def _mentions_pruning(doc):
    """True where a row says it is a pruned model, in a column or in a value."""
    for row in _rows_of(doc):
        for k, v in row.items():
            if PRUNED_COL.search(k):
                return True
            if isinstance(v, str) and re.search(r"pruned|-\s*\d+(\.\d+)?%\s*param", v):
                return True
    return False


def prune_mode_of(family, census, caps):
    """(mode, source) for the rungs behind a file, or (None, None) if it has none.

    Two ways to know, and the source column says which. A rung whose worker wrote a
    mode label settles it directly. A rung with no label is not unknown: effdet and
    ssdlite declare `independent` alone in family.yaml, and scripts/pruning.sh refuses
    a mode a family has not declared, so iterative was never reachable for them.
    """
    fams = [f.strip() for f in family.split("+")]
    labels = set()
    for f in fams:
        labels |= set(census["by_family"].get(f, {}))
    if labels:
        if labels & set(census["iterative_labels"]):
            return "iterative", "logs"
        return "independent", "logs"
    unlabelled = sum(census["rungs_without_a_label"].get(f, 0) for f in fams)
    declared = set()
    for f in fams:
        declared |= set(caps[f]["prune_modes"]) if f in caps else set()
    if unlabelled and declared == {"independent"}:
        return "independent", "declared"
    return None, None


def main():
    caps = {r["family"]: r for r in json.loads(CAPS.read_text())["rows"]}
    census = json.loads(CENSUS.read_text())["prune_modes"]
    rows, unrecorded = [], []

    for rel, family in FILES:
        doc = json.loads((REPO / "results" / rel).read_text())
        crit = criteria_in(doc)
        scope, scope_key = scope_in(doc)
        paths, path_keys = paths_in(doc)

        if scope is None:
            scope = "backbone" if family in BACKBONE_SCOPE_FAMILIES else "full-model"
            scope_key = "by family"
        # A "+" in the family name spans two families, so collect each arm: "labelled/distill" is a real answer.
        arms = {caps[f.strip()]["recovery"] for f in family.split("+") if f.strip() in caps}
        recovery = "/".join(sorted(arms)) if arms else None
        mode, mode_from = prune_mode_of(family, census, caps)

        cols = set()
        for v in doc.values():
            if isinstance(v, list):
                for r in v:
                    if isinstance(r, dict):
                        cols |= set(r)
        exported = any(EXPORTED_COL.search(c) for c in cols)
        # clip_rn50/zero_shot names its pruned row in a value ("-30% params"), not a column, so test both.
        pruned = (bool(crit) or any(EXPORTED_COL and PRUNED_COL.search(c) for c in cols)
                  or _mentions_pruning(doc))
        if not paths and not exported:
            paths = ["n/a"]
        if not crit and not pruned:
            crit = ["n/a"]
        # A file with no rungs has no mode: imagenet_backbones declares both and has run neither.
        if mode is None and crit == ["n/a"]:
            mode, mode_from = "n/a", "no rungs"
        missing = [ax for ax, val in (("criteria", crit), ("export_path", paths)) if not val]
        if missing:
            unrecorded.append({"file": f"results/{rel}", "axes": missing})
        rows.append({
            "file": f"results/{rel}", "family": family,
            "scope": scope, "scope_from": scope_key,
            "criteria": crit or None,
            "recovery": recovery, "recovery_from": "capabilities.json",
            "export_paths": paths or None,
            "export_paths_from": path_keys or None,
            "prune_mode": mode,
            "prune_mode_from": mode_from,
        })

    out = {
        "produced_by": "results/protocol_audit/pipeline_provenance.py",
        "what": ("Per results file, which branches of the pipeline produced it: ratio scope, "
                 "importance criteria, recovery arm and export path. Rendered into the "
                 "'Which part of the pipeline' table in results/README.md."),
        "convention": ("`scope_from` and `export_paths_from` name the key the value was read "
                       "from; \"by family\" means it was derived from the family rather than "
                       "recorded in the file. A null means the file records nothing on that "
                       "axis, and the file is repeated under `unrecorded`."),
        "note": ("PRUNE_MODE comes from criterion_census.json, which tallies the label each "
                 "worker wrote on the rung: 0 of 121 rungs are iterative. Beware the collision: "
                 "`iterative_steps` in pruning_matrix.json and clip_rn50/global_allocation.json "
                 "is torch_pruning's step count WITHIN one pruning call, and line_seg's "
                 "`structured_incremental_param` is its parameter-target walk inside one call. "
                 "Neither is PRUNE_MODE=iterative, which feeds a recovered model back into the "
                 "next round of pruning."),
        "files": len(rows),
        "unrecorded": unrecorded,
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    w = max(len(r["file"]) for r in rows)
    print(f"{'file':<{w}}  {'scope':<11}{'criteria':<26}{'recovery':<10}"
          f"{'mode':<12}export path")
    for r in rows:
        print(f"{r['file']:<{w}}  {r['scope']:<11}"
              f"{','.join(r['criteria'] or ['-']):<26}{r['recovery'] or '-':<10}"
              f"{r['prune_mode'] or '-':<12}{','.join(r['export_paths'] or ['-'])}")
    print(f"\n{len(unrecorded)} files leave an axis unrecorded:")
    for u in unrecorded:
        print(f"   {u['file']}: {', '.join(u['axes'])}")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
