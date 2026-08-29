"""Backfill the const-data byte budget for every int8 tflite in outputs/,
pair each with its .pt parameter count, and emit a cross-family table."""
import sys, json, glob, os, re, datetime
# Repo root, from this file's location (results/protocol_audit/<this>). It was a hard-coded absolute
# path when this ran out of outputs/; deriving it is what lets the script run from a clone.
import os as _os
from pathlib import Path as _Path
REPO = _os.environ.get("ETPU_REPO") or str(_Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO+"/src")
# The pruned checkpoints are whole-model pickles, so unpickling needs the module the class
# was defined in on sys.path. Without this torch.load raises ModuleNotFoundError and every
# one of those rows lands with params=null: 28 of the 46 rows in the first pass.
for _f in sorted(_Path(REPO, "families").glob("*/*")):
    if _f.is_dir():
        sys.path.insert(0, str(_f))
import torch
from int8_pruning.report.tflite_size import measure

os.chdir(REPO)
tflites = sorted(set(glob.glob("outputs/**/*_int8.tflite", recursive=True)))
# outputs/ is gitignored, so in a fresh clone this glob is EMPTY. Without this guard the script
# writes {"models": []} over a 46-row provenance file and exits 0 -- following this directory's own
# README would destroy the record and report success.
if not tflites:
    sys.exit("no *_int8.tflite under outputs/ -- refusing to overwrite "
             "results/tflite_size_audit.json with an empty table. outputs/ is gitignored; "
             "this script only runs where the pipeline has actually been run.")
pts = {os.path.basename(p)[:-3]: p
       for p in glob.glob("outputs/**/*.pt", recursive=True) if os.path.islink(p) is False}

def params_of(stem):
    p = pts.get(stem)
    if not p: return None, None
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        m = obj.module if hasattr(obj, "module") and hasattr(obj.module, "parameters") else obj
        if not hasattr(m, "parameters"): return None, p
        return sum(x.numel() for x in m.parameters()), p
    except Exception as e:
        return None, f"{p} (load failed: {type(e).__name__})"

rows = []
for t in tflites:
    stem = os.path.basename(t)[:-len("_int8.tflite")]
    b = measure(t)
    n, src = params_of(stem)
    fam = re.split(r"_(coco|imagenet|market|cityscapes|tusimple)", stem)[0]
    # packaging_frac is not comparable across export paths: onnx2tf's tensor names alone are 54-64% of
    # an effdet file, the litert path's are 7.3%. Label the row so a reader cannot average the two.
    path = "litert-torch+ai-edge-quantizer" if "/tflite_int8_litert/" in t else "onnx2tf"
    r = {"stem": stem, "family": fam, "tflite": t, "export_path": path, **b.as_dict(),
         "params": n, "pt_source": src}
    if n:
        r["bytes_per_param_raw"] = round(b.file_bytes / n, 4)
        if b.const_bytes:
            r["bytes_per_param_const"] = round(b.const_bytes / n, 4)
    rows.append(r)
    print(f"{stem[:56]:56} {b.file_bytes/1e6:7.3f}MB  pack={100*(b.packaging_frac or 0):5.1f}%  "
          f"params={n if n else '?'}", flush=True)

out = {"generated": datetime.date.today().isoformat(),
       "produced_by": "etpu.report.tflite_size.measure over outputs/**/*_int8.tflite",
       "what": ("Per-file byte budget of every int8 tflite. `const_bytes` is the sum of "
                "non-empty flatbuffer Buffers (real weights/biases/quant constants); "
                "`packaging_frac` is the fraction that is NOT constant data. Raw file "
                "size is only a weight proxy when packaging_frac is small."),
       "models": rows}

OUT = "results/tflite_size_audit.json"
# Fields a human wrote into the committed file that this script does not compute.
# export_path_scope is the only thing telling a reader which export path the rows come
# from, and measured_on / measured_on_note are the notice framing every int8 figure here
# as NOT re-measured on the current path. Dropping them turns a regeneration into a
# silent accuracy claim.
CARRY = ("measured_on", "measured_on_note",
         "export_path_scope", "coverage", "lite2_defect", "identity_notes")
if os.path.exists(OUT):
    prev = json.load(open(OUT))
    for k in CARRY:
        if k in prev and k not in out:
            out[k] = prev[k]
            print(f"[note] carried the hand-written '{k}' field forward unchanged. It states "
                  f"row counts and dates that this run may have just invalidated -- re-read it.")
    # A committed row whose .tflite is gone cannot be re-measured, and dropping it would delete a
    # published measurement. Carry it flagged, so the table stays a record rather than a snapshot.
    fresh = {r.get("tflite") for r in rows}
    carried = [r for r in prev.get("models", ()) if r.get("tflite") not in fresh]
    for r in carried:
        r["source_present"] = False
        r.setdefault("measured", prev.get("generated"))
        r.setdefault("export_path", "litert-torch+ai-edge-quantizer"
                     if "/tflite_int8_litert/" in (r.get("tflite") or "") else "onnx2tf")
    if carried:
        print(f"[note] carrying {len(carried)} row(s) whose source is no longer on disk; "
              f"they are flagged source_present=false and keep their original date:")
        for r in sorted(carried, key=lambda x: x.get("tflite") or ""):
            print(f"          {r.get('measured')}  {r.get('tflite')}")
        out["models"] = carried + out["models"]
    # Per-row annotations a human wrote onto a specific artifact (`defect` is the only
    # one so far; the lite2 --pool-ceil rows carried it until they were rebuilt on
    # 2026-08-27 and it was cleared). They are keyed to the file, not computed, so a
    # regeneration would silently drop them.
    prev_by = {r.get("tflite"): r for r in prev.get("models", ())}
    ROW_CARRY = ("defect",)
    for r in rows:
        o = prev_by.get(r["tflite"])
        for k in ROW_CARRY:
            if o and k in o and k not in r:
                r[k] = o[k]
                print(f"[note] carried hand-written '{k}' onto {r['stem']} -- re-read it if "
                      f"that artifact was rebuilt.")

    # Re-measured rows that moved: a silent byte change means the file was rebuilt.
    for r in rows:
        o = prev_by.get(r["tflite"])
        if o and o.get("file_bytes") != r.get("file_bytes"):
            print(f"[warn] re-measured and CHANGED since {prev.get('generated')}: "
                  f"{r['tflite']}  {o.get('file_bytes')} -> {r.get('file_bytes')} bytes")

os.makedirs("results/detection", exist_ok=True)
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n{'='*104}\nCROSS-FAMILY BYTE BUDGET  (packaging = everything that is not constant data)")
print(f"{'='*104}")
print(f"{'model':<52} {'file MB':>8} {'const MB':>9} {'names MB':>9} {'pack%':>6} {'tensors':>8} {'B/param':>8}")
for r in sorted(rows, key=lambda x: -(x.get("packaging_frac") or 0)):
    print(f"{r['stem'][:52]:<52} {r['file_bytes']/1e6:>8.3f} "
          f"{(r['const_bytes'] or 0)/1e6:>9.3f} {(r['name_bytes'] or 0)/1e6:>9.3f} "
          f"{100*(r['packaging_frac'] or 0):>5.1f}% {r['num_tensors'] or 0:>8} "
          f"{r.get('bytes_per_param_const', float('nan')):>8.2f}")
print("\nwritten: results/tflite_size_audit.json")
