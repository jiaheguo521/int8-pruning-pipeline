#!/usr/bin/env python3
"""Count what each importance criterion has actually been run on.

Section 3 of docs/PRUNING_HAZARDS.md opens with a table of how many rungs and
artifacts exist per criterion. That table was counted by hand over
`outputs/pruning_logs/`, which is gitignored, so a reader could not check it and
a re-count did not reproduce it. This script does the counting and commits the
result.

It reads `outputs/`, so unlike the other scripts here it needs the full working
tree rather than a clone -- `outputs/` holds several GB of pipeline products and
is not in git. That is exactly why its OUTPUT is committed: the JSON is the
checkable artifact, and this file is the method that produced it.

A "rung" is one worker-produced JSON carrying an `importance` field. Evaluation
and report artifacts that happen to live in the same directory (CLIP zero-shot
dumps, Re-ID discriminability reports, sweep roll-ups) are not rungs and are
counted separately.

    python3 results/protocol_audit/criterion_census.py
"""
import collections
import json
import re
import os as _os
import sys
from pathlib import Path as _Path

REPO = _Path(_os.environ.get("ETPU_REPO") or _Path(__file__).resolve().parents[2])
OUT = REPO / "results" / "criterion_census.json"

# Every place a worker has ever written a run log.
LOG_DIRS = [
    ("local", "outputs/pruning_logs"),
    ("cluster", "outputs/_cluster_full_logs"),
    ("cluster", "outputs/_cluster_train2017_logs"),
    ("local", "outputs/_local_oneshot"),
    ("local", "outputs/_attic_backbone_scope"),
]
# Two kinds of .pt, counted separately because they answer different questions: sweep artifacts are
# "was this rung produced at all", recovery artifacts are "was it fine-tuned back". Conflating them
# is what made the old hand-count read 44 for one criterion and 4 for another under one column.
PT_DIRS = [("sweep_pt", "outputs/pytorch_pruned"),
           ("finetuned_pt", "outputs/k1_recovery"),
           ("finetuned_pt", "outputs/pareto_recovery")]
ALL_IMPORTANCES = ["magnitude_l1", "magnitude_l2", "fpgm", "random", "lamp"]


def family_of(rec, path):
    for k in ("family", "task"):
        if rec.get(k):
            return rec[k]
    model = str(rec.get("model") or _Path(path).stem)
    for fam in ("efficientdet", "ssdlite", "line_seg", "reid", "clip", "resnet", "efficientnet"):
        if fam in model:
            return {"efficientdet": "effdet", "clip": "clip_rn50",
                    "resnet": "classification", "efficientnet": "classification"}.get(fam, fam)
    return "unknown"


def main():
    rungs = collections.Counter()
    by_origin = collections.defaultdict(collections.Counter)
    families = collections.defaultdict(set)
    non_rung = []
    scanned = 0
    # The same rung can be recorded twice: once as a 0-epoch pre-recovery eval (the
    # coco-val2017 files) and once as the real 40-epoch run (coco-train2017), under names
    # differing only in the dataset token. Deduplicate on the rung's identity, keep whichever
    # record carries more recovery epochs, and list every dropped file in
    # `superseded_duplicates` so the exclusion is auditable rather than a silent filter.
    seen = {}
    dropped = []

    def rung_key(path, rec):
        stem = re.sub(r"_coco-[a-z0-9]+", "", path.stem)
        return (stem, rec.get("importance") or rec.get("criterion"),
                rec.get("checkpoint_pct"), rec.get("scope"))

    for origin, rel in LOG_DIRS:
        d = REPO / rel
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.json")):
            scanned += 1
            try:
                rec = json.loads(path.read_text())
            except Exception:
                non_rung.append({"file": str(path.relative_to(REPO)), "why": "unparsable"})
                continue
            if not isinstance(rec, dict):
                non_rung.append({"file": str(path.relative_to(REPO)), "why": "not an object"})
                continue
            imp = rec.get("importance") or rec.get("criterion")
            if not imp:
                non_rung.append({"file": str(path.relative_to(REPO)),
                                 "why": "no importance field (eval/report artifact)"})
                continue
            key = rung_key(path, rec)
            epochs = len(rec.get("ft_history") or [])
            prev = seen.get(key)
            if prev is not None:
                # More recovery epochs wins. On a tie the two records are the same run seen twice (every
                # such pair here has bit-identical final mAP, the cluster logs having been copied down
                # into outputs/pruning_logs), so attribute it to the cluster: keeping the local copy would
                # silently reclassify 21 cluster rungs as local.
                better = epochs > prev["epochs"]
                tie_to_cluster = (epochs == prev["epochs"]
                                  and origin == "cluster" and prev["origin"] != "cluster")
                takeover = better or tie_to_cluster
                loser = prev["path"] if takeover else path
                dropped.append({"file": str(loser.relative_to(REPO)),
                                "why": (f"same rung as {key[0]}; kept the "
                                        + ("cluster record (identical result, local copy dropped)"
                                           if tie_to_cluster or (not takeover and epochs == prev["epochs"])
                                           else f"record with {max(epochs, prev['epochs'])} recovery epochs"))})
                if not takeover:
                    continue
                rungs[prev["imp"]] -= 1
                by_origin[prev["imp"]][prev["origin"]] -= 1
            seen[key] = {"path": path, "epochs": epochs, "imp": imp, "origin": origin,
                         "mode": rec.get("prune_mode"), "family": family_of(rec, path)}
            rungs[imp] += 1
            by_origin[imp][origin] += 1
            families[imp].add(family_of(rec, path))

    # `iterative` and `iterative_distill` are the iterative labels (imagenet_backbones/
    # prune.py:660, clip_rn50/prune.py:358, relu_clip/prune.py:426). Every other label is a
    # per-family name for one trajectory per rung, `structured_incremental_param` included --
    # that one is line_seg's parameter-target walk inside a single call
    # (families/segmentation/line_seg/prune.py:323), not an iterative mode.
    ITERATIVE_LABELS = {"iterative", "iterative_distill"}
    modes = collections.defaultdict(collections.Counter)
    no_mode = collections.Counter()
    for rec in seen.values():
        if rec["mode"]:
            modes[rec["family"]][rec["mode"]] += 1
        else:
            no_mode[rec["family"]] += 1
    iterative = {f: {m: n for m, n in c.items() if m in ITERATIVE_LABELS}
                 for f, c in modes.items()}
    iterative = {f: v for f, v in iterative.items() if v}

    pts = collections.defaultdict(collections.Counter)
    n_pt = 0
    for kind, rel in PT_DIRS:
        d = REPO / rel
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.pt")):
            n_pt += 1
            for imp in sorted(ALL_IMPORTANCES, key=len, reverse=True):
                if imp in p.name:
                    pts[kind][imp] += 1
                    break

    table = []
    for imp in ALL_IMPORTANCES:
        table.append({
            "criterion": imp,
            "rungs": rungs.get(imp, 0),
            "rungs_local": by_origin[imp].get("local", 0),
            "rungs_cluster": by_origin[imp].get("cluster", 0),
            "sweep_pt": pts["sweep_pt"].get(imp, 0),
            "finetuned_pt": pts["finetuned_pt"].get(imp, 0),
            "families": sorted(families.get(imp, ())),
        })
    for imp in sorted(set(rungs) - set(ALL_IMPORTANCES)):
        table.append({"criterion": imp, "rungs": rungs[imp],
                      "rungs_local": by_origin[imp].get("local", 0),
                      "rungs_cluster": by_origin[imp].get("cluster", 0),
                      "sweep_pt": pts["sweep_pt"].get(imp, 0),
                      "finetuned_pt": pts["finetuned_pt"].get(imp, 0),
                      "families": sorted(families[imp]),
                      "note": "not in ALL_IMPORTANCES"})

    run = [r for r in table if r["rungs"]]
    out = {
        "produced_by": "results/protocol_audit/criterion_census.py",
        "what": ("How many pruning rungs and local .pt artifacts exist per importance "
                 "criterion, counted over every worker-produced run log. Backs the "
                 "'Five criteria are offered; two have been run' table in "
                 "docs/PRUNING_HAZARDS.md section 3."),
        # PT_DIRS has two entries under "finetuned_pt", and a dict comprehension keyed on the kind silently
        # kept only the last, so outputs/k1_recovery vanished from the record while still being counted.
        "counted_over": {"logs": [rel for _, rel in LOG_DIRS],
                         "checkpoints": {k: [r for kk, r in PT_DIRS if kk == k]
                                         for k, _ in PT_DIRS}},
        "note": ("Source directories are under outputs/, which is gitignored; this JSON is "
                 "the committed record. A rung is a worker log carrying an `importance` "
                 "field. Files without one are eval/report artifacts and are listed in "
                 "`non_rung_artifacts`."),
        "json_files_scanned": scanned,
        "superseded_duplicates": dropped,
        "rungs_total": sum(rungs.values()),
        "prune_modes": {
            "what": ("The mode label each worker wrote on the rung. Backs the statement in "
                     "both READMEs that no ladder here was built with PRUNE_MODE=iterative."),
            "iterative_labels": sorted(ITERATIVE_LABELS),
            "rungs_iterative": sum(sum(v.values()) for v in iterative.values()),
            "by_family": {f: dict(sorted(c.items())) for f, c in sorted(modes.items())},
            "rungs_without_a_label": dict(sorted(no_mode.items())),
            "note": ("A rung with no label is not an unknown mode. Those families declare "
                     "`independent` alone in families/*/family.yaml, and scripts/pruning.sh "
                     "refuses a mode a family has not declared (via `manifest prune-mode`), "
                     "so iterative was never reachable for them. `structured_incremental_param` "
                     "is line_seg's parameter-target walk inside one pruning call, not this "
                     "mode; the same warning applies to `iterative_steps` in "
                     "results/detection/pruning_matrix.json."),
        },
        "pt_files_scanned": n_pt,
        "criteria_with_any_run": [r["criterion"] for r in run],
        "criteria_never_run": [r["criterion"] for r in table if not r["rungs"]],
        "table": table,
        "non_rung_artifacts": non_rung,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    print(f"scanned {scanned} json in {len(LOG_DIRS)} log dirs; {sum(rungs.values())} are rungs")
    print(f"{'criterion':<16}{'rungs':>7}{'local':>7}{'cluster':>9}"
          f"{'sweep.pt':>10}{'ft.pt':>7}   families")
    for r in table:
        print(f"{r['criterion']:<16}{r['rungs']:>7}{r['rungs_local']:>7}"
              f"{r['rungs_cluster']:>9}{r['sweep_pt']:>10}{r['finetuned_pt']:>7}"
              f"   {', '.join(r['families'])}")
    print(f"\nprune modes: {out['prune_modes']['rungs_iterative']} rungs iterative; "
          f"{sum(sum(c.values()) for c in modes.values())} labelled, "
          f"{sum(no_mode.values())} unlabelled (families that declare `independent` alone)")
    for f, c in sorted(modes.items()):
        print(f"   {f:<14}{', '.join(f'{m} x{n}' for m, n in sorted(c.items()))}")
    print(f"\nnever run: {', '.join(out['criteria_never_run']) or '(none)'}")
    print(f"non-rung artifacts in those dirs: {len(non_rung)}")
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
