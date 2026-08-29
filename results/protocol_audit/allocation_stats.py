#!/usr/bin/env python3
"""Why global ranking x mean-normalised magnitude guts narrow early layers.

Section 1 of docs/PRUNING_HAZARDS.md explains the allocation pathology with four
statistics -- the per-group normalised mean, the raw L2 norms, the fraction of
channels in the long left tail, and the coefficient of variation. None of them
had a committed source; this produces all four.

The claim being checked is that the driver is NOT cross-layer scale (which
`normalizer='mean'` already cancels) but WITHIN-layer skew after normalisation:
a narrow early layer has a long left tail, so one global threshold sweeps it to
the bone before touching the wide late stages.

Allocation only -- no forward pass, no data, no fine-tuning, CPU is fine. Takes
about a minute.

    python3 results/protocol_audit/allocation_stats.py
"""
import json
import os as _os
import statistics
import sys
from pathlib import Path as _Path

REPO = _os.environ.get("ETPU_REPO") or str(_Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO + "/src")
sys.path.insert(0, REPO + "/families/detection/effdet")

import torch
import torch_pruning as tp

from int8_pruning.prune.core import get_importance, seed_everything
from prune import get_ignored_layers_effdet, load_baseline_effdet

BASE = REPO + "/outputs/models/efficientdet_lite1.pt"
IMG, NC = 384, 90
OUT = _os.path.join(REPO, "results", "detection", "allocation_stats.json")


def _raw_variants(group):
    """Several plausible readings of "the raw L2 norm of a layer".

    docs/PRUNING_HAZARDS.md quotes 0.5130 for blocks.1.0 and 0.2689 for
    blocks.5.1 without saying which reduction it used, so emit the candidates
    rather than guess. What the argument needs is only the ORDERING (early >
    late), and every variant below agrees on that.
    """
    w = group[0][0].target.module.weight.detach()
    per_out = w.flatten(1).norm(p=2, dim=1)
    return {
        "mean_per_out_l2": round(float(per_out.mean()), 4),
        "median_per_out_l2": round(float(per_out.median()), 4),
        "whole_tensor_l2_over_sqrt_n": round(float(w.norm(p=2) / w.numel() ** 0.5), 4),
        "mean_abs": round(float(w.abs().mean()), 4),
    }


def group_name(group):
    """Name a dependency group by its root module, as the prune logs do."""
    dep = group[0][0]
    return dep.target.name


def main():
    if not _os.path.exists(BASE):
        sys.exit(f"missing baseline checkpoint: {BASE}\n"
                 "fetch it with scripts/download_and_finetune_models.sh")

    seed_everything(42)
    model = load_baseline_effdet(BASE, NC, IMG).eval()
    example = torch.randn(1, 3, IMG, IMG)

    # Same protocol as families/detection/effdet/prune.py: global ranking, magnitude_l2, normalizer='mean'.
    normalised = get_importance("magnitude_l2")
    raw = tp.importance.MagnitudeImportance(p=2, normalizer=None)

    pruner = tp.pruner.MetaPruner(
        model=model, example_inputs=example, importance=normalised,
        pruning_ratio=1.0, iterative_steps=400, global_pruning=True,
        ignored_layers=get_ignored_layers_effdet(model))

    backbone_mods = {id(m) for m in model.backbone.modules()}

    rows = []
    for group in pruner.DG.get_all_groups(
            ignored_layers=pruner.ignored_layers,
            root_module_types=pruner.root_module_types):
        if id(group[0][0].target.module) not in backbone_mods:
            continue
        n_imp = normalised(group).detach().cpu()
        r_imp = raw(group).detach().cpu()
        vals = [float(v) for v in n_imp]
        if len(vals) < 2:
            continue
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals)   # sample (ddof=1), matching the published CV
        rows.append({
            "group": group_name(group),
            "channels": len(vals),
            "normalised_mean": round(mean, 4),
            "raw_l2_mean": round(float(r_imp.mean()), 4),
            "raw_l2_variants": _raw_variants(group),
            "frac_below_half": round(sum(v < 0.5 for v in vals) / len(vals), 4),
            "cv": round(sd / mean, 4) if mean else None,
        })

    rows.sort(key=lambda r: r["group"])
    means = {r["normalised_mean"] for r in rows}

    def find(prefix):
        return [r for r in rows if prefix in r["group"]]

    out = {
        "produced_by": "results/protocol_audit/allocation_stats.py",
        "what": ("Per-group importance statistics for EfficientDet-Lite1's backbone under "
                 "the published protocol (global ranking, magnitude_l2, torch_pruning's "
                 "default normalizer='mean'). Backs the 'Why' subsection of "
                 "docs/PRUNING_HAZARDS.md section 1. Allocation only, no forward pass."),
        "model": "efficientdet_lite1",
        "protocol": {"importance": "magnitude_l2", "normalizer": "mean",
                     "global_pruning": True, "scope": "backbone"},
        "n_groups": len(rows),
        "normalised_means_observed": sorted(means),
        "reading": ("normalizer='mean' makes every group's mean exactly 1.0, so cross-layer "
                    "scale is already cancelled and cannot be the driver. What differs is the "
                    "WITHIN-group shape: narrow early groups have a long left tail (high "
                    "frac_below_half, high cv), wide late groups have almost none. A single "
                    "global threshold therefore empties one narrow layer before it touches "
                    "the wide ones."),
        "groups": rows,
    }
    _os.makedirs(_os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"{len(rows)} prunable backbone groups")
    print(f"distinct normalised means: {sorted(means)}")
    print(f"\n{'group':<42}{'ch':>5}{'mean':>8}{'rawL2':>9}{'<0.5':>8}{'CV':>8}")
    for r in rows:
        print(f"{r['group']:<42}{r['channels']:>5}{r['normalised_mean']:>8.4f}"
              f"{r['raw_l2_mean']:>9.4f}{r['frac_below_half']*100:>7.1f}%{r['cv']:>8.3f}")

    print("\n-- the statements in docs/PRUNING_HAZARDS.md section 1 'Why' --")
    print(f"  every group's normalised mean is exactly 1.0000 : {means == {1.0}}  (n={len(rows)})")
    for p in ("blocks.1.0.conv_dw", "blocks.5.1.conv_dw", "blocks.5.4.conv_dw"):
        for r in find(p):
            print(f"  {p:<22} raw L2 {r['raw_l2_mean']:.4f}   "
                  f"below-0.5 {r['frac_below_half'] * 100:.1f}%   CV {r['cv']:.4f}")
    early, late = find("blocks.1.0.conv_dw"), find("blocks.5.1.conv_dw")
    if early and late:
        print(f"  raw scale runs early > late under every variant measured: "
              f"{all(early[0]['raw_l2_variants'][k] > late[0]['raw_l2_variants'][k] for k in early[0]['raw_l2_variants'])}")
    print(f"\nwrote {_os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
