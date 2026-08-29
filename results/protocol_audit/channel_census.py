#!/usr/bin/env python3
"""Channel census for an EfficientDet ladder, and the tile-width fit behind it.

This is the evidence for the "below 64 channels, pruning stops buying time" subsection
of docs/PRUNING_HAZARDS.md section 2. That subsection previously rested entirely on
prose inside results/detection/full_mapping_ladder_lite1.json: the R^2-by-tile-width
table, the 4892 -> 3041 padded-MAC span, and the 7% / 99% / 32% / 59% channel figures
appeared nowhere else in the repository.

What it does, per rung:

  * traces every Conv2d with a forward hook at 384x384 and records
    (in_channels, out_channels, groups, kernel_area, out_h, out_w);
  * recomputes MACs with channel counts rounded UP to a tile width W, for several W.
    A dense conv costs up(cout) * up(cin) * k * oh * ow; a depthwise conv occupies one
    lane per channel, so up(cout) * k * oh * ow;
  * counts how many convs sit below 64 output channels and how many are at a multiple
    of 8.

Then it fits measured latency against the padded MAC series for each W. The peak picks
out the accelerator's systolic array width.

NOTE on the W=1 baseline. Rounding up to 1 is the identity, so `r2_by_width["1"]` is a
raw-MAC fit -- but it is NOT the same number as the ladder's `latency_law.r2`, because
this script recounts MACs by tracing the model while the ladder's `mmacs` column comes
from the cluster logs. Compare the W=64 result against this script's own W=1 value.

Needs torch and the pruned checkpoints under outputs/pytorch_pruned/, which are
gitignored. One forward pass per rung; about a minute each on CPU.

    python results/protocol_audit/channel_census.py
    python results/protocol_audit/channel_census.py --check   # assert published values
"""
import argparse
import json
import math
import os as _os
import sys
from pathlib import Path as _Path

REPO = _os.environ.get("ETPU_REPO") or str(_Path(__file__).resolve().parents[2])
REPO = _Path(REPO)

PT_DIR = REPO / "outputs" / "pytorch_pruned"
DET = REPO / "results" / "detection"

# One entry per ladder. input_size sets the oh/ow of every traced record, so tracing a
# 448 px model at 384 pads the wrong MAC counts. `published` is what --check asserts, and
# is None for a ladder whose result has not reached the documents yet.
MODELS = {
    "lite1": {
        "ladder": DET / "full_mapping_ladder_lite1.json",
        "out": DET / "channel_census_lite1.json",
        "prefix": "efficientdet_lite1_coco-train2017",
        "label": "Lite1",
        "input_size": 384,
        "published": {"peak_width": "64", "r2": 0.992},
    },
    "lite2": {
        "ladder": DET / "full_mapping_ladder_lite2.json",
        "out": DET / "channel_census_lite2.json",
        "prefix": "efficientdet_lite2_coco-train2017",
        "label": "Lite2",
        "input_size": 448,
        "published": {"peak_width": "64", "r2": 0.9829},
    },
}

SUPERSEDES_LITE1 = (
    "The channel figures previously quoted in full_mapping_ladder_lite1.json mixed "
    "three counting bases in one sentence: dense '7% below 64' is out_all (6.6%), "
    "dense '99% at a multiple of 8' is in_all (99.5%), and the 90% rung's '32%' and "
    "'59%' are out_non_depthwise (32.5%, 58.3%). Under one basis -- out_all -- the "
    "series is dense 6.6% / 95.3% and the deepest rung 24.4% / 67.6%.")

WIDTHS = [1, 8, 16, 32, 64, 128, 256]
TILE_HINT = 64          # the value this census is meant to test, not assume


def stem_for(row, prefix):
    """The checkpoint filename a ladder row was built from."""
    if row["importance"] == "dense":
        return f"{prefix}_pruned0pct"
    return (f"{prefix}_pruned{row['target_pct']}pct"
            f"_{row['importance']}_full")


def layer_specs(pt_path, input_size):
    import torch
    model = torch.load(str(pt_path), map_location="cpu", weights_only=False).eval()
    recs, hooks = [], []

    def hook(m, inp, out):
        oh, ow = out.shape[-2:]
        recs.append((m.in_channels, m.out_channels, m.groups,
                     m.kernel_size[0] * m.kernel_size[1], int(oh), int(ow)))

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.randn(1, 3, input_size, input_size))
    for h in hooks:
        h.remove()
    return recs


def padded_macs(specs, w):
    up = lambda c: -(-c // w) * w
    total = 0
    for cin, cout, groups, k, oh, ow in specs:
        if groups == 1:
            total += up(cout) * up(cin) * k * oh * ow
        else:                       # depthwise: one lane per channel, no cin fan-in
            total += up(cout) * k * oh * ow
    return total


# Four ways to count a "channel count", because the figures this census replaces were quoted under
# three of them in one sentence. `out_all` is the headline basis: every conv pads its output lanes,
# depthwise included. The others are recorded so the choice is visible instead of implied.
BASES = {
    "out_all":        lambda s: [cout for _, cout, _, *_ in s],
    "in_all":         lambda s: [cin for cin, *_ in s],
    "in_and_out":     lambda s: [cout for _, cout, _, *_ in s] + [cin for cin, *_ in s],
    "out_non_depthwise": lambda s: [cout for _, cout, g, *_ in s if g == 1],
}


def census(specs):
    out = {"n_convs": len(specs)}
    for name, pick in BASES.items():
        v = pick(specs)
        out[name] = {
            "n": len(v),
            "frac_below_64": round(sum(1 for c in v if c < TILE_HINT) / len(v), 4),
            "frac_multiple_of_8": round(sum(1 for c in v if c % 8 == 0) / len(v), 4),
        }
    # headline, for callers that just want the numbers the document quotes
    out["frac_below_64"] = out["out_all"]["frac_below_64"]
    out["frac_multiple_of_8"] = out["out_all"]["frac_multiple_of_8"]
    return out


def fit(xs, ys):
    """Least squares y = a + b*x, returning (a, b, r2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    sst = sum((y - my) ** 2 for y in ys)
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, 1 - sse / sst


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), default="lite1",
                    help="which EfficientDet ladder to census (default: lite1)")
    ap.add_argument("--check", action="store_true",
                    help="assert the values quoted in docs/PRUNING_HAZARDS.md section 2")
    args = ap.parse_args()

    spec = MODELS[args.model]
    LADDER, OUT = spec["ladder"], spec["out"]
    INPUT_SIZE = spec["input_size"]

    ladder = json.loads(LADDER.read_text())
    rows = ladder["rows"]

    per_rung, missing = [], []
    for row in rows:
        stem = stem_for(row, spec["prefix"])
        pt = PT_DIR / f"{stem}.pt"
        if not pt.is_file():
            missing.append(stem)
            continue
        print(f"  tracing {stem}", flush=True)
        specs = layer_specs(pt, INPUT_SIZE)
        rec = {"stem": stem,
               "importance": row["importance"],
               "target_pct": row["target_pct"],
               "realized_full_pct": row["realized_full_pct"],
               "mmacs_ladder": row["mmacs"],
               "tpu_ms_mean": row["tpu_ms_mean"],
               "padded_mmacs": {str(w): round(padded_macs(specs, w) / 1e6, 1)
                                for w in WIDTHS}}
        rec.update(census(specs))
        per_rung.append(rec)

    if missing:
        print(f"\n[!] {len(missing)} checkpoints not found under {PT_DIR}:", file=sys.stderr)
        for m in missing:
            print(f"      {m}", file=sys.stderr)
        if not per_rung:
            sys.exit(1)

    y = [r["tpu_ms_mean"] for r in per_rung]
    fits = {}
    for w in WIDTHS:
        x = [r["padded_mmacs"][str(w)] for r in per_rung]
        a, b, r2 = fit(x, y)
        fits[str(w)] = {"intercept_ms": round(a, 3),
                        "slope_us_per_mmac": round(b * 1000, 4),
                        "r2": round(r2, 4)}

    best = max(fits, key=lambda w: fits[w]["r2"])

    # Does padding explain the CRITERION split? On lite2 the raw-MAC fit's residuals sort by
    # pruning criterion, and the ladder names channel alignment as the untested candidate.
    # Fitting each arm on both axes is that test, recorded here rather than asserted in prose.
    def _arms(w):
        blk = {}
        for imp in sorted({r["importance"] for r in per_rung} - {"dense"}):
            sub = [r for r in per_rung if r["importance"] == imp]
            if len(sub) < 3:
                continue
            xa, ya = ([r["padded_mmacs"][w] for r in sub],
                      [r["tpu_ms_mean"] for r in sub])
            a2, b2, r22 = fit(xa, ya)
            pooled_a, pooled_b = fits[w]["intercept_ms"], fits[w]["slope_us_per_mmac"] / 1000
            res = [y - (pooled_a + pooled_b * x) for x, y in zip(xa, ya)]
            blk[imp] = {
                "intercept_ms": round(a2, 3),
                "slope_us_per_mmac": round(b2 * 1000, 4),
                "r2": round(r22, 4),
                "n_above_pooled_line": sum(1 for v in res if v > 0),
                "n": len(sub),
                "residual_ms_mean": round(sum(res) / len(res), 3),
                "residual_ms_range": [round(min(res), 3), round(max(res), 3)],
            }
        if len(blk) == 2:
            (p1, p2) = blk.values()
            blk["slope_gap_us_per_mmac"] = round(
                abs(p1["slope_us_per_mmac"] - p2["slope_us_per_mmac"]), 4)
        return blk

    criterion_split = {w: _arms(w) for w in ("1", best) if _arms(w)}
    span = lambda key: (max(r[key] if isinstance(r[key], float) else r["padded_mmacs"][key]
                            for r in per_rung),
                        min(r[key] if isinstance(r[key], float) else r["padded_mmacs"][key]
                            for r in per_rung))
    raw_hi, raw_lo = max(r["mmacs_ladder"] for r in per_rung), min(r["mmacs_ladder"] for r in per_rung)
    p64_hi = max(r["padded_mmacs"]["64"] for r in per_rung)
    p64_lo = min(r["padded_mmacs"]["64"] for r in per_rung)
    lat_hi, lat_lo = max(y), min(y)
    _dense_ms = next(r["tpu_ms_mean"] for r in per_rung if r["importance"] == "dense")
    _deep_ms = min(y)

    dense = next(r for r in per_rung if r["importance"] == "dense")
    deepest = max(per_rung, key=lambda r: r["realized_full_pct"])

    out = {
        "produced_by": "results/protocol_audit/channel_census.py",
        "what": (f"Per-rung Conv2d channel census for the EfficientDet-{spec['label']} ladder, "
                 "and the latency fit against MACs recomputed on channel counts rounded up "
                 "to a tile width. Backs the 'below 64 channels, pruning stops buying time' "
                 "subsection of docs/PRUNING_HAZARDS.md section 2."),
        "measured_against": f"{LADDER.relative_to(REPO)} (latency column)",
        "input_size": INPUT_SIZE,
        "n_rungs": len(per_rung),
        "r2_by_width": {w: fits[w]["r2"] for w in fits},
        "fits": fits,
        "peak_width": int(best),
        # Keyed by tile width. "1" is the raw-MAC axis (this script's recount, not the ladder's latency_law).
        "criterion_split": criterion_split,
        "spans": {
            "raw_mmacs": [raw_hi, raw_lo, round(raw_hi / raw_lo, 2)],
            "padded64_mmacs": [p64_hi, p64_lo, round(p64_hi / p64_lo, 2)],
            # Two conventions, latency not being monotone in MACs: the slowest rung is lamp -10%, above the unpruned one.
            "latency_ms_max_over_min": [lat_hi, lat_lo, round(lat_hi / lat_lo, 2)],
            "latency_ms_dense_over_deepest": [_dense_ms, _deep_ms,
                                              round(_dense_ms / _deep_ms, 2)],
        },
        "reading": (
            f"The fit peaks at W={best}. Raw MACs span {raw_hi/raw_lo:.1f}x across the ladder "
            f"but padded-{best} MACs span only {p64_hi/p64_lo:.1f}x, and latency spans "
            f"{_dense_ms/_deep_ms:.2f}x from the unpruned rung to the deepest one "
            f"({lat_hi/lat_lo:.2f}x between the slowest and fastest rungs -- latency is not "
            f"monotone, lamp -10% sits above unpruned). "
            f"Channels cut below the tile width cost accuracy and buy no "
            f"time. NOTE the W=1 entry is this script's own raw-MAC fit, not the ladder's "
            f"latency_law.r2 -- MACs are recounted here by tracing the model. Compare the peak "
            f"against W=1 within this table. This is a whole-model fit; the layer-level "
            f"corollary has NOT been tested at layer granularity."),
        # A note about prose this file replaced, which only ever existed for lite1; unpacked inline to keep its position.
        **({"supersedes": SUPERSEDES_LITE1} if args.model == "lite1" else {}),
        "channel_census": {
            "basis": "out_all — output channels of every Conv2d, depthwise included",
            "dense": {"n_convs": dense["n_convs"], **dense["out_all"]},
            "deepest_rung": {"stem": deepest["stem"], "n_convs": deepest["n_convs"],
                             **deepest["out_all"]},
        },
        "rows": per_rung,
    }
    if missing:
        out["missing_checkpoints"] = missing

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=True) + "\n")

    print(f"\n{'W':>5} {'R^2':>8} {'intercept':>11} {'us/MMAC':>10}")
    for w in WIDTHS:
        mark = "  <- peak" if str(w) == best else ""
        print(f"{w:>5} {fits[str(w)]['r2']:>8.4f} {fits[str(w)]['intercept_ms']:>10.2f} ms"
              f" {fits[str(w)]['slope_us_per_mmac']:>9.3f}{mark}")
    print(f"\nraw MACs span {raw_hi/raw_lo:.2f}x ({raw_hi:.0f} -> {raw_lo:.0f} MMACs)")
    print(f"padded-{best} span {p64_hi/p64_lo:.2f}x ({p64_hi:.0f} -> {p64_lo:.0f} MMACs)")
    print(f"latency span  {lat_hi/lat_lo:.2f}x ({lat_hi:.3f} -> {lat_lo:.3f} ms)")
    print(f"\ndense       : {dense['n_convs']} convs, "
          f"{100*dense['frac_below_64']:.0f}% below 64 ch, "
          f"{100*dense['frac_multiple_of_8']:.0f}% at a multiple of 8")
    print(f"deepest rung: {deepest['n_convs']} convs, "
          f"{100*deepest['frac_below_64']:.0f}% below 64 ch, "
          f"{100*deepest['frac_multiple_of_8']:.0f}% at a multiple of 8")
    print(f"\nwrote {OUT.relative_to(REPO)}")

    if args.check:
        pub = spec["published"]
        if pub is None:
            # Nothing to assert yet. Say so rather than exiting 0, which would read as "the published result reproduces".
            print(f"\nNOTE - no published tile-width result for {args.model} to assert "
                  f"against.\n       This run measured peak W={best} "
                  f"(R^2 {fits[best]['r2']}). Put those in MODELS['{args.model}']"
                  f"['published']\n       once a document quotes them.")
            return
        bad = []
        if best != pub["peak_width"]:
            bad.append(f"peak width is {best}, published {pub['peak_width']}")
        if abs(fits[pub["peak_width"]]["r2"] - pub["r2"]) > 5e-4:
            bad.append(f"R^2 at W={pub['peak_width']} is "
                       f"{fits[pub['peak_width']]['r2']}, published {pub['r2']}")
        if bad:
            print("\nMISMATCH:", *bad, sep="\n  ")
            sys.exit(1)
        print("\nOK - the tile-width result reproduces.")


if __name__ == "__main__":
    main()
