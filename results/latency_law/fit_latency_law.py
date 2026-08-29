#!/usr/bin/env python3
"""Refit the Edge TPU weight-transfer latency law, and test it for a step.

This is the script behind section 2 of docs/PRUNING_HAZARDS.md. It exists because
that section used to state four coefficients, an R^2 and an F-statistic with no
committed code to produce them.

Two datasets, one specification:

  A. This repository's own panel  - results/line_seg/param_ladder_oncar.json,
     30 compiled models across three backbones, 0.08-27 MiB of weights, each
     timed with 100 invokes on a Coral USB Edge TPU.

  B. The DSD 2026 paper this repo extends - results/dsd2026_reference/
     resnet50_table.csv, 64 rows (ResNet-50 on CIFAR-100). Optional: if the CSV
     is absent the script runs A only and says so. That file is a transcription
     of someone else's published table, so nothing here depends on it.

The question in both cases is whether latency has a DISCONTINUITY where a model
stops streaming weights. It does not. Latency is linear in off-chip bytes with a
steep slope; adding a step indicator buys nothing, and its coefficient points the
wrong way. The knee in a speed-up plot is `off ~ max(0, size - cap)` composed with
a reciprocal, not a break in latency.

Deliberately stdlib-only (no numpy/scipy): a verification script that needs an
environment built first is not much of a verification.

    python3 results/latency_law/fit_latency_law.py            # print + write JSON
    python3 results/latency_law/fit_latency_law.py --check    # also assert the
                                                              # published values
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PANEL = REPO / "results" / "line_seg" / "param_ladder_oncar.json"
PAPER = REPO / "results" / "dsd2026_reference" / "resnet50_table.csv"
REID = REPO / "results" / "reid" / "coral_latency.json"
OUT = REPO / "results" / "latency_law" / "latency_law_fit.json"


# Least squares + the nested F-test, by hand.
def _solve(a):
    """Gauss-Jordan on an augmented matrix, in place. Returns the solution block."""
    n = len(a)
    w = len(a[0])
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(a[r][c]))
        a[c], a[piv] = a[piv], a[c]
        d = a[c][c]
        if abs(d) < 1e-14:
            raise ValueError("singular design matrix")
        a[c] = [v / d for v in a[c]]
        for r in range(n):
            if r != c and a[r][c]:
                f = a[r][c]
                a[r] = [a[r][k] - f * a[c][k] for k in range(w)]
    return [row[n:] for row in a]


def ols(X, y):
    """Ordinary least squares. Returns (beta, stderr, r2, sse, dof)."""
    n, p = len(y), len(X[0])
    xtx = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    xty = [[sum(X[i][a] * y[i] for i in range(n))] for a in range(p)]

    beta = [r[0] for r in _solve([xtx[i][:] + xty[i] for i in range(p)])]
    yhat = [sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    sse = sum((y[i] - yhat[i]) ** 2 for i in range(n))
    ybar = sum(y) / n
    sst = sum((v - ybar) ** 2 for v in y)
    dof = n - p
    s2 = sse / dof

    eye = [[1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
    inv = _solve([xtx[i][:] + eye[i] for i in range(p)])
    se = [math.sqrt(s2 * inv[i][i]) for i in range(p)]
    return beta, se, 1.0 - sse / sst, sse, dof


def _betacf(a, b, x, itmax=300, eps=3e-16):
    """Continued fraction for the incomplete beta (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def f_sf(f, d1, d2):
    """P(F > f) for an F(d1, d2) variate."""
    if f <= 0:
        return 1.0
    return _betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def nested_f(sse_small, sse_big, dof_big, q=1):
    """F for adding q regressors: (dSSE/q) / (SSE_big/dof_big)."""
    f = ((sse_small - sse_big) / q) / (sse_big / dof_big)
    return f, f_sf(f, q, dof_big)


# The two datasets
def load_panel():
    rows = json.loads(PANEL.read_text())["rows"]
    return [dict(lat=r["tpu_ms_mean"], off=r["offchip_mib"], on=r["onchip_mib"],
                 size=r["onchip_mib"] + r["offchip_mib"], tag=r["variant"]) for r in rows]


def load_reid(): 
    """The six reid_youtu_lite rungs, measured standalone on the same device.

    A second family, a different backbone and a different task -- the point of
    refitting here is that the OFF-CHIP coefficient should come out the same if
    it really is a transfer rate, and the on-chip one should not.
    """
    if not REID.exists():
        return None
    rows = json.loads(REID.read_text())["pruning_ladder"]
    return [dict(lat=r["tpu_ms_mean"], off=r["offchip_mib"], on=r["onchip_mib"],
                 size=r["onchip_mib"] + r["offchip_mib"],
                 tag=f"{r['target_pct']}pct") for r in rows]


def load_paper():
    if not PAPER.exists():
        return None
    out = []
    with PAPER.open() as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            out.append(dict(lat=float(r["tpu_latency_ms"]), off=float(r["offchip_mb"]),
                            on=float(r["onchip_mb"]), size=float(r["size_mb"]),
                            tag=r["importance"]))
    return out


def analyse(rows, name, with_onchip):
    """Fit the transfer law, then test the two ways of defining 'crossed over'."""
    y = [r["lat"] for r in rows]
    base = (lambda r: [1.0, r["off"], r["on"]]) if with_onchip else (lambda r: [1.0, r["off"]])
    names = ["intercept", "offchip", "onchip"] if with_onchip else ["intercept", "offchip"]

    b, se, r2, sse, dof = ols([base(r) for r in rows], y)
    law = {n: round(v, 4) for n, v in zip(names, b)}
    res = {
        "dataset": name, "n": len(y), "law": law, "r2": round(r2, 6),
        "t_stats": {n: round(v / s, 2) for n, s, v in zip(names, se, b)},
        "steps": {},
    }
    if with_onchip:
        res["streamed_byte_cost_ratio"] = round(b[1] / b[2], 3)

    # Two natural readings of "the model crossed the on-chip threshold". A step would show up as a
    # large, significant coefficient of the sign that makes streaming models slower than the law predicts.
    for key, f in (("I[offchip>0]", lambda r: 1.0 if r["off"] > 0 else 0.0),
                   ("I[size<8MB]", lambda r: 1.0 if r["size"] < 8.0 else 0.0)):
        col = [f(r) for r in rows]
        if len(set(col)) < 2:
            res["steps"][key] = {"skipped": "indicator is constant on this dataset"}
            continue
        bs, ses, r2s, sses, dofs = ols([base(r) + [f(r)] for r in rows], y)
        fstat, p = nested_f(sse, sses, dofs)
        res["steps"][key] = {
            "coef_ms": round(bs[-1], 4), "stderr_ms": round(ses[-1], 4),
            "t": round(bs[-1] / ses[-1], 2), "F": round(fstat, 4),
            "df": [1, dofs], "p": round(p, 4), "r2": round(r2s, 6),
            "n_positive": int(sum(col)),
        }
    return res


def render(res):
    print(f"\n=== {res['dataset']}  (n={res['n']}) ===")
    terms = " + ".join(f"{v:.3f}*{k}" for k, v in res["law"].items() if k != "intercept")
    print(f"  lat_ms = {terms} + {res['law']['intercept']:.3f}       R^2 = {res['r2']:.4f}")
    if "streamed_byte_cost_ratio" in res:
        print(f"  a streamed byte costs {res['streamed_byte_cost_ratio']:.2f}x a cached byte")
    for key, s in res["steps"].items():
        if "skipped" in s:
            print(f"  step {key:14} -- {s['skipped']}")
            continue
        verdict = "significant" if s["p"] < 0.05 else "not significant"
        print(f"  step {key:14} coef {s['coef_ms']:+.4f} ms (t={s['t']:+.2f})  "
              f"F(1,{s['df'][1]})={s['F']:.4f}  p={s['p']:.3f}  [{verdict}]")


READING = [
    "",
    "reading: latency is linear in off-chip bytes with a steep slope. No dataset here",
    "  supports a DOWNWARD step where streaming stops, which is what a cliff would be:",
    "",
    "  * I[offchip>0] is the direct test, and it is flat on the two datasets with the",
    "    power to test it (p = 0.80 and p = 0.76; the 6-rung reid panel leaves 2 dof",
    "    and settles nothing at p = 0.11). Its coefficient is negative on all three,",
    "    i.e. streaming models come in slightly FASTER than the continuous law",
    "    predicts -- the opposite sign to a penalty.",
    "",
    "  * I[size<8MB] does come out significant on both large panels, but read the sign:",
    "    POSITIVE, so models that fit are SLOWER than the law predicts, not faster.",
    "    That is the compute floor, not a memory effect. The law has only",
    "    weight-transfer terms, so once off-chip reaches 0 it has nothing left to",
    "    explain the fixed compute cost, and the indicator absorbs it. On this repo's",
    "    panel 11 of the 18 on-chip rows carry under 1 MiB of weights, which is exactly",
    "    the regime results/line_seg/w96_deep_rungs.json shows the law under-predicting",
    "    by 0.4-0.5 ms.",
    "",
    "  Under both definitions the step points away from a cliff. What a threshold",
    "  decides is how many bytes have to stream; it does not decide what a streamed",
    "  byte costs.",
    "",
    "  The reid panel is here for a different reason: it is a second family, a second",
    "  backbone and a second task, measured on the same device. Its off-chip",
    "  coefficient lands at 2.294 against this repo's 2.256 -- 1.7% apart. Its on-chip",
    "  coefficient (0.256 vs 0.915) and intercept (2.887 vs 0.852) do not agree at all,",
    "  which is what should happen: those terms absorb each family's own compute, while",
    "  the off-chip term is a transfer rate and transfers.",
]

PUBLISHED = {
    "this repo (30 compiled models, 3 backbones)": {
        "law": {"intercept": 0.8523, "offchip": 2.2561, "onchip": 0.9148},
        "r2": 0.996332, "step": ("I[offchip>0]", -0.2426, 0.10),
        "quoted_in": ["docs/PRUNING_HAZARDS.md", "README.md", "README.zh-CN.md"],
    },
    "DSD 2026 ResNet-50 table (CIFAR-100)": {
        "law": None, "r2": None, "step": ("I[offchip>0]", -0.030, 0.09),
        "quoted_in": [],
    },
    # n=6 leaves 2 degrees of freedom, so the step test here is noise and is not asserted.
    "reid_youtu_lite ladder (6 rungs, Market-1501)": {
        "law": {"intercept": 2.8867, "offchip": 2.2941, "onchip": 0.2556},
        "r2": 0.997945, "step": None,
        "quoted_in": ["README.md", "README.zh-CN.md"],
    },
}


# The two EfficientDet ladders are not fitted here: each carries its own single-model
# fit, computed by its assembler from rows this script never sees. What is checkable
# from a clean clone is that the documents still print those coefficients. Section 2
# gained a Lite2 subsection on 2026-08-28 whose numbers had no guard at all.
# a is quoted to 2dp and b to 3dp, matching how the documents print them.
EFFDET_LADDERS = {
    "lite1": {
        "path": "results/detection/full_mapping_ladder_lite1.json",
        # The READMEs quote Lite1's floor but not its slope, so only the intercept is required of them.
        "quoted_in": {"docs/PRUNING_HAZARDS.md": ("a", "b"),
                      "README.md": ("a",), "README.zh-CN.md": ("a",)},
    },
    "lite2": {
        "path": "results/detection/full_mapping_ladder_lite2.json",
        "quoted_in": {"docs/PRUNING_HAZARDS.md": ("a", "b"),
                      "README.md": ("a", "b"), "README.zh-CN.md": ("a", "b")},
    },
}


def check_effdet_prose():
    """Assert each document still prints the ladder fits it argues from."""
    bad = []
    for tag, spec in EFFDET_LADDERS.items():
        src = REPO / spec["path"]
        if not src.exists():
            print(f"note: {spec['path']} is absent; skipping its cross-check.")
            continue
        law = json.loads(src.read_text())["findings"]["latency_law"]
        lits = {"a": f'{law["a_ms"]:.2f}', "b": f'{law["b_us_per_MMAC"]:.3f}'}
        for rel, keys in spec["quoted_in"].items():
            doc = REPO / rel
            if not doc.exists():
                print(f"note: {rel} is absent; skipping its cross-check.")
                continue
            text = doc.read_text(encoding="utf-8")
            for k in keys:
                if lits[k] not in text:
                    bad.append(f"{rel} does not print {tag} {k}={lits[k]}")
    return bad


def check(results):
    """Assert the values quoted in docs/PRUNING_HAZARDS.md section 2."""
    bad = check_effdet_prose()
    for res in results:
        exp = PUBLISHED.get(res["dataset"])
        if not exp:
            continue
        if exp["law"]:
            for k, v in exp["law"].items():
                if abs(res["law"][k] - v) > 5e-4:
                    bad.append(f"{res['dataset']}: {k} {res['law'][k]} != published {v}")
            if abs(res["r2"] - exp["r2"]) > 5e-5:
                bad.append(f"{res['dataset']}: R2 {res['r2']} != published {exp['r2']}")
        if not exp["step"]:
            continue
        key, coef, fstat = exp["step"]
        got = res["steps"][key]
        if abs(got["coef_ms"] - coef) > 5e-3:
            bad.append(f"{res['dataset']}: step coef {got['coef_ms']} != published {coef}")
        if abs(got["F"] - fstat) > 5e-2:
            bad.append(f"{res['dataset']}: step F {got['F']} != published {fstat}")
    # Reproducing PUBLISHED says nothing about what the document prints. Until 2026-08-28
    # section 2 carried a pre-refit 2.260 / 0.917 / 0.861 against this dict's
    # 2.256 / 0.915 / 0.852, and --check reported OK throughout because it had never read
    # the file it named. So read it.
    for res in results:
        exp = PUBLISHED.get(res["dataset"])
        if not exp or not exp["law"]:
            continue
        for rel in exp.get("quoted_in", []):
            doc = REPO / rel
            if not doc.exists():
                print(f"note: {rel} is absent; skipping its cross-check.")
                continue
            text = doc.read_text(encoding="utf-8")
            for k, v in exp["law"].items():
                lit = f"{v:.3f}"
                if lit not in text:
                    bad.append(f"{rel} does not print {k}={lit} ({res['dataset']})")

    if bad:
        print("\nFAIL — published values not reproduced:")
        for b in bad:
            print("  " + b)
        return 1
    quoted = sorted({r for e in PUBLISHED.values() for r in e.get("quoted_in", [])})
    print("\nOK — the fits reproduce, and each document that prints them was read to")
    print(f"     confirm it still does: {', '.join(quoted)}")
    print(f"     The {len(EFFDET_LADDERS)} EfficientDet ladders are not refitted here, but the "
          "same\n     documents were read for the coefficients those files publish.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="assert the values published in docs/PRUNING_HAZARDS.md")
    args = ap.parse_args()

    results = [analyse(load_panel(), "this repo (30 compiled models, 3 backbones)",
                       with_onchip=True)]
    paper = load_paper()
    if paper is None:
        print(f"note: {PAPER.relative_to(REPO)} is absent; skipping the paper-table refit.")
    else:
        # The paper reports on-chip as very nearly constant across its 63 pruned rows, so an on-chip term
        # is collinear with the intercept there and is dropped. Off-chip alone carries the fit.
        results.append(analyse(paper, "DSD 2026 ResNet-50 table (CIFAR-100)",
                               with_onchip=False))

    reid = load_reid()
    if reid is None:
        print(f"note: {REID.relative_to(REPO)} is absent; skipping the cross-family refit.")
    else:
        results.append(analyse(reid, "reid_youtu_lite ladder (6 rungs, Market-1501)",
                               with_onchip=True))

    for res in results:
        render(res)

    for line in READING:
        print(line)

    OUT.write_text(json.dumps({
        "produced_by": "results/latency_law/fit_latency_law.py",
        "what": ("Weight-transfer latency law for the Coral USB Edge TPU, and a test for a "
                 "discontinuity at the on-chip/off-chip boundary. Backs section 2 of "
                 "docs/PRUNING_HAZARDS.md."),
        "sources": {
            "this repo (30 compiled models, 3 backbones)": "results/line_seg/param_ladder_oncar.json",
            "DSD 2026 ResNet-50 table (CIFAR-100)": "results/dsd2026_reference/resnet50_table.csv",
        },
        "fits": results,
        "reading": "\n".join(l for l in READING if l).replace("  ", " "),
    }, indent=1) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")

    return check(results) if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
