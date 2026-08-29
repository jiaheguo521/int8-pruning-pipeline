#!/usr/bin/env python3
"""Generate the README figures from the committed measurements.

Every figure here is drawn from a file under results/ that is already in git, and
the fitted line in figure 1 comes from results/latency_law/fit_latency_law.py --
this script imports that fit rather than restating its coefficients, so a figure
can never drift away from the number the document publishes.

    python3 results/figures/make_figures.py --check   # report input readiness only
    python3 results/figures/make_figures.py           # write every figure
    python3 results/figures/make_figures.py --only latency_law

Unlike the verification scripts under results/, this one needs matplotlib. That
is deliberate: reproducing a *claim* must not require an environment, but
redrawing a picture may.

Figures written (PNG, 150 dpi, into this directory):

  fig_latency_law.png       findings 1 and 2. Left: latency against off-chip
                            MiB with the on-chip term removed, so the fitted law
                            is a single straight line and the reader can see that
                            nothing happens where streaming stops. Right: the
                            same measurements as speed-up, where the reciprocal
                            manufactures a knee.
                            <- results/line_seg/param_ladder_oncar.json

  fig_detection_criteria.png
                            finding 5, one panel per model. Post-recovery mAP
                            against realized full-model parameter reduction, for
                            the one protocol criterion tested (magnitude_l2) and
                            the one outside the protocol that recovers it (lamp).
                            Plotted as a percentage of each model's own unpruned
                            fp32, which is what makes the replication legible:
                            the two baselines differ.
                            <- results/detection/full_mapping_ladder_lite1.json
                               results/detection/full_mapping_ladder_lite2.json

  fig_pruning_return.png    finding 2, across families -- and the page's lead
                            figure, promoted above the fold, so finding 2 itself
                            carries the table and points back here. Left: the
                            speed-up a
                            ~90% parameter cut buys, against how many MiB of
                            weights the model started with, on a log axis with
                            the 8 MiB cache marked. Right: the same six models'
                            full ladders, so the reader can see the two that
                            never stream stay flat the whole way.
                            <- results/line_seg/param_ladder_oncar.json
                               results/reid/coral_latency.json
                               results/detection/full_mapping_ladder_lite1.json
                               results/detection/full_mapping_ladder_lite2.json

  fig_detection_floor.png   findings 3 and 4, on the family that ships. Left: both
                            EfficientDets' 19 rungs against raw MACs, each with
                            its own fit and its own shaded I/O floor -- two
                            thirds of a dense Lite1 inference, and 59% of a
                            Lite2 one, is a cost pruning cannot reach. The fits
                            are never pooled; see _ladder_law. Right: how well
                            latency fits padded MACs as the padding width
                            varies -- both models, both peaking at the
                            accelerator's own 64, which makes the peak a
                            property of the device rather than of one network.
                            <- results/detection/full_mapping_ladder_lite1.json
                               results/detection/full_mapping_ladder_lite2.json
                               results/detection/channel_census_lite1.json
                               results/detection/channel_census_lite2.json
                               results/detection/io_floor_control.json

  fig_detection_macs.png    finding 4. Raw MACs do not fix latency inside one
                            model; MACs padded to the 64-wide tile do. Left: the pooled fit
                            with both criterion arms on it, and each arm's own
                            fit over the top. Right: the same as residuals,
                            where all nine lamp rungs sit above the line and
                            seven of nine magnitude_l2 below.
                            <- results/detection/full_mapping_ladder_lite2.json

  fig_allocation_skew.png   finding 6, the mechanism. Left: the share of each
                            group's channels that a single global threshold can
                            reach, by depth -- the early depthwise groups have a
                            long left tail and the late ones do not. Right: the
                            control, raw magnitude, which runs the other way, so
                            what drives the collapse is within-group skew after
                            normalisation and not cross-layer scale.
                            <- results/detection/allocation_stats.json
"""
import argparse
import json
import sys
from pathlib import Path


def _plain_log_ticks(axis, values, highlight=None):
    """Label a log axis in its own units instead of 10^k.

    Matplotlib's default LogLocator prints 10^0 / 10^1, which is exactly the
    wrong thing when the point of the axis is where a threshold falls: a reader
    cannot see that 8 MiB sits between the two decade ticks. Give it explicit
    ticks in MiB, and bold the one that is the threshold.
    """
    from matplotlib.ticker import NullFormatter
    axis.set_ticks(values)
    axis.set_ticklabels([f"{v:g}" for v in values])
    axis.set_minor_formatter(NullFormatter())
    if highlight is not None:
        for t, v in zip(axis.get_ticklabels(), values):
            if v == highlight:
                t.set_fontweight("bold")
                t.set_color("0.15")

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PANEL = REPO / "results" / "line_seg" / "param_ladder_oncar.json"
LADDER = REPO / "results" / "detection" / "full_mapping_ladder_lite1.json"
LADDER2 = REPO / "results" / "detection" / "full_mapping_ladder_lite2.json"
ALLOC = REPO / "results" / "detection" / "allocation_stats.json"
REID = REPO / "results" / "reid" / "coral_latency.json"
CENSUS = REPO / "results" / "detection" / "channel_census_lite1.json"
CENSUS2 = REPO / "results" / "detection" / "channel_census_lite2.json"
FLOOR = REPO / "results" / "detection" / "io_floor_control.json"

sys.path.insert(0, str(REPO / "results" / "latency_law"))

LADDERS = {"lite1": LADDER, "lite2": LADDER2}
CENSUSES = {"lite1": CENSUS, "lite2": CENSUS2}


def mib(s):
    """Parse the ladders' byte strings -- "7.14MiB", "701.62KiB", "0.00B" -- to MiB.

    Not a one-liner on the MiB suffix, which is what this used to be. Lite1
    streams "0.00B" at every rung, so dropping everything that did not end in
    MiB happened to be right there; Lite2 streams "701.62KiB" at its dense rung
    and "43.31KiB" at lamp -10%, which that reading silently turned into 0 and
    with it the one off-chip crossing this repository has on a detector.
    """
    for suffix, scale in (("MiB", 1.0), ("KiB", 1 / 1024), ("B", 1 / 1024 ** 2)):
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * scale
    raise ValueError(f"unrecognised byte string: {s!r}")


def _ladder_law(tag):
    """The `latency_law` finding a ladder publishes for itself.

    Read rather than restated, for the same reason figure 1 imports its fit: a
    figure that hardcodes a coefficient can drift away from the document while
    both still look right. Keys are `a_ms`, `b_us_per_MMAC`, `r2`, `n`, and the
    line is a + b*(MMACs/1000) with MACs in G. The two ladders are kept apart on
    purpose -- see results/README.md and todo.md, which forbid a pooled fit
    because it would conflate two different I/O floors.
    """
    return json.loads(LADDERS[tag].read_text())["findings"]["latency_law"]


def _law_line(law, xs):
    """(x0, x1), (y0, y1) for a law's line across the span of xs, x in GMACs."""
    lo, hi = min(xs) * 0.97, max(xs) * 1.03
    f = lambda x: law["a_ms"] + law["b_us_per_MMAC"] * x
    return (lo, hi), (f(lo), f(hi))


DPI = 150

# One colour per line_seg backbone and per detection criterion, fixed so the two figures cannot clash.
VARIANT_STYLE = {
    "line_seg_base128":  ("#4C72B0", "o", "base128  0.25 MiB int8 (already on-chip)"),
    "line_seg_w96":      ("#DD8452", "s", "w96  8.30 MiB int8 (just over the 8 MiB cap)"),
    "line_seg_link_r34": ("#C44E52", "^", "link_r34  27.50 MiB int8 (3.4x the cap)"),
}
CRIT_STYLE = {
    "magnitude_l2": ("#C44E52", "o", "in the protocol's criterion set"),
    "lamp":         ("#4C72B0", "s", "not in that set"),
}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


# Figure 1 -- the latency law, and the reciprocal artefact
def fig_latency_law(plt):
    from fit_latency_law import analyse, load_panel

    rows = load_panel()
    fit = analyse(rows, "panel", with_onchip=True)
    b0 = fit["law"]["intercept"]
    b_off = fit["law"]["offchip"]
    b_on = fit["law"]["onchip"]
    step = fit["steps"]["I[offchip>0]"]

    raw = json.loads(PANEL.read_text())["rows"]
    by_variant = {}
    for r in raw:
        by_variant.setdefault(r["variant"], []).append(r)
    for v in by_variant:
        by_variant[v].sort(key=lambda r: r["rung"])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))

    # -- left: partial-residual plot ---------------------------------------
    # y is latency with the on-chip contribution subtracted out, so the fitted surface collapses to
    # the single line b_off * x + b0 and all 30 models can be shown against it at once.
    for name, (colour, marker, label) in VARIANT_STYLE.items():
        rs = by_variant.get(name, [])
        ax.scatter([r["offchip_mib"] for r in rs],
                   [r["tpu_ms_mean"] - b_on * r["onchip_mib"] for r in rs],
                   c=colour, marker=marker, s=34, zorder=3, label=label,
                   edgecolors="white", linewidths=0.5)

    xmax = max(r["offchip_mib"] for r in raw) * 1.06
    ax.plot([0, xmax], [b0, b0 + b_off * xmax], color="0.35", lw=1.3, zorder=2,
            label=f"fit: {b_off:.3f}·offchip + {b0:.3f}   R² = {fit['r2']:.4f}")

    n_onchip = sum(1 for r in raw if r["offchip_mib"] == 0)
    ax.axvspan(-1.0, 0.45, color="0.90", zorder=0)
    ax.text(0.03, 0.97,
            f"shaded: the {n_onchip} of {len(raw)} models that stream nothing.\n"
            f"They sit on the same line as the rest — a step\n"
            f"indicator buys {step['coef_ms']:+.3f} ms, F(1,{step['df'][1]}) = {step['F']:.2f}, p = {step['p']:.2f}.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color="0.25")

    ax.set_xlabel("off-chip weights (MiB)")
    ax.set_ylabel(f"latency − {b_on:.3f}·onchip_MiB  (ms)")
    ax.set_title("Latency is linear in bytes streamed — with no step where streaming stops")
    ax.set_xlim(-1.0, xmax)
    ax.legend(loc="lower right", fontsize=7.5)

    # -- right: the same numbers as speed-up -------------------------------
    for name, (colour, marker, label) in VARIANT_STYLE.items():
        rs = by_variant.get(name, [])
        if not rs:
            continue
        base = rs[0]["tpu_ms_mean"]
        bx.plot([r["rung"] for r in rs], [base / r["tpu_ms_mean"] for r in rs],
                color=colour, marker=marker, ms=4.5, lw=1.4,
                label=f"{label.split()[0]}  →  {base / rs[-1]['tpu_ms_mean']:.2f}×")
        # where this backbone stops streaming
        crossed = [r for r in rs if r["offchip_mib"] == 0]
        if crossed and crossed[0]["rung"] > 0:
            bx.axvline(crossed[0]["rung"], color=colour, lw=0.8, ls="--", alpha=0.45)

    bx.set_xlabel("parameter reduction (%)")
    bx.set_ylabel("speed-up  (baseline latency / latency)")
    bx.set_title("The same measurements, plotted as speed-up")
    bx.set_xticks([r["rung"] for r in by_variant["line_seg_base128"]])
    bx.legend(loc="upper left", title="dashed line = off-chip reaches 0",
              title_fontsize=7.5, fontsize=8)
    bx.text(0.04, 0.46,
            "speed-up = 1/latency. A straight line through a\n"
            "reciprocal comes out as a knee; the latency it is\n"
            "computed from is the smooth line on the left.",
            transform=bx.transAxes, ha="left", va="bottom", fontsize=8, color="0.25")

    fig.text(0.5, -0.045,
             "30 compiled models, three backbones, 0.08–27 MiB of weights; 100 invokes each on a Coral USB Edge TPU.  "
             "Data: results/line_seg/param_ladder_oncar.json  ·  fit: results/latency_law/fit_latency_law.py",
             ha="center", fontsize=7.5, color="0.4")
    return fig, "fig_latency_law.png"


# Figure 2 -- criterion transfer on COCO detection
def _ladder_series(tag="lite1"):
    d = json.loads(LADDERS[tag].read_text())
    dense = next(r for r in d["rows"] if r["importance"] == "dense")
    out = {}
    for r in d["rows"]:
        if r["importance"] == "dense":
            continue
        out.setdefault(r["importance"], []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r["target_pct"])
    return d, dense, out


def fig_detection_criteria(plt):
    """Post-recovery mAP against realized reduction, fp32 and int8, per criterion.

    Both columns come off the same build, so the int8 series is the honest one to
    read for deployment and the fp32 one shows what pruning alone cost.

    One panel per model, because the claim the section makes is that the split
    between the two criteria *replicates* -- and a replication is only visible
    if both models are drawn. The y axis is each model's percentage of its own
    unpruned fp32 mAP: the two baselines differ (0.3219 against 0.3588), so
    absolute mAP would put the two ladders at different heights and hide the
    thing they have in common.
    """
    panels = [("lite1", "EfficientDet-Lite1 @384"),
              ("lite2", "EfficientDet-Lite2 @448")]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)

    for (tag, label), ax in zip(panels, axes):
        d, dense, series = _ladder_series(tag)
        base = dense["fp32_mAP"]
        pct = lambda v: 100 * v / base

        for crit, rs in sorted(series.items()):
            colour, marker, note = CRIT_STYLE[crit]
            ax.plot([r["realized_full_pct"] for r in rs],
                    [pct(r["fp32_mAP"]) for r in rs],
                    color=colour, marker=marker, ms=5, lw=1.8,
                    label=f"{crit} — fp32  ({note})")
            ax.plot([r["realized_full_pct"] for r in rs],
                    [pct(r["int8_mAP"]) for r in rs],
                    color=colour, marker=marker, ms=3.5, lw=1.2, ls="--",
                    alpha=0.75, label=f"{crit} — int8 on the Edge TPU build")

        ax.axhline(100, color="0.72", lw=0.8, ls="--", zorder=0)
        ax.text(99, 100, f"unpruned fp32 {base:.4f}  ", fontsize=7.5,
                color="0.45", va="bottom", ha="right")
        ax.axhline(pct(dense["int8_mAP"]), color="0.85", lw=0.8, ls=":", zorder=0)

        # The 60% rung is where the protocol states its claim.
        at60 = {c: next(r for r in rs if r["target_pct"] == 60)
                for c, rs in series.items()}
        ax.axvline(60, color="0.6", lw=0.8, ls="-.", zorder=0)
        ax.text(60, 113, "~60% reduction  ", fontsize=7.5, color="0.45",
                va="top", ha="right")
        for crit, r in at60.items():
            lamp = crit == "lamp"
            ax.annotate(f"{pct(r['fp32_mAP']):.1f}%",
                        xy=(r["realized_full_pct"], pct(r["fp32_mAP"])),
                        xytext=(7, 9) if lamp else (-8, -17),
                        textcoords="offset points",
                        ha="left" if lamp else "right",
                        fontsize=8.5, color=CRIT_STYLE[crit][0],
                        fontweight="bold")

        ax.set_xlabel("realized full-model parameter reduction (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 118)
        ax.set_title(f"{label}  —  {d['rows'][0]['tpu_ops']}/"
                     f"{d['rows'][0]['tpu_ops']} operators on the Edge TPU",
                     fontsize=10)
    axes[0].set_ylabel("COCO val2017 mAP, % of that model's unpruned fp32")
    axes[0].legend(loc="lower left", fontsize=7.5)

    fig.suptitle("The criterion decides whether accuracy survives — on both models",
                 fontsize=12)
    fig.tight_layout()
    fig.text(0.5, -0.04,
             "DET_SCOPE=full, 40 epochs on COCO train2017, evaluated on val2017 "
             "(5000 images), 19 rungs each.  "
             "Data: results/detection/full_mapping_ladder_lite1.json, "
             "full_mapping_ladder_lite2.json",
             ha="center", fontsize=7.5, color="0.4")
    return fig, "fig_detection_criteria.png"


# Figure 3 -- why global ranking guts the early layers.
# The three group kinds are drawn apart because the depthwise groups carry the skew and
# the pointwise-linear projections have no left tail. Merging them hides the contrast.
KIND_STYLE = {
    "conv_dw":  ("#C44E52", "o", "depthwise"),
    "conv_pwl": ("#4C72B0", "s", "pointwise-linear (projection)"),
    "conv_pw":  ("#55A868", "^", "pointwise projection (block 0)"),
}


def _alloc_groups():
    d = json.loads(ALLOC.read_text())
    out = []
    for i, g in enumerate(d["groups"]):
        name = g["group"].split(" (")[0].replace("backbone.", "")
        out.append({"i": i, "name": name, "kind": name.rsplit(".", 1)[1],
                    "block": name.split(".")[1], **g})
    return d, out


def _block_ticks(ax, groups):
    """Tick once per block, at the first group that belongs to it."""
    seen, ticks, labels = set(), [], []
    for g in groups:
        if g["block"] not in seen:
            seen.add(g["block"])
            ticks.append(g["i"])
            labels.append(f"block {g['block']}")
    ax.set_xticks(ticks)
    # Block 0 holds one group, so "block 0" and "block 1" land one index apart and collide at any rotation.
    ax.set_xticklabels([l.replace("block ", "") for l in labels], fontsize=8)
    ax.set_xlabel("backbone block")


def fig_allocation_skew(plt):
    d, groups = _alloc_groups()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))

    for kind, (colour, marker, label) in KIND_STYLE.items():
        gs = [g for g in groups if g["kind"] == kind]
        if not gs:
            continue
        ax.scatter([g["i"] for g in gs], [100 * g["frac_below_half"] for g in gs],
                   c=colour, marker=marker, s=38, label=label, zorder=3,
                   edgecolors="white", linewidths=0.5)
        bx.scatter([g["i"] for g in gs], [g["raw_l2_mean"] for g in gs],
                   c=colour, marker=marker, s=38, label=label, zorder=3,
                   edgecolors="white", linewidths=0.5)

    named = {g["name"]: g for g in groups}
    # Both labels point left: the right edge belongs to the legend, and blocks.5.4 sits close enough
    # to be clipped otherwise. It also sits at 2.7% inside a dense cluster, so a short offset would
    # land the text on its neighbours; reach up into the empty band instead.
    for who, off in (("blocks.1.0.conv_dw", (12, 4)), ("blocks.5.4.conv_dw", (-14, 78))):
        g = named[who]
        ax.annotate(f"{who}\n{100 * g['frac_below_half']:.1f}% of channels below 0.5",
                    xy=(g["i"], 100 * g["frac_below_half"]),
                    xytext=off, textcoords="offset points", fontsize=8, color="0.25",
                    ha="left" if off[0] > 0 else "right",
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7))

    ax.set_ylabel("channels below 0.5 relative importance  (%)")
    ax.set_title("A single global threshold reaches the early layers first")
    ax.set_ylim(-3, 68)
    _block_ticks(ax, groups)

    ax.text(0.03, 0.72,
            f"All {d['n_groups']} groups measure a normalised mean of exactly\n"
            f"{d['normalised_means_observed'][0]:.4f}, so cross-layer scale is already\n"
            f"cancelled and cannot be what drives the cut.",
            transform=ax.transAxes, fontsize=8, color="0.25", va="top")

    bx.set_yscale("log")
    _plain_log_ticks(bx.yaxis, [0.3, 0.5, 1, 2, 5, 10])
    bx.set_ylabel("raw L2 magnitude, group mean  (log)")
    bx.set_title("The control: raw magnitude runs the other way")
    _block_ticks(bx, groups)
    # One legend above both panels: the marker scheme is shared, and the corners are needed for callouts.
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.07), fontsize=8, title="group kind",
               title_fontsize=8, columnspacing=1.6)
    fig.text(0.5, -0.055,
             "Right: early groups are LARGER in raw magnitude than late ones, yet they are the ones emptied — "
             "the driver is within-group skew after normalisation, not scale across layers.",
             ha="center", fontsize=8, color="0.3")
    fig.text(0.5, -0.115,
             f"{d['model']}, {d['protocol']['importance']}, normalizer='{d['protocol']['normalizer']}', "
             f"global_pruning={d['protocol']['global_pruning']}, scope={d['protocol']['scope']}.  "
             "Data: results/detection/allocation_stats.json  ·  "
             "regenerate: results/protocol_audit/allocation_stats.py",
             ha="center", fontsize=7.5, color="0.4")
    return fig, "fig_allocation_skew.png"


# Figure 4 -- what the return on pruning is actually set by.
# Colour per MODEL, marker per family: colour has to separate five curves in the right
# panel, so it cannot also carry the family. The three line_seg colours are figure 1's.
RETURN_STYLE = {
    "base128":           ("#4C72B0", "o"),
    "efficientdet_lite1": ("#55A868", "D"),
    # Same family as lite1, so a darker shade of the green rather than a sixth hue; the marker separates them.
    "efficientdet_lite2": ("#2E7D5B", "v"),
    "w96":               ("#DD8452", "o"),
    "reid_youtu_lite":   ("#8172B3", "^"),
    "link_r34":          ("#C44E52", "o"),
}
CAP_MIB = 8.0          # the Coral USB's on-chip parameter cache


def _return_ladders():
    """Five models, three families, each as
    (label, family, [(reduction%, ms, total_MiB, offchip_MiB)]).

    Every series is read from a committed measurement file and starts at its own
    dense rung, so a speed-up here is always a model against itself. off-chip is
    carried per rung, not just for the dense one, because the rung where it
    reaches 0 is the rung where a model becomes fully resident -- which is the
    thing a reader wants to locate and cannot infer from a ratio.
    """
    out = []
    panel = {}
    for r in json.loads(PANEL.read_text())["rows"]:
        panel.setdefault(r["variant"], []).append((r["realized_param_pct"],
                                                   r["tpu_ms_mean"],
                                                   r["onchip_mib"] + r["offchip_mib"],
                                                   r["offchip_mib"]))
    for v in ("line_seg_base128", "line_seg_w96", "line_seg_link_r34"):
        rs = sorted(panel[v])
        out.append((v.replace("line_seg_", ""), "line_seg", rs))

    # The lamp arm, the one that survives to the top rung. Both models, because they start
    # on opposite sides of the interesting line: every Lite1 rung is already resident, while
    # Lite2's dense rung sits at 7.14 + 0.69 MiB and still spills.
    for tag, label in (("lite1", "efficientdet_lite1"),
                       ("lite2", "efficientdet_lite2")):
        lad = [r for r in json.loads(LADDERS[tag].read_text())["rows"]
               if r["importance"] in ("dense", "lamp")]
        out.append((label, "effdet",
                    sorted((r["realized_full_pct"], r["tpu_ms_mean"],
                            mib(r["onchip"]) + mib(r["offchip"]),
                            mib(r["offchip"]))
                           for r in lad)))

    reid = json.loads(REID.read_text())["pruning_ladder"]
    out.append(("reid_youtu_lite", "reid",
                sorted((r["realized_full_pct"], r["tpu_ms_mean"],
                        r["onchip_mib"] + r["offchip_mib"], r["offchip_mib"])
                       for r in reid)))
    return out


def _fully_resident_rung(rs):
    """The first rung at which nothing streams, or None if it never streams or
    never stops. rs is the 4-tuple series from _return_ladders()."""
    if rs[0][3] == 0:
        return None                      # on-chip from the start; nothing to cross
    for pct, ms, _, off in rs:
        if off == 0:
            return pct, ms
    return None


def fig_pruning_return(plt):
    series = sorted(_return_ladders(), key=lambda t: t[2][0][2])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4))

    # -- left: the deepest rung of each model against where it started ------
    # Label offsets are per-model: reid and link_r34 start 1.6 MiB apart and end within 0.4x, so they collide.
    NUDGE = {"reid_youtu_lite": (0, 19, "center"), "link_r34": (11, -9, "left"),
             # The two effdets start 1.5 MiB and 0.06x apart, so they need opposite sides of their markers.
             "efficientdet_lite1": (-10, 10, "right"),
             "efficientdet_lite2": (10, -12, "left"),
             # base128 sits at 1.26x, where the default downward label runs off the axes. Put it above the marker.
             "base128": (10, 13, "left")}
    for label, fam, rs in series:
        base_pct, base_ms, base_mib = rs[0][:3]
        top_pct, top_ms = rs[-1][0], rs[-1][1]
        colour, marker = RETURN_STYLE[label]
        gain = base_ms / top_ms
        ax.scatter(base_mib, gain, c=colour, marker=marker, s=80, zorder=3,
                   edgecolors="white", linewidths=0.6)
        dx, dy, ha = NUDGE.get(label, (9, -11, "left"))
        ax.annotate(f"{label}\n\N{MINUS SIGN}{top_pct:.1f}%  \u2192  {gain:.2f}\u00d7",
                    (base_mib, gain), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, ha=ha, va="center", color=colour)
    # The cache size as a reference mark, not a predicate for residency: Lite2 starts 0.17 MiB inside
    # this band and still spills. The band says how big a model is; the rings on the right say what it does.
    ax.axvspan(0.01, CAP_MIB, color="#eef2f6", zorder=0)
    ax.axvline(CAP_MIB, color="0.35", ls="--", lw=1.2, zorder=1)
    # Three labels on one line collide, and the bold 8 on the axis already names the threshold.
    ax.annotate("8 MiB cache", (CAP_MIB, 9.2), rotation=90, fontsize=8,
                color="0.3", va="center", ha="right",
                xytext=(-3, 0), textcoords="offset points")
    ax.annotate("smaller than the cache", (0.42, 17.3), fontsize=8.5, color="0.45",
                va="top", ha="left")
    ax.annotate("larger than the cache", (30, 17.3), fontsize=8.5, color="0.45",
                va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(0.2, 45)
    ax.set_ylim(0, 18)
    # Decade ticks would put 8 between 10^0 and 10^1 with nothing to read it against.
    _plain_log_ticks(ax.xaxis, [0.25, 0.5, 1, 2, 4, 8, 16, 32], highlight=8)
    ax.set_xlabel("weights the dense model starts with (MiB, compiler's on-chip + off-chip)")
    ax.set_ylabel("speed-up at the deepest rung")
    ax.set_title("The return is set by how far above the cache you start")

    # -- right: the whole ladder, so "flat" is visible rather than asserted -
    for label, fam, rs in series:
        base_ms = rs[0][1]
        colour, marker = RETURN_STYLE[label]
        # Residency is measured, not read off which side of the cap a rung sits on: Lite2 starts
        # at 7.83 MiB, under the 8 MiB cache, and still spills 0.69 MiB.
        under = rs[0][3] == 0
        bx.plot([r[0] for r in rs], [base_ms / r[1] for r in rs],
                color=colour, marker=marker, ms=4.5, lw=1.7,
                ls="--" if under else "-", zorder=3, markeredgecolor="white",
                markeredgewidth=0.4,
                label=f"{label}  \u2014  {rs[0][2]:.1f} MiB, "
                      f"{'already on-chip' if under else 'streams at first'}")
        # Ring the rung where the model stops streaming: a ratio does not tell a reader whether the pruned
        # model still spills off-chip, and that is what the whole panel turns on.
        hit = _fully_resident_rung(rs)
        if hit:
            hp, hms = hit
            bx.scatter([hp], [base_ms / hms], s=190, facecolors="none",
                       edgecolors=colour, lw=1.8, zorder=5)
            bx.annotate(f"{hp:.0f}%", (hp, base_ms / hms), textcoords="offset points",
                        xytext=(0, 15), fontsize=8, color=colour, ha="center")
    bx.axhline(1.0, color="0.75", lw=1, zorder=1)
    bx.scatter([], [], s=150, facecolors="none", edgecolors="0.4", lw=1.6,
               label="the rung where off-chip reaches 0")
    bx.set_xlabel("realized full-model parameter reduction (%)")
    bx.set_ylabel("speed-up against the model's own dense rung")
    bx.set_title("The jump is where the model stops streaming, not where the ratio is")
    bx.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig, "fig_pruning_return.png"


# Figure 5 -- what actually binds on detection, once nothing streams
def fig_detection_floor(plt):
    """The finding the published family rests on, and the one with no picture.

    EfficientDet-Lite1 streams zero bytes at all 19 rungs, so figure 1's
    mechanism has nothing to act on. What is left is a large fixed I/O cost plus
    a term in MACs -- and the MAC term only fits well once channel counts are
    padded to the accelerator's array width. Left panel shows how little of the
    dense latency pruning can reach; right shows where the array width is.
    """
    d = json.loads(CENSUS.read_text())
    fl = json.loads(FLOOR.read_text())
    rows = sorted(d["rows"], key=lambda r: r["padded_mmacs"]["64"])
    rate = fl["derivation"]["return_bandwidth_mib_per_s"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    # -- left: two models, two floors, neither of them reachable ------------
    # Raw MACs, not padded-64: only Lite1 has a channel census to pad with, which is also
    # why the padding argument stays Lite1-only in the right panel. The two fits are drawn
    # separately and never pooled -- results/README.md and todo.md both forbid it, because
    # one line through both averages two I/O floors into a number describing neither.
    TRAFFIC = {"lite1": "0.42 MiB in + 2.48 out",
               "lite2": "0.57 MiB in + 3.37 out"}
    MODELS = (("lite2", "EfficientDet-Lite2 @448", "#2E7D5B"),
              ("lite1", "EfficientDet-Lite1 @384", "#55A868"))
    floors = {}
    for tag, label, ink in MODELS:
        law = _ladder_law(tag)
        rs = json.loads(LADDERS[tag].read_text())["rows"]
        floors[tag] = law["a_ms"]
        xs = [r["mmacs"] / 1000 for r in rs]
        ax.scatter(xs, [r["tpu_ms_mean"] for r in rs], c=ink, s=34, zorder=4,
                   edgecolors="white", linewidths=0.5,
                   label=f'{label} — {law["n"]} rungs')
        (flo, fhi), (ylo, yhi) = _law_line(law, xs)
        ax.plot([flo, fhi], [ylo, yhi], color=ink, lw=1.3, zorder=3)
        ax.axhline(law["a_ms"], color=ink, lw=1.1, ls="--", zorder=2)
        dense_ms = next(r["tpu_ms_mean"] for r in rs if r["importance"] == "dense")
        # Short and right-aligned so it clears the Lite1 cloud, which ends at 1.95 G astride Lite2's floor line.
        ax.annotate(f'floor {law["a_ms"]:.2f} ms — '
                    f'{100 * law["a_ms"] / dense_ms:.0f}% of {dense_ms:.1f}\n'
                    f'{TRAFFIC[tag]}',
                    (3.54, law["a_ms"] + 0.8), fontsize=7.8, color=ink,
                    ha="right", va="bottom")
    # Both bands shaded from zero, taller first, so the strip between the floors reads as Lite2's extra.
    ax.axhspan(0, floors["lite2"], color="#eef4f0", zorder=0)
    ax.axhspan(0, floors["lite1"], color="#dfeae3", zorder=0)

    ax.annotate(f"Lite2's floor is {floors['lite2'] / floors['lite1']:.2f}\u00d7 "
                "Lite1's.\nBoth are the same return path at "
                f"{rate} MiB/s,\ncharged on a bigger tensor \u2014 not on a\n"
                "bigger model, and not on its weights.\nPruning reaches none of either floor.",
                (0.60, 19.0), fontsize=8.2, color="0.3", ha="left", va="center")
    ax.set_ylim(0, 92)
    ax.set_xlim(0.45, 3.58)
    ax.set_xlabel("MACs (G)")
    ax.set_ylabel("Edge TPU latency (ms)")
    ax.set_title("Two models, two I/O floors, neither one reachable")
    ax.legend(loc="upper left", fontsize=8)

    # -- right: the fit peaks at the systolic array width, on both models ---
    # Two censuses now, so the peak is a claim about the DEVICE rather than about one network.
    widths = sorted(int(k) for k in d["r2_by_width"])
    peak = d["peak_width"]
    for tag, label, ink in (("lite2", "Lite2 @448", "#2E7D5B"),
                            ("lite1", "Lite1 @384", "#55A868")):
        c = json.loads(CENSUSES[tag].read_text())
        ws = sorted(int(k) for k in c["r2_by_width"])
        bx.plot(ws, [c["r2_by_width"][str(w)] for w in ws], color=ink, lw=1.6,
                marker="o", ms=5, markeredgecolor="white", markeredgewidth=0.5,
                zorder=3, label=f'{label} — peak W={c["peak_width"]}')
        bx.scatter([c["peak_width"]], [c["r2_by_width"][str(c["peak_width"])]],
                   s=150, facecolors="none", edgecolors=ink, lw=1.6, zorder=4)
    bx.annotate(f"W = {peak} — the Edge TPU's systolic array width,\n"
                "and the peak on both models.\nCutting a layer from 88 "
                "channels to 40 costs\nthe same as cutting it to 64.",
                (peak, d["r2_by_width"][str(peak)]), textcoords="offset points",
                xytext=(-14, -74), fontsize=8, color="0.3", ha="right",
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7,
                                shrinkA=2, shrinkB=10))
    bx.set_xscale("log", base=2)
    _plain_log_ticks(bx.xaxis, widths, highlight=peak)
    bx.set_ylim(0.55, 1.02)
    bx.set_xlabel("channel counts rounded up to a tile of width W")
    bx.set_ylabel("R\u00b2 of latency against padded MACs")
    bx.set_title("Below the array width, pruning buys accuracy loss and no time")
    bx.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.text(0.5, -0.05,
             f"Left: EfficientDet-Lite1 @384 and Lite2 @448, {_ladder_law('lite1')['n']} rungs each, "
             f"fitted separately on raw MACs.  Right: the same ladders against MACs "
             f"padded to a tile of width W.  "
             f"Data: results/detection/full_mapping_ladder_lite{{1,2}}.json, "
             f"channel_census_lite{{1,2}}.json, io_floor_control.json",
             ha="center", fontsize=8, color="0.4")
    return fig, "fig_detection_floor.png"


# Figure 6 -- on Lite2, equal MACs do not mean equal milliseconds
def fig_detection_macs(plt):
    """Why Lite2's two criterion arms take different slopes: the axis was wrong.

    The ladder fits latency against raw MACs and the residuals sort by pruning
    criterion -- recorded as MACs_do_not_fix_latency, which names channel
    alignment as the untested candidate. The channel census tests it. Both
    panels are residuals about a pooled fit, and the ONLY thing that changes
    between them is whether channel counts are padded to the array width first;
    MACs are recounted the same way in both, by tracing the model, so padding is
    isolated as the single variable.

    What the right panel does not do is remove the split entirely, and it is
    drawn so that the surviving offset is visible rather than described.
    """
    c = json.loads(CENSUS2.read_text())
    peak = str(c["peak_width"])
    rows = c["rows"]
    split = c["criterion_split"]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)
    panels = [("1", "against raw MACs", axes[0]),
              (peak, f"against MACs padded to {peak}", axes[1])]

    for w, label, ax in panels:
        f = c["fits"][w]
        a, b = f["intercept_ms"], f["slope_us_per_mmac"] / 1000
        ax.axhline(0, color="0.35", lw=1.3, zorder=2)
        for crit in ("lamp", "magnitude_l2"):
            colour, marker, _ = CRIT_STYLE[crit]
            sub = sorted((r for r in rows if r["importance"] == crit),
                         key=lambda r: r["realized_full_pct"])
            ax.plot([r["realized_full_pct"] for r in sub],
                    [r["tpu_ms_mean"] - (a + b * r["padded_mmacs"][w]) for r in sub],
                    color=colour, marker=marker, ms=5.5, lw=1.6, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.5, label=crit)
        gap = split[w]["slope_gap_us_per_mmac"]
        ax.set_title(f"{label}\npooled R² {f['r2']:.4f}   ·   arms' slope gap "
                     f"{gap:.3f} µs/MMAC", fontsize=10)
        ax.set_xlabel("realized full-model parameter reduction (%)")

    # What the padding buys, and what it does not.
    raw_gap = split["1"]["slope_gap_us_per_mmac"]
    pad_gap = split[peak]["slope_gap_us_per_mmac"]
    sep = lambda b: (b["lamp"]["residual_ms_mean"]
                     - b["magnitude_l2"]["residual_ms_mean"])
    axes[0].annotate("Every lamp rung above the line,\nevery magnitude_l2 rung but two below.",
                     (0.5, 0.04), xycoords="axes fraction", fontsize=8,
                     color="0.3", ha="center", va="bottom")
    axes[1].annotate(f"Padding collapses the slope gap {raw_gap / pad_gap:.0f}×.\n"
                     f"What survives is a level offset: the arms' mean\n"
                     f"residuals still differ by {abs(sep(split[peak])):.2f} ms, "
                     f"down from {abs(sep(split['1'])):.2f}.\nThat remainder is not explained.",
                     (0.5, 0.04), xycoords="axes fraction", fontsize=8,
                     color="0.3", ha="center", va="bottom")
    axes[0].set_ylabel("measured − pooled fit  (ms)")
    axes[0].set_ylim(-4.6, 3.4)
    axes[0].legend(loc="upper left", fontsize=8)

    fig.suptitle("The two arms are not obeying different laws — they leave "
                 "different channel counts", fontsize=12)
    fig.tight_layout()
    fig.text(0.5, -0.04,
             f"EfficientDet-Lite2 @448, {c['n_rungs']} rungs. MACs are recounted by tracing "
             "every Conv2d, so the left panel is not identical to the ladder's own "
             "latency_law; padding is the only difference between the panels.  "
             "Data: results/detection/channel_census_lite2.json",
             ha="center", fontsize=7.8, color="0.4")
    return fig, "fig_detection_macs.png"


FIGURES = {"latency_law": fig_latency_law,
           "pruning_return": fig_pruning_return,
           "detection_criteria": fig_detection_criteria,
           "detection_floor": fig_detection_floor,
           "detection_macs": fig_detection_macs,
           "allocation_skew": fig_allocation_skew}


def check():
    """Report whether each figure's input is present and complete. Writes nothing."""
    ok = True
    print(f"input files")
    for p in (PANEL, LADDER, LADDER2, ALLOC, REID, CENSUS, FLOOR):
        mark = "ok " if p.exists() else "MISSING"
        print(f"  [{mark}] {p.relative_to(REPO)}")
        ok &= p.exists()

    if PANEL.exists():
        rows = json.loads(PANEL.read_text())["rows"]
        print(f"\nlatency_law: {len(rows)} compiled models, "
              f"{sum(1 for r in rows if r['offchip_mib'] > 0)} of them streaming — complete")

    print("\ndetection_criteria: one panel per model, so both must be complete")
    for tag in ("lite1", "lite2"):
        if not LADDERS[tag].exists():
            continue
        d, dense, series = _ladder_series(tag)
        print(f"  {tag}: {len(d['rows'])} rungs "
              f"(1 dense + {sum(len(v) for v in series.values())} pruned)")
        for crit, rs in sorted(series.items()):
            got = [r["target_pct"] for r in rs]
            missing = [p for p in range(10, 100, 10) if p not in got]
            # Every panel needs both mAP columns: a null here draws a gap rather than failing.
            holes = [r["target_pct"] for r in rs
                     if r.get("fp32_mAP") is None or r.get("int8_mAP") is None]
            print(f"    {crit:13} {len(rs)}/9  "
                  f"{'complete' if not missing else f'missing {missing}'}"
                  f"{'' if not holes else f'  mAP nulls at {holes}'}")
        for note in d.get("open", [])[:1]:
            print(f"    open: {note[:88]}...")

    if LADDER2.exists():
        d2 = json.loads(LADDER2.read_text())
        law = d2["findings"]["latency_law"]
        lat = [r["tpu_ms_mean"] for r in d2["rows"]]
        nulls = [r["target_pct"] for r in d2["rows"] if r["tpu_ms_mean"] is None]
        want = {"latency_law", "MACs_do_not_fix_latency",
                "off_chip_streaming_not_visible"}
        gone = sorted(want - set(d2["findings"]))
        print(f"\ndetection_macs: {law['n']} rungs, "
              f"latency {max(lat):.1f} \u2192 {min(lat):.1f} ms, "
              f"fit {law['a_ms']} + {law['b_us_per_MMAC']} (R\u00b2 {law['r2']})")
        print(f"  findings: {'all three present' if not gone else f'MISSING {gone}'}"
              f"  |  latency nulls: {nulls if nulls else 'none'}")
        ok &= not gone and not nulls

    if ALLOC.exists():
        d, groups = _alloc_groups()
        print(f"\nallocation_skew: {len(groups)} backbone groups — complete")

    if CENSUS.exists() and FLOOR.exists():
        d = json.loads(CENSUS.read_text())
        rows = d["rows"]
        streaming = [r for r in rows if r.get("offchip", "0.00B") != "0.00B"]
        lat = [r["tpu_ms_mean"] for r in rows]
        print(f"\ndetection_floor: {len(rows)} rungs, "
              f"latency {max(lat):.1f} \u2192 {min(lat):.1f} ms, "
              f"peak fit at W={d['peak_width']} (R\u00b2 {d['r2_by_width'][str(d['peak_width'])]})")
        missing = [w for w in (1, 8, 16, 32, 64, 128, 256) if str(w) not in d["r2_by_width"]]
        print(f"  padding widths: {'complete' if not missing else f'missing {missing}'}"
              f"  |  rungs that stream: {len(streaming)} (expected 0)")

    if PANEL.exists() and LADDER.exists() and REID.exists():
        series = sorted(_return_ladders(), key=lambda t: t[2][0][2])
        print(f"\npruning_return: {len(series)} models across "
              f"{len({f for _, f, _ in series})} families")
        for label, fam, rs in series:
            base_mib, base_ms, top_ms = rs[0][2], rs[0][1], rs[-1][1]
            # "streams" is the measured off-chip at the dense rung, not which side of the cap it lands on.
            print(f"  {label:20} {len(rs):2} rungs  {base_mib:6.2f} MiB "
                  f"({'under' if base_mib < CAP_MIB else 'over ':5} cap, "
                  f"{'streams' if rs[0][3] else 'resident'})  "
                  f"{base_ms / top_ms:6.2f}x at -{rs[-1][0]:.1f}%")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="report input readiness and exit; write no figures")
    ap.add_argument("--only", choices=sorted(FIGURES), help="build one figure")
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check() else 1)

    plt = _mpl()
    for name in ([args.only] if args.only else sorted(FIGURES)):
        fig, fname = FIGURES[name](plt)
        out = HERE / fname
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out.relative_to(REPO)}  ({out.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
