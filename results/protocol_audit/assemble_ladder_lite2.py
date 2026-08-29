#!/usr/bin/env python3
"""Assemble results/detection/full_mapping_ladder_lite2.json from measured artifacts.

Mirrors the lite1 template field for field. Latency columns are written only when
a bench JSON is present; without a Coral USB they stay null and `open` says so.
"""
import json, re, os, sys, glob

# Run from the repo root.
D = "outputs/edgetpu/single_20260827_154635"
EV = "outputs/effdet_lite2_int8_eval"
BENCH = "outputs/effdet_lite2_latency"          # optional, produced by the docker bench
OUT = "results/detection/full_mapping_ladder_lite2.json"

grid = {(r["importance"], r["target_pct"]): r
        for r in json.load(open("results/detection/full_scope_grid.json"))["rows"]
        if r["model"] == "efficientdet_lite2"}


def struct(stem):
    cl = open(f"{D}/{stem}_compile.log", encoding="utf-8", errors="replace").read()
    el = open(f"{D}/{stem}_int8_edgetpu.log", encoding="utf-8", errors="replace").read()
    g = lambda p: re.search(p, cl).group(1)
    mapped = sum(int(x) for x in re.findall(r"^\S+\s+(\d+)\s+Mapped to Edge TPU", el, re.M))
    total = int(g(r"Total number of operations: (\d+)"))
    return dict(tpu_ops=mapped, cpu_ops=total - mapped,
                tpu_subgraphs=int(g(r"Number of Edge TPU subgraphs: (\d+)")),
                onchip=g(r"On-chip memory used for caching model parameters: (\S+)"),
                offchip=g(r"Off-chip memory used for streaming uncached model parameters: (\S+)"))


def latency(stem):
    p = f"{BENCH}/{stem}.json"
    if not os.path.exists(p):
        return dict(tpu_ms_run1=None, tpu_ms_run2=None, tpu_ms_mean=None)
    b = json.load(open(p))
    r1, r2 = b["run1_ms"], b["run2_ms"]
    return dict(tpu_ms_run1=round(r1, 3), tpu_ms_run2=round(r2, 3),
                tpu_ms_mean=round((r1 + r2) / 2, 3))


def int8_map(stem):
    p = f"{EV}/{stem}.json"
    if not os.path.exists(p):
        sys.exit(f"[ERROR] missing int8 eval: {p}")
    return json.load(open(p))["mAP"]


rows = []
p0 = json.load(open("outputs/pruning_logs/efficientdet_lite2_coco-train2017_lamp_10pct_full.json"))
dense_fp32 = p0["ref_eval"]["mAP"]
specs = [("dense", 0, "efficientdet_lite2_coco-train2017_pruned0pct",
          p0["pre_params_full"], p0["pre_macs"], 0.0, dense_fp32)]
for imp in ("lamp", "magnitude_l2"):
    for pct in range(10, 100, 10):
        j = json.load(open(f"outputs/pruning_logs/efficientdet_lite2_coco-train2017_{imp}_{pct}pct_full.json"))
        specs.append((imp, pct, f"efficientdet_lite2_coco-train2017_pruned{pct}pct_{imp}_full",
                      j["post_params_full"], j["post_macs"],
                      grid[(imp, pct)]["realized_full_pct"], j["final_eval"]["mAP"]))

for imp, pct, stem, params, macs, realized, fp32 in specs:
    i8 = int8_map(stem)
    row = dict(importance=imp, target_pct=pct, realized_full_pct=realized,
               params=params, mmacs=round(macs / 1e6, 1))
    row.update(struct(stem))
    row.update(fp32_mAP=fp32, int8_mAP=i8,
               int8_loss_pct=round((i8 - fp32) / fp32 * 100, 2))
    row.update(latency(stem))
    row["abs_drop_mAP"] = round(fp32 - i8, 6)
    rows.append(row)

have_lat = any(r["tpu_ms_mean"] is not None for r in rows)


def _open_channel_alignment():
    """The channel-alignment entry: a missing fit before the census, a narrower
    question after it."""
    if CENSUS is None:
        return ("No channel-alignment fit for Lite2, and it is the file's largest open "
                "question rather than a missing measurement: MACs_do_not_fix_latency shows "
                "the two criterion arms taking different slopes with weight traffic ruled "
                "out. What is missing is a padded-MAC or per-layer channel census, not a "
                "Coral.")
    peak = str(CENSUS["peak_width"])
    raw, pad = CENSUS["criterion_split"]["1"], CENSUS["criterion_split"][peak]
    sep = lambda b: abs(b["lamp"]["residual_ms_mean"]
                        - b["magnitude_l2"]["residual_ms_mean"])
    return (
        "The channel-alignment fit now exists (results/detection/channel_census_lite2.json, "
        f"{CENSUS['n_rungs']} rungs traced at {CENSUS['input_size']} px) and it answers most "
        "of what MACs_do_not_fix_latency left open: padding to W="
        f"{peak} takes the arms' slope gap from {raw['slope_gap_us_per_mmac']} to "
        f"{pad['slope_gap_us_per_mmac']} us/MMAC. Two things stay open. First, a level "
        f"offset of {sep(pad):.2f} ms between the arms survives the padding and is not "
        "explained. Second, this is a whole-model fit like Lite1's; neither model has been "
        "tested at layer granularity, so 'channels below the tile width buy no time' remains "
        "an inference about layers drawn from a fit over networks.")


def _arms_explained():
    """What the channel census says about why the two arms differ.

    Before the census this was an untested candidate. It is now measured, and it
    is measured to be most but not all of the effect, which is what this says.
    """
    if CENSUS is None:
        return ("Why the two arms differ is not measured here. The obvious candidate is "
                "the shape of what survives -- the Edge TPU pads channel counts, so two "
                "networks of equal MAC count need not cost equally -- but that is the "
                "channel-alignment fit this file does not have.")
    peak = str(CENSUS["peak_width"])
    raw, pad = CENSUS["criterion_split"]["1"], CENSUS["criterion_split"][peak]
    sep = lambda b: b["lamp"]["residual_ms_mean"] - b["magnitude_l2"]["residual_ms_mean"]
    return (
        "MOSTLY EXPLAINED, by channel alignment, since "
        "results/detection/channel_census_lite2.json was built. Recounting MACs with "
        f"channel counts rounded up to the accelerator's tile width (W={peak}, which is "
        "also where this ladder's fit peaks) collapses the slope difference between the "
        f"arms from {raw['slope_gap_us_per_mmac']} to {pad['slope_gap_us_per_mmac']} "
        f"us/MMAC, a factor of "
        f"{round(raw['slope_gap_us_per_mmac'] / pad['slope_gap_us_per_mmac'])}. So the two "
        "arms are not obeying different laws; they are leaving different channel counts, "
        "and the raw-MAC axis cannot see the padding. What survives is a level offset: the "
        f"arms' mean residuals about the pooled line still differ by {abs(sep(pad)):.2f} ms "
        f"(down from {abs(sep(raw)):.2f}), and lamp still sits above the line at "
        f"{pad['lamp']['n_above_pooled_line']} of {pad['lamp']['n']} rungs. That remainder "
        "is not explained here. NOTE the padded fit's intercept is not the I/O floor and "
        "does not claim to be -- see latency_law.io_floor, which is fitted on raw MACs.")


def ols(pts):
    """Least squares of tpu_ms on MMACs. Stdlib only, like fit_latency_law.py."""
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    a = my - b * mx
    ss = sum((y - my) ** 2 for _, y in pts)
    return dict(a_ms=round(a, 3), b_us_per_MMAC=round(b * 1000, 3),
                r2=round(1 - sum((y - a - b * x) ** 2 for x, y in pts) / ss, 4), n=n)


# Bytes crossing USB per inference, identical at every rung: one 1x448x448x3 int8 input,
# ten int8 head outputs (810 and 36 channels at 56/28/14/7/4). Read off the int8 tflites
# with ai_edge_litert; the compiled artifacts cannot be allocated without the delegate.
IN_MIB, OUT_MIB = 0.5742, 3.3733
IO_MIB = IN_MIB + OUT_MIB
RATE_MIB_S = 86.8   # io_floor_control.json's measured return bandwidth, NOT refitted here:
                    # applying it unchanged is what makes Lite2 an out-of-sample check
L1 = json.load(open("results/detection/full_mapping_ladder_lite1.json"))

# The channel census, if it has been run. Its criterion_split tests what MACs_do_not_fix_latency
# names but could not run: whether padding channel counts to the array width accounts for the split.
try:
    CENSUS = json.load(open("results/detection/channel_census_lite2.json"))
except FileNotFoundError:
    CENSUS = None
L1_LAW = L1["findings"]["latency_law"]
L1_DENSE = [r for r in L1["rows"] if r["importance"] == "dense"][0]


def latency_findings():
    pts = [(r["mmacs"], r["tpu_ms_mean"]) for r in rows]
    fit = ols(pts)
    io_ms = round(IO_MIB / RATE_MIB_S * 1000, 2)
    per = {c: ols([(r["mmacs"], r["tpu_ms_mean"]) for r in rows if r["importance"] == c])
           for c in ("lamp", "magnitude_l2")}
    # Same MACs, different criterion: does the MAC count fix the latency?
    pairs = []
    for L in (r for r in rows if r["importance"] == "lamp"):
        M = min((r for r in rows if r["importance"] == "magnitude_l2"),
                key=lambda r: abs(r["mmacs"] - L["mmacs"]))
        if abs(M["mmacs"] - L["mmacs"]) / L["mmacs"] < 0.05:
            pairs.append(dict(lamp_pct=L["target_pct"], magnitude_l2_pct=M["target_pct"],
                              lamp_mmacs=L["mmacs"], magnitude_l2_mmacs=M["mmacs"],
                              lamp_ms=L["tpu_ms_mean"], magnitude_l2_ms=M["tpu_ms_mean"],
                              lamp_onchip=L["onchip"], magnitude_l2_onchip=M["onchip"],
                              lamp_minus_magnitude_l2_ms=round(L["tpu_ms_mean"] - M["tpu_ms_mean"], 3)))
    max_gap = round(max(p["lamp_minus_magnitude_l2_ms"] for p in pairs), 2)
    res = {c: [round(r["tpu_ms_mean"] - fit["a_ms"] - fit["b_us_per_MMAC"] / 1000 * r["mmacs"], 3)
               for r in rows if r["importance"] == c] for c in ("lamp", "magnitude_l2")}
    mib = lambda t: float(t.replace("MiB", ""))
    heavier_faster = sum(mib(q["magnitude_l2_onchip"]) > mib(q["lamp_onchip"])
                         and q["lamp_minus_magnitude_l2_ms"] > 0 for q in pairs)
    # The lamp arm's own off-chip=0 rungs, extrapolated onto the two that stream.
    on = ols([(r["mmacs"], r["tpu_ms_mean"]) for r in rows
              if r["importance"] == "lamp" and r["offchip"] == "0.00B"])
    stream = [dict(rung=f"{r['importance']} -{r['target_pct']}%", offchip=r["offchip"],
                   predicted_ms=round(on["a_ms"] + on["b_us_per_MMAC"] / 1000 * r["mmacs"], 3),
                   measured_ms=r["tpu_ms_mean"],
                   residual_ms=round(r["tpu_ms_mean"] - on["a_ms"]
                                     - on["b_us_per_MMAC"] / 1000 * r["mmacs"], 3))
              for r in rows if r["offchip"] != "0.00B"]
    return {
      "latency_law": {
        "form": "latency_ms = a + b * MMACs",
        **fit,
        "per_criterion": per,
        "io_floor": f"Every rung accepts {IN_MIB} MiB and returns {OUT_MIB} MiB, "
            f"{round(IO_MIB, 4)} MiB in "
            "total, and the shapes do not move along the ladder (the heads emit 810 and 36 "
            "channels at five levels at every rung), so this ladder cannot fit the return "
            f"rate any more than lite1's could. Charged at the {RATE_MIB_S} MiB/s measured in "
            f"io_floor_control.json, that traffic is {io_ms} ms against a fitted floor of "
            f"{fit['a_ms']} ms, {round(io_ms / fit['a_ms'] * 100)}% of it. That rate was "
            "measured on lite1-derived control "
            "models and is applied here without refitting, so Lite2 is an out-of-sample "
            f"check on it: the floor rises from {L1_LAW['a_ms']} ms to "
            f"{fit['a_ms']} ms, {round(fit['a_ms'] / L1_LAW["a_ms"], 2)}x, where the I/O rises "
            f"2.90 -> {round(IO_MIB, 2)} MiB, {round(IO_MIB / 2.90, 2)}x.",
        "evidence": "results/detection/io_floor_control.json",
      },
      "MACs_do_not_fix_latency": {
        "what": "The pooled fit's residuals are not noise, they are the criterion. All "
            f"{len(res['lamp'])} lamp rungs sit above the line; "
            f"{sum(v < 0 for v in res['magnitude_l2'])} of the {len(res['magnitude_l2'])} "
            "magnitude_l2 rungs sit below it, the exceptions being the deepest two, where the "
            "arms have converged on nearly the same network. Fitted "
            f"separately the two arms reach r2 {per['lamp']['r2']} and "
            f"{per['magnitude_l2']['r2']} on slopes that differ by "
            f"{round(per['lamp']['b_us_per_MMAC'] - per['magnitude_l2']['b_us_per_MMAC'], 3)} "
            f"us/MMAC. Paired at matched MAC counts the gap reaches {max_gap} ms.",
        "residuals_ms": res,
        "ruled_out": "Not weight traffic. Both arms compile to 0.00 B off-chip at every "
            "rung in these pairs, and the FASTER model is the one carrying MORE on-chip "
            f"parameters in {heavier_faster} of the {len(pairs)} pairs.",
        "not_established": _arms_explained(),
        "matched_macs_pairs": pairs,
      },
      "off_chip_streaming_not_visible": {
        "what": "Lite2's dense rung streams 701.62 KiB and lamp -10% streams 43.31 KiB; "
            "everything below is 0 B. Extrapolating the lamp arm's own off-chip=0 fit onto "
            "those two rungs predicts them SLOWER than they measure, not faster: "
            + "; ".join(f"{s['rung']} predicted {s['predicted_ms']}, measured "
                        f"{s['measured_ms']} ({s['residual_ms']:+} ms)" for s in stream)
            + ". The repo's off-chip law (2.2561 ms/MiB, results/latency_law/"
            "latency_law_fit.json) would charge 701.62 KiB at 1.55 ms; the measurement "
            "goes the other way. So pruning Lite2 onto the chip is worth nothing "
            "measurable in latency, which is the same sign the repo-wide cliff test found.",
        "caveat": "Both rungs sit ABOVE the fitted MAC range (3361.8 and 3250.8 against a "
            "3129.9 maximum), so this is an extrapolation, and the residual std of that "
            "8-point fit is 0.612 ms. It bounds the streaming term below a few ms, it does "
            "not resolve it.",
        "control_fit": on,
      },
    }


latfind = latency_findings() if have_lat else {}

doc = {
  "measured": "2026-08-27 (export + mapping), 2026-08-28 (int8 mAP"
              + (" + on-device latency)" if have_lat else "); on-device latency NOT measured"),
  "device": "Coral USB Edge TPU (USB 3.0, 5000M), host AMD Ryzen 7 8845H / 16 threads"
            if have_lat else
            "no Coral USB attached when this file was written -- latency columns are null",
  "compiler_verbatim": "Edge TPU Compiler version 16.0.384591198",
  "what": "EfficientDet-Lite2 full-scope pruning ladder, 19 rungs, exported so that EVERY "
          "operator maps to the Edge TPU (340/340, 0 on CPU). The Lite1 counterpart is "
          "full_mapping_ladder_lite1.json; this file closes the 'int8 COCO val2017 mAP' row of "
          "the Lite1/Lite2 completeness table in docs/PRUNING_HAZARDS.md section 3."
          + ("" if have_lat else " The 'on-device latency' row of that table is still open: "
                                 "no Coral USB was attached, so tpu_ms_* are null."),
  "export": {
    "path": "PyTorch -> litert-torch (float32 tflite) -> ai-edge-quantizer static_wi8_ai8 "
            "-> buffers inlined -> edgetpu_compiler 16.0",
    "same_pool_parity": "Lite2 @448 pools 56->28->14->7->4. TF SAME pads by 1 on an even "
            "extent (asymmetric, reproduced by ceil_mode) and by 2 on an odd one (symmetric, "
            "reproduced by padding=1), so the rewrite is chosen per site from the extent's "
            "parity -- src/int8_pruning/convert/export_litert.py:_same_pool_args. A single "
            "all-or-nothing choice is wrong for Lite2: ceil_mode everywhere gives P7 3x3 "
            "instead of 4x4 (a different network), and no rewrite at all leaves a PADV2 the "
            "compiler declines, mapping only 83 of 375 operators. Lite1 @384 pools 48->24->12->6, "
            "all even, so the parity rule reproduces its previous build byte-identically.",
  },
  "protocol": {
    "mAP": "families/detection/effdet/eval.py eval, full COCO val2017 (5000 images), int8 tflite on CPU "
           "+ anchor decode/NMS, score-thresh 0.001, --input-size 448. Same harness as lite1: "
           "re-running the lite1 dense rung through it reproduced 0.31748 exactly, so the two "
           "ladders' int8 columns are directly comparable.",
    "fp32_mAP_source": "outputs/pruning_logs/efficientdet_lite2_coco-train2017_<criterion>_<pct>pct_full.json "
           "(final_eval.mAP); same values as the efficientdet_lite2 rows of full_scope_grid.json.",
    "latency": "src/int8_pruning/backends/edgetpu/bench.py in the edgetpu_ros image, 15 warmup "
           "+ 100 invokes, random input, two independent passes."
           + ("" if have_lat else " NOT RUN -- no Coral USB attached."),
  },
  "findings": {
    "mapping": "340 of 340 operators on the Edge TPU, 0 on CPU, 1 subgraph, at every rung. "
        "Reached only after the SAME-pool parity fix: the previous Lite2 export was a different "
        "network (P7 3x3), and a correct Lite2 with no pool rewrite at all maps 83 of 375.",
    "int8_cost_is_comparable_to_lite1": {
        "what": "Lite2 does NOT quantize much worse than Lite1. Measured on the same harness "
            "(re-running lite1 dense reproduced 0.31748 exactly): dense int8 loss is -1.65% on "
            "Lite2 against -1.38% on Lite1.",
        "supersedes": "full_mapping_ladder_lite1.json used to record -5.9% for the Lite2 dense "
            "rung. That figure was measured on the defective --pool-ceil export, i.e. on a network "
            "whose P7 level was 3x3 instead of 4x4. It does not describe EfficientDet-Lite2 and is "
            "withdrawn.",
        "lamp_mean_10_to_60": -1.89,
        "magnitude_l2_mean_10_to_60": -3.63,
    },
    "int8_cost_depends_on_criterion": "Same split as Lite1, and it survives quantization. Under "
        "lamp the int8 cost stays under 2.2% out to the 60% rung before kneeing (-5.25% at 80%); "
        "under magnitude_l2 it is already -3.14% at the 10% rung. In absolute terms lamp's int8 "
        "mAP at a 50% cut (0.3216) still beats magnitude_l2's int8 mAP at a 10% cut (0.2839).",
    "pruning_buys_lite2_on_chip_residency": "Unlike Lite1, which streams 0 B off-chip at every "
        "rung, Lite2 does not start on-chip: the dense rung streams 701.62 KiB and lamp -10% "
        "streams 43.31 KiB. From lamp -20% and magnitude_l2 -10% onward it is 0 B. So on this "
        "model pruning buys something Lite1 never needed"
        + (", and it is worth nothing measurable: see off_chip_streaming_not_visible."
           if have_lat else " -- how much latency that is worth is unmeasured here."),
  },
  "open": ([
    _open_channel_alignment(),
    "No I/O-floor control for Lite2. The floor is checked here against a rate measured on "
    "lite1-derived models, not against Lite2 controls of its own; the ladder cannot fit that rate "
    f"because all {len(rows)} rungs move exactly {round(IO_MIB, 4)} MiB.",
  ] if have_lat else [
    "On-device latency is NOT measured. No Coral USB was attached when this file was written, so "
    "tpu_ms_run1/run2/mean are null at every rung and no latency law is fitted. The artifacts are "
    "ready: outputs/edgetpu/single_20260827_154635/. Protocol is the same as lite1's -- "
    "src/int8_pruning/backends/edgetpu/bench.py, 15 warmup + 100 invokes, two independent passes.",
    "No channel-alignment fit and no I/O-floor control for Lite2. Both need latency.",
  ]) + [
    "Lite1 remains the deployment target for the downstream robot car (decided 2026-08-22). Lite2 "
    + ("now has all four columns -- fp32 ladder, int8 ladder, Edge TPU artifacts and on-device "
       f"latency -- so that is a choice, not a data gap. The dense rung costs "
       f"{round(rows[0]['tpu_ms_mean'] - L1_DENSE["tpu_ms_mean"], 1)} ms more per inference than Lite1's "
       f"({rows[0]['tpu_ms_mean']} against {L1_DENSE["tpu_ms_mean"]}). Roughly "
       f"{round(latfind['latency_law']['a_ms'] - L1_LAW["a_ms"], 1)} ms of that is the larger fixed "
       "I/O the 448 px input buys, which pruning cannot touch, and the rest is arithmetic "
       f"Lite2 has {round(rows[0]['mmacs'] - L1_DENSE['mmacs'])} MMACs more of. Pruning "
       f"reaches the second part only: the deepest Lite2 rung still costs "
       f"{round(min(r['tpu_ms_mean'] for r in rows) - L1_DENSE["tpu_ms_mean"], 1)} ms more than a DENSE "
       "Lite1."
       if have_lat else
       "now has a complete fp32 ladder, a complete int8 ladder and complete Edge TPU artifacts, "
       "so that is a choice rather than a data gap -- but the latency row above is still open."),
  ],
  "rows": rows,
}
doc["findings"].update(latfind)
json.dump(doc, open(OUT, "w"), ensure_ascii=False, indent=1)
open(OUT, "a").write("\n")
print(f"wrote {OUT}: {len(rows)} rows, latency={'measured' if have_lat else 'null'}")
