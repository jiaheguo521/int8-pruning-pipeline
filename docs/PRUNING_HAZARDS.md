# Four things to know before you trust a sweep from this repo

All four are established here, on the models this repo ships. Each changes what a result
means, and none of them is visible from a ratio-vs-accuracy plot.

Their scope differs, and it matters which is which. §1, §3 and §4 are properties of the
models and the toolchain: they hold wherever this pipeline runs. §2 is a property of one
Coral USB Accelerator compiled with `edgetpu_compiler` 16.0. Every latency and memory
figure in it was measured on that device and does not travel.

1. [**Allocation**](#1-allocation-global-ranking--mean-normalised-magnitude-guts-narrow-early-layers):
   one criterion/ranking combination destroys the task metric while every summary
   statistic still looks reasonable.
2. [**Latency**](#2-latency-on-a-coral-usb-cost-tracks-bytes-moved-a-steep-slope-not-a-step):
   pruning buys time in proportion to how far the *starting point* exceeds on-chip
   capacity, not in proportion to the pruning ratio. Once nothing streams at all,
   a second and much flatter cost takes over.
3. [**Criterion coverage**](#3-criterion-coverage-what-this-repo-can-and-cannot-compare):
   the repo offers five importance criteria; only two have ever been run. Which one you
   pick decides whether the published accuracy claim reproduces at all.
4. [**Quantization**](#4-quantization-this-repos-default-is-coarser-than-the-protocols):
   two families are quantized at a coarser granularity, by a different tool, than the
   protocol they are compared against. Neither publishes an accuracy number today, so the
   divergence is latent rather than damaging, but it is armed.

Everything below is measured unless marked *(inferred)*. Raw data is under
[`results/`](../results/) and each subsection names its file; the runs behind §1 and §3
are in [`results/protocol_audit/`](../results/protocol_audit/). Some claims rest on weaker
evidence than the rest, and several have been corrected; every one of them is listed in
[Provenance](#provenance-what-backs-each-number-and-which-claims-are-weaker) at the end,
not quietly dropped.

---

## 1. Allocation: global ranking × mean-normalised magnitude guts narrow early layers

`torch_pruning`'s `MagnitudeImportance` defaults to `normalizer='mean'`, which divides
each group by its own mean. Combined with `global_pruning=True`, this concentrates
almost the entire cut into the network's **narrowest early layers**, leaving the deep
wide stages nearly untouched.

Measured on three architectures, end-to-end on two of them:

| backbone | worst-hit layer | kept | task metric |
|---|---|---:|---|
| EfficientDet-Lite1 (inverted residual) | `blocks.1.2` 144 → 12 | 8.3% | mAP **0.3220 → 0.0003** |
| CLIP RN50 (plain residual) | `stages.0.0.conv2_kxk` 64 → 5 | 7.8% | zero-shot **99.0% → 0.0%** |
| Youtu Re-ID / OSNet | stem 64 → 1 | 1.6% | cos(student, teacher) 0.77 → 0.39 |

CLIP RN50 reached that at only **−30.2% parameters**, and its per-stage retention shows
the shape plainly: the deepest, widest stage keeps 84.4% while everything before it
loses more than half:

    stem 46.9%   stage0 41.1%   stage1 46.7%   stage2 40.5%   stage3 84.4%

> **The trigger is layer WIDTH, not the block type.** A plain ResNet fails the same way
> an inverted-residual backbone does. Do not write "on depthwise-separable backbones":
> that qualifier is measurably wrong.

Evidence: [`results/clip_rn50/global_allocation.json`](../results/clip_rn50/global_allocation.json)
for the CLIP row; [`results/detection/pruning_matrix.json`](../results/detection/pruning_matrix.json)
(`allocation_control_lite1`) for the EfficientDet row. The Re-ID row is recorded only as a
code comment; see the provenance table at the end.

### Why

Not cross-layer scale: `normalizer='mean'` already cancels that (all 27 EfficientDet
backbone groups measure mean exactly 1.0000), and the raw L2 norms run the *other* way
(`blocks.1.0.conv_dw` 1.0126 vs `blocks.5.1.conv_dw` 0.6121, early above late under every
reduction measured).

The driver is **within-layer skew after normalisation**. Narrow early layers have a long
left tail (`blocks.1.0.conv_dw`: 59.4% of channels below 0.5 relative importance, CV
1.623) while late wide layers have almost none (`blocks.5.4.conv_dw`: 2.7%). A single
global threshold therefore sweeps one narrow layer to the bone before touching anything
else.

Evidence: [`results/detection/allocation_stats.json`](../results/detection/allocation_stats.json),
regenerate with [`results/protocol_audit/allocation_stats.py`](../results/protocol_audit/allocation_stats.py)
(allocation only, no forward pass, about a minute on CPU).

### Four consequences that are easy to get wrong

All four are measured in
[`results/detection/pruning_matrix.json`](../results/detection/pruning_matrix.json)
(`allocation_control_lite1`, `per_layer_floor_sweep_lite1`, `damage_saturation_lite1`,
`recovery_finetune_lite1`); the runs that produced them are in
[`results/protocol_audit/`](../results/protocol_audit/).

**`global_pruning=True` is not the dividing line.** The same global allocator with a
different criterion does not collapse; it can beat the local control while cutting its
worst layer *deeper*. EfficientDet-Lite1, backbone −10%, no fine-tuning:

    global + magnitude_l2   worst layer  8.3% kept    mAP 0.0003
    global + random         worst layer 87.5% kept    mAP 0.0674   (*)
    global + lamp           worst layer 80.5% kept    mAP 0.1092   <- best
    local  + magnitude_l2   worst layer 91.7% kept    mAP 0.0820

The claim is about the **combination**, not about global ranking on its own.

(*) The `random` row came from an ad-hoc control script that was never committed, so it
is **not reproducible from this repo**; see
[§3](#3-criterion-coverage-what-this-repo-can-and-cannot-compare). The three rows around
it are worker-produced and are.

**Raising the per-layer floor is not a fix.** `max_pruning_ratio` is per-*group*, not
per-layer; it is a pre-step admission gate with no clamp, and it also applies to input
channels. The realised floor is **8.33%, not the 10% the value suggests**. Tightening it
changes how deep the worst cut goes and never changes where it lands, measured at
backbone −10%, no fine-tuning:

    cap 0.9   min survival  8.3%   mAP 0.0003
    cap 0.5   min survival 37.5%   mAP 0.0000   <- worse than 0.9
    cap 0.3   min survival 56.2%   mAP 0.0003
    cap 0.1   min survival 76.0%   mAP 0.0091

At every cap the five worst-hit layers are all in stages 1–2. `cap=0.1` reaches a
minimum survival comparable to LAMP's and is still 12× worse in mAP, because LAMP does
not touch the early stages at all.

**The post-prune metric is not an estimator of recoverability.** It overstates the damage
by up to 180×, and it gets the *ordering* wrong:

    lamp   g=True   pre-FT 0.1092  ->  post-FT 0.3027   (94.0% of baseline)
    mag_l2 g=True   pre-FT 0.0003  ->  post-FT 0.2043   (63.4%)
    full -44%       pre-FT 0.0014  ->  post-FT 0.2555   (79.3%)

Pre-fine-tune, LAMP leads the local control by 33%; after recovery that collapses to
2.5%, and the local arm got there while pruning *more*. Never truncate a grid, pick a
Pareto point, or headline a ratio on pre-fine-tune numbers.

**A flat post-prune metric across ratios is the fingerprint of this pathology,** not
evidence of robustness. The damage saturates by ~2%: deleting the right 80,594
parameters (1.90% of the model) takes mAP from 0.3220 to 0.0002, so 10 / 20 / 30% all
read the same ~0.0003. At that point the worst-hit layer is `blocks.1.0.conv_pw`,
96 → 17 channels, a shallower cut than the 8.3% above, and already fatal.

### How to check a run in 20 seconds

No forward pass, no fine-tuning:

```python
import torch, torch.nn as nn
dense  = torch.load("outputs/models/<base>.pt", weights_only=False, map_location="cpu")
pruned = torch.load("outputs/pytorch_pruned/<base>_pruned<P>pct_<imp>.pt",
                    weights_only=False, map_location="cpu")
convs = lambda m: {n: c for n, c in m.named_modules() if isinstance(c, nn.Conv2d)}
D, P = convs(dense), convs(pruned)
worst = sorted(((P[n].out_channels / D[n].out_channels, n) for n in D if n in P))[:5]
for k, n in worst:
    print(f"{n:<44} {D[n].out_channels:>5} -> {P[n].out_channels:<5} {k:.1%}")
```

If the worst layers are in the stem or the first stage and are down to single-digit
percentages, the run is measuring the pathology, not the method.

### Where each family stands

`PRUNING_PROTOCOL` is deliberately kept local to each `families/<task>/<name>/prune.py` so the
choice is visible at the point of use.

| family | `global_pruning` | note |
|---|---|---|
| `line_seg`, `reid` | `False` | switched after this was measured; `reid` exposes a `--global_pruning` CLI flag |
| `effdet`, `ssdlite` | `True` | kept: reproducing the published protocol faithfully **is** the finding |
| `imagenet_backbones`, `clip_rn50` | `True` | unchanged; see the caveat below |

**Caveat on `imagenet_backbones`.** `resnet50_imagenet` has only been audited statically
(prune bookkeeping, no forward pass). Its allocation is pathological in the same way,
but its published top-1 is *unverified*, not shown wrong. Do not cite it as refuted.

---

## 2. Latency on a Coral USB: cost tracks bytes moved, a steep slope, not a step

Fitted across **30 compiled models spanning three backbones** (0.08–27 MiB of weights),
each timed with 100 invokes on a Coral USB Edge TPU:

```
lat_ms = 2.256 · offchip_MiB + 0.915 · onchip_MiB + 0.852          R² = 0.9963
```

A streamed byte costs **2.47×** a cached byte. That ratio is the whole mechanism.

**The boundary is not a step.** Adding an indicator for "off-chip > 0" to that fit buys
nothing: coefficient **−0.243 ms at F(1,26) = 0.10** (p = 0.75), and its sign is the
opposite of a penalty: streaming models come in marginally *faster* than the continuous
law predicts, not slower. Refitting on the ResNet-50 appendix table of the DSD 2026 paper
whose protocol this repo follows (64 rows: 63 pruned configurations plus the unpruned baseline) gives
the same answer: step-indicator coefficient **−0.030 ms at F(1,61) = 0.09**.

The same table carries the positive result, and it is the stronger of the two. Latency
there is linear in off-chip bytes to **R² = 0.9998**:

```
lat_ms = 3.246 · offchip_MiB + 1.316                                R² = 0.9998
```

So the paper's mechanism, *these transfers dominate the latency*, is not merely right,
it is tighter on its own data than it was stated to be. What the data do not support is a
**discontinuity** at the crossing.

> A threshold decides **how many bytes have to stream**. It does not decide what a
> streamed byte costs. The knee people see in a speedup plot comes from
> `off ≈ max(0, size − cap)` composed with a reciprocal, not from a discontinuity in
> latency. ResNet-50 going 60% → 70% is the clean case: the speedup reads 5.95× → 34.02×,
> which looks like a cliff, while the latency it is computed from goes 8.56 → 1.50 ms,
> within 0.2 ms of what the straight line predicts on both sides.

**One caveat, because the answer depends on how you define "crossed over".** Defining the
step as `size < 8 MB` rather than `off-chip > 0` *does* give a significant term: +2.535 ms
(F = 14.2) on the panel above, +0.309 ms (F = 9.7) on the paper's table. Read the sign. It
is **positive**: models that fit are *slower* than the law predicts, not faster, so it
is still not a cliff. It is the compute floor: the law carries only weight-transfer terms,
so once off-chip reaches 0 it has nothing left to charge the fixed compute cost to, and
the indicator absorbs it. That is the same effect as the applicability limit below.

Evidence: [`results/line_seg/param_ladder_oncar.json`](../results/line_seg/param_ladder_oncar.json)
and [`results/dsd2026_reference/resnet50_table.csv`](../results/dsd2026_reference/resnet50_table.csv).
Every coefficient, R² and F above is regenerated by
[`results/latency_law/fit_latency_law.py`](../results/latency_law/fit_latency_law.py)
(stdlib only; `--check` asserts the values printed here) into
[`results/latency_law/latency_law_fit.json`](../results/latency_law/latency_law_fit.json).

### Applicability limit

The fit has **only weight-transfer terms**. Below roughly 1 MiB of parameters, compute
takes over and it under-predicts by 0.4–0.5 ms, on four `line_seg_w96` rungs measured
1.383–2.097 ms against predictions of 0.987–1.592 ms. Two models with almost no weights
left still differ 2× (`base128` 0.640 ms at 128² vs `w96_p99` 1.383 ms at 192²), and the
difference is input resolution, not bytes.

That is a limit of magnitude. The harder limit is one of **scope**: the fit has only
weight-transfer terms, so in a family where nothing streams at all it has nothing to act
on. Substituting EfficientDet-Lite1's unpruned rung (0.00 B off-chip, 6.29 MiB on-chip)
gives `2.256·0 + 0.915·6.29 + 0.852 = 6.61 ms` against **52.404 ms measured**, under by
**7.9×**. A different and much flatter relation governs there; the next subsection is what
it turned out to be.

Use it as a weight-transfer model in the regime where transfer dominates. Do not present
it as a general latency model.

Evidence: [`results/line_seg/w96_deep_rungs.json`](../results/line_seg/w96_deep_rungs.json),
[`results/detection/full_mapping_ladder_lite1.json`](../results/detection/full_mapping_ladder_lite1.json).

### What pruning actually buys, per family

Same device, same criterion, same parameter axis. Latency is the `tpu_ms_mean` column of
[`param_ladder_oncar.json`](../results/line_seg/param_ladder_oncar.json), the litert-torch
build, the same column the law above is fitted on. That file also carries a
`tpu_ms_mean_onnx2tf` column from the earlier export; the two are not comparable and this
table used to quote it by mistake.

| model | baseline int8 | latency range | **speedup** |
|---|---:|---|---:|
| `line_seg_base128` | 0.26 MB (already fully on-chip) | 0.807 → 0.640 ms | **1.26×** |
| `line_seg_w96` | 8.71 MB (sits on the cap) | 12.698 → 2.281 ms | **5.57×** |
| `line_seg_link_r34` | 28.8 MB (far above the cap) | 55.677 → 3.704 ms | **15.03×** |

**The return on pruning is proportional to how far the starting point exceeds on-chip
capacity.** A model that already fits gains only what its MAC cut buys, bounded below by
its family's fixed costs: 1.26× on `line_seg_base128` here, and 1.29× on
EfficientDet-Lite1 for an 88.7% parameter cut (next subsection).

### Detection: nothing streams, so a different cost decides

Two independent Coral passes on the earlier **backbone**-scope rungs, the pre-2026-08-19
semantics; see the ratio-semantics note at the end of this document
([`results/detection/coral_latency.json`](../results/detection/coral_latency.json)). This
table is kept because it is what that file holds; it was produced by the **onnx2tf export
that was retired on 2026-08-22**:

| model | 0% | 30% | 50% | speedup | off-chip |
|---|---:|---:|---:|---:|---|
| `efficientdet_lite1` | 33.43 ms | 25.96 | 24.77 | **1.35×** | **0.00 B at every rung** |
| `ssdlite_mobilenetv3` | 18.09 ms | 16.82 | 13.87 | **1.30×** | ≈ 0 |

EfficientDet-Lite1 is fully on-chip **before any pruning**: 4.62 MiB under that retired
compile, 6.29 MiB once the whole graph is mapped, and 0.00 B off-chip either way and at
every rung. There is no weight streaming to eliminate, so the expected gain from the
8 MiB streaming-elimination mechanism is exactly 0 ms. That part was right, and it still
is. What was wrong was the reason given for the rest.

**The operator split was an artifact of the export tool, not a property of the
architecture.** That earlier build put 389 of 479 operators on the CPU because onnx2tf
lowers effdet's BiFPN fusion to a rank-5 `CONCAT`/`SUM` and timm's SAME-pool to `PADV2`,
neither of which the Edge TPU compiler can map, so the TPU subgraph stopped at the
backbone. Re-exported through litert-torch + ai-edge-quantizer, with two arithmetically
identical model-side rewrites (effdet's `FpnCombine` stack+sum as sequential adds; timm's
`MaxPool2dSame` with `ceil_mode=True`), the same checkpoints compile to **305 of 305
operators on the TPU and none on the CPU, at every one of 19 rungs**. Coral's own
`efficientdet_lite1_384_ptq` is 315/7 for comparison. Do not read 479 → 305 as a pruning
result: they are two lowerings of the same network.

With the whole graph on the accelerator, what binds is neither weight memory nor operator
partitioning. It is the accelerator's **fixed I/O**:

```
lat_ms = 34.95 + 9.286 · MMACs/1000        R² = 0.9352   n = 19
```

**34.95 of the unpruned rung's 52.404 ms, 67%, is a floor pruning cannot reach.** Each
inference moves 0.42 MiB in and 2.48 MiB out, and the head emits 810 channels at every
rung, pruned or not. The return path alone measures ~81 MB/s, which puts 2.48 MiB at
32.0 ms, or 92% of that floor. Across the full 19-rung ladder, a 3.4× cut in MACs and an
88.7% cut in parameters, latency falls only 52.404 → 40.773 ms, **1.29×**. Latency is not
even monotone in MACs at the top: the `lamp` −10% rung measures 52.995 ms, above the
unpruned one.

**What full mapping buys is a clean measurement.** With nothing on the CPU, latency is a
property of the accelerator and the model, and the host cannot enter it. The backbone-scope
table at the top of this subsection is the counter-example: 389 of its 479 operators ran on
a host CPU that `coral_latency.json` records as *not captured*, so those numbers are partly
a measurement of an unidentified desktop, and comparing rungs on them compared bench
conditions alongside models. Every rung of the 19 is one Edge TPU subgraph with zero CPU
operators, which is why they are comparable to each other and to any other Coral. It is
also why a partial map being *faster here* (26.96 ms against 52.78 ms, by returning
0.26 MiB instead of 2.48 MiB) is not the argument it looks like: that speed is bought back
with the confound, and on a weaker host it is not obviously bought at all. Which split a
given robot should ship is a question about that robot's host.

**That the floor is I/O is measured, not assumed, and this ladder could not have shown
it.** All 19 rungs return exactly 2.48 MiB and accept exactly 0.42 MiB, so the variable has
*zero variance* along the pruning axis and no fit on these rungs can attribute the
intercept to anything. A separate control varies it: two models with the same on-chip
footprint (4.89 vs 4.97 MiB), no off-chip traffic and one subgraph each, differing by one
1×1 conv 88→810 and by what the TPU returns, measure **22.75 → 42.55 ms**. Charging that
conv's own 164.2 MMACs at the fit's 9.286 µs/MMAC leaves **18.3 ms on the return path for
1.586 MiB, 86.8 MiB/s**. A third model produces the identical wide tensor on the *host*
(one `RESIZE` falls back to CPU, so the narrow tensor is what crosses USB) and costs
**+0.22 ms**. At 86.8 MiB/s the 2.90 MiB each inference moves is 33.4 ms, or **96% of the
34.95 ms floor**.

*(inferred)* So on a fully mapped detector the lever is the head's output width, not the
backbone's weights. That step is inference: no model in this repository was pruned at the
head.

### Below 64 channels, pruning stops buying time

Recomputing MACs on channel counts rounded **up** to a tile width `W` improves the
EfficientDet fit, with a clear peak at `W = 64`, the Edge TPU's systolic array width:

| W | 1 | 8 | 16 | 32 | **64** | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R², Lite1 @384 | 0.9367 | 0.9356 | 0.9582 | 0.9697 | **0.9920** | 0.9490 | 0.6124 |
| R², Lite2 @448 | 0.9518 | 0.9516 | 0.9566 | 0.9695 | **0.9829** | 0.9581 | 0.8661 |

**Both models peak at 64**, at two input resolutions and two model scales. That is still
one architecture family, so it is not proof the peak is universal — but it is no longer a
number measured once.

Raw MACs span 3.4× across the ladder (1954 → 574 MMACs) but padded-64 MACs span only 1.6×
(4892 → 3041), and latency spans 1.29×. Two rungs 18% apart in raw MACs but 1.7% apart
padded-64 measure 43.45 and 43.62 ms, 0.4% apart. **Channels cut below the array width
cost accuracy and buy no time.**

Counting the output channels of all 213 convs, the unpruned rung has **6.6% below 64
channels and 95.3% at a multiple of 8**; the deepest rung has 24.4% and 67.6%. Pruning
moves the distribution across the tile boundary, which is what the padded-MAC series picks
up.

This is a property of the device, not of EfficientDet, so it bears on any family with
narrow layers: `line_seg_base128` is (16, 32, 64, 96) *before* any pruning
([`families/segmentation/line_seg/lane_seg_models.py`](../families/segmentation/line_seg/lane_seg_models.py)). It has
not been tested at layer granularity anywhere; the evidence is two whole-model fits and the
two-rung comparison above.

Note the baseline in that table: `W = 1` reads 0.9367 rather than the 0.9352 of the fit
above it, because the padded series recounts MACs by tracing every `Conv2d` through the
model while the `mmacs` column comes from the cluster logs. Compare 0.992 against 0.9367.

Evidence: [`channel_census_lite1.json`](../results/detection/channel_census_lite1.json) and
[`channel_census_lite2.json`](../results/detection/channel_census_lite2.json),
regenerate with [`results/protocol_audit/channel_census.py`](../results/protocol_audit/channel_census.py)
`--model {lite1,lite2}` (`--check` asserts each model's peak and the R² printed here; needs
`torch` and the gitignored checkpoints). Latency comes from
[`results/detection/full_mapping_ladder_lite1.json`](../results/detection/full_mapping_ladder_lite1.json);
the export that produced it is in
[`results/protocol_audit/export_litert/`](../results/protocol_audit/export_litert/).

### A second detector, and what it does to the account above

Everything so far in this section is EfficientDet-Lite1. A second model, **Lite2 at 448 px,
19 rungs, 340/340 operators mapped**, was measured on 2026-08-28 through the same harness,
and it tests three of the claims above rather than restating them.

**The floor is I/O, checked out of sample.** Lite2's ladder fits
`lat_ms = 46.43 + 9.750 · MMACs/1000` at R² = 0.9503. That intercept was not fitted to the
I/O account: every Lite2 rung moves **3.95 MiB** across USB, and charging it at the
**86.8 MiB/s** measured on the *Lite1-derived* control models above — no refit, no new
constant — gives **45.48 ms, 98% of the fitted floor**. The floor rises 1.33× from Lite1's
34.95 ms where the traffic rises 1.36×. A number derived on one model predicted another
model's intercept to two percent, which is the strongest form this claim has been put in.

**No off-chip cliff, on a second architecture.** Unlike Lite1, Lite2 does not start
resident: its dense rung streams 701.62 KiB and `lamp` −10% streams 43.31 KiB, reaching 0 B
from `lamp` −20% onward. So this model has the transition Lite1 never had. Extrapolating
the `lamp` arm's own off-chip-free fit onto those two streaming rungs predicts them
**slower** than they measure, by 2.53 ms and 1.34 ms, where this repository's off-chip law
would have charged 701.62 KiB at 1.55 ms — the wrong sign for a streaming penalty, and the
same wrong sign the repository-wide cliff test produced. *(inferred)* This is an
extrapolation beyond the fitted MAC range on two rungs, so it **bounds** the streaming term
rather than resolving it.

**But MACs do not fix latency inside one model.** The pooled fit's R² of 0.9503 hides
structure: all nine `lamp` rungs sit above the line and seven of nine `magnitude_l2` rungs
below it. Fitted separately the arms reach R² **0.9917** and **0.9703** on slopes 2.0
µs/MMAC apart, and at matched MAC counts `magnitude_l2` runs up to **5.48 ms faster** —
while carrying *more* on-chip weight in six of the seven matched pairs, with both arms
streaming 0 B. Weight traffic is therefore ruled out. What is left is the shape of what
survives — which is exactly the channel-alignment fit of the previous subsection.

**And that fit says the split is mostly an artifact of the axis.** A channel census of the
Lite2 ladder ([`channel_census_lite2.json`](../results/detection/channel_census_lite2.json),
229 convs traced per rung at 448 px) lets latency be fitted against MACs recounted with
channel counts rounded up to the tile width. Padding is the only thing that changes — MACs
are recounted by tracing the model on both axes — and it takes the gap between the arms'
slopes from **1.964 to 0.088 µs/MMAC, a factor of 22**:

| | raw MACs | padded to 64 |
|---|---:|---:|
| pooled R² | 0.9518 | **0.9829** |
| `lamp` slope, µs/MMAC | 9.928 | 6.752 |
| `magnitude_l2` slope | 7.964 | **6.664** |
| arms' slope gap | 1.964 | **0.088** |
| mean residual separation | 2.44 ms | **1.14 ms** |

The raw column reads 1.964 where the paragraph above says 2.0, for the reason the previous
subsection's `W = 1` note gives: this table recounts MACs by tracing every `Conv2d`, while
the ladder's `mmacs` column comes from the cluster logs. Both columns here are traced, so
padding is the only difference between them.

So the two arms are not obeying different laws. They leave different channel counts, and
the raw-MAC axis cannot see the padding: pairs matched to within 5% on raw MACs sit up to
**12.3%** apart once padded, and the pair furthest apart on the padded axis is the pair with
the largest latency gap. **What does not go away is a level offset** — the arms' mean
residuals still differ by 1.14 ms and `lamp` still sits above the pooled line at eight of
nine rungs. That remainder is not explained here.

The same test on Lite1 runs the same way, more weakly: slope gap 2.3889 → **0.3909** µs/MMAC,
separation 1.64 → 0.35 ms. Two models, one direction.

**Read the two fits as answering different questions.** The raw-MAC fit's intercept is the
I/O floor, and it is the one the 86.8 MiB/s control validates to 98%. The padded-MAC fit
explains variation *between* rungs far better, but its intercept is **29.82 ms** on Lite2
and 20.77 on Lite1 — not the floor, and not offered as one. Quote 46.43 and 34.95 for what
an inference costs before compute; quote the padded fit for what pruning a channel buys.

Evidence: [`results/detection/full_mapping_ladder_lite2.json`](../results/detection/full_mapping_ladder_lite2.json),
whose `latency_law`, `MACs_do_not_fix_latency` and `off_chip_streaming_not_visible`
findings are recomputed from the rows by
[`results/protocol_audit/assemble_ladder_lite2.py`](../results/protocol_audit/assemble_ladder_lite2.py).
The two ladders are fitted separately and never pooled: one line through both would average
two I/O floors into a number describing neither model.

### Reporting model size

Use the **compiled** `_edgetpu.tflite`, never the intermediate `_int8.tflite`. Under the
onnx2tf export, the default until 2026-08-22 and still selectable, 54–64% of the intermediate file
**was** tensor-name strings: onnx2tf lowers each BiFPN depthwise conv into
`SPLIT → 2816 single-channel CONV_2D → CONCAT`, and MLIR names every split output with the
fused location list of all 88 siblings (longest 2454 chars). Those bytes were a fixed floor
that did not shrink with pruning, because BiFPN is in `ignored_layers`. A 50% backbone cut
read as −13.8% on the intermediate file and −39.8% on the compiled one; the latter tracks
the −43.9% full-model parameter cut.

**That ratio is a property of the toolchain, not of EfficientDet. The rule is not.**
The current litert-torch export of the same network carries 7.3% name bytes (678 tensors,
305 operators) against the old 65% (11,916 tensors, 3,171 operators). Both paths are now in
`tflite_size_audit.json`: 77 rows on each path across the 154 audited, each tagged
with an `export_path` field, because `packaging_frac` is not comparable across the two. Always
report the compiled artifact anyway:
what the intermediate over- or under-states varies by export path, and only the compiled
file is what the device loads.
See [`results/tflite_size_audit.json`](../results/tflite_size_audit.json) for the
intermediate byte budget and [`results/detection/compiled_size.json`](../results/detection/compiled_size.json)
for the compiled sizes; the operator lowering and the 2,454-character names are documented
in [`src/int8_pruning/report/tflite_size.py`](../src/int8_pruning/report/tflite_size.py), which produced
the former.

---

## 3. Criterion coverage: what this repo can and cannot compare

### Five criteria are offered; two have been run

`ALL_IMPORTANCES = ["magnitude_l1", "magnitude_l2", "fpgm", "random", "lamp"]`, but the
evidence behind them is very uneven. Counted over every worker-produced run log, 121
rungs across five log directories, local and cluster:

<!-- generated: census -->
| criterion | rungs | local / cluster | sweep `.pt` | fine-tuned `.pt` | families | status |
|---|---:|---|---:|---:|---|---|
| `magnitude_l2` | **101** | 79 / 22 | 62 | 3 | `clip_rn50`, `effdet`, `line_seg`, `reid`, `ssdlite` | every family except `imagenet_backbones` and `relu_clip` |
| `lamp` | **20** | 3 / 17 | 18 | 4 | `effdet`, `reid` | detection and Re-ID only |
| `random` | 0 | — | 0 | 0 | — | measured **only** in ad-hoc control scripts that were never committed; the numbers are on record but are **not reproducible from this repo** |
| `magnitude_l1` | 0 | — | 0 | 0 | — | implemented, never executed |
| `fpgm` | 0 | — | 0 | 0 | — | implemented, never executed |
<!-- /generated -->

`imagenet_backbones` is the one family with **no committed measurement of any kind**: no rung
in the census, no `results/` directory, no operator split, no byte figure. Anything this
document says about it is inherited from the families that were measured, not measured.
`relu_clip` is the other family absent from the census, for a different reason: it has
byte, latency and int8-cost figures, but they were measured upstream rather than produced
by a rung of this pipeline, so no run log exists here for the census to count.

The rung counts deduplicate: the same rung can appear both as a cluster log and as the
copy pulled down into `outputs/pruning_logs`, and once as a 0-epoch pre-recovery eval.
Every dropped file is listed in the JSON's `superseded_duplicates`, and ties are attributed
to the cluster, where the run happened.

Evidence: [`results/criterion_census.json`](../results/criterion_census.json), recounted by
[`results/protocol_audit/criterion_census.py`](../results/protocol_audit/criterion_census.py).
The two `.pt` columns are deliberately separate: sweep checkpoints answer "was this rung
produced", fine-tuned ones answer "was it recovered", and they live in different
directories. An earlier hand count reported 89 and 13 rungs by conflating them.

A sweep in this repo is a `magnitude_l2` sweep unless its filename says otherwise. Treat
`magnitude_l1`, `fpgm`, and `random` as untested code paths.

### This repo covers four of the protocol's seven criteria, for two different reasons

The DSD 2026 evaluation protocol this repo follows names seven criteria: `magnitude_l1`,
`magnitude_l2`, `bn_scale`, `fpgm`, `taylor`, `obdc`, `random`.

| | criteria |
|---|---|
| in both | `magnitude_l1` `magnitude_l2` `fpgm` `random` |
| **in the protocol, not run here** | `bn_scale` `taylor` `obdc` |
| **added here, not in the protocol** | `lamp` |

The three this repo does not cover are `bn_scale`, `taylor` and `obdc`, the criteria
that carry the CIFAR-100 study's high-ratio accuracy results. **They are absent for two
different reasons, and the distinction matters more than the absence does.**

**Not evaluated, compute budget.** `magnitude_l1`, `fpgm`, `taylor` and `random` would
all run on this bench as-is. Measured cost of one criterion, from `duration_s.total` in
the grid logs:

    efficientdet_lite1   5.16 h / rung  (40 ep, full COCO train2017)
    efficientdet_lite2  14.90 h / rung
    one criterion, 9 rungs x 2 models   ~181 GPU-hours

Two criteria are on record, and since 2026-08-26 all 36 of their rungs are **measured
rather than projected: 361 GPU-hours spent.** Adding the four runnable ones costs a
further **~722 GPU-hours; the full panel is 3x what has been spent.** That is the
reason, and it is checkable; "not enough time" is not.

Evidence: [`results/detection/full_scope_grid.json`](../results/detection/full_scope_grid.json)
(`cost_per_criterion`).

**Not evaluable: this bench does not expose what they need.** These two are *not* a
budget question:

* **OBD-C** samples labels from the class posterior. `effdet`'s `DetBenchTrain` returns
  only `{loss, class_loss, box_loss}`, so there is no posterior to sample. A detection
  bench that exposed one would run OBD-C fine.
* **`bn_scale`** presumes a 100-epoch sparsity pre-training stage. That is a different
  training pipeline, not a different pruning flag, and this repo never built one.

> *(inferred)* **Some importance criteria need more from a task than a loss value, and
> that is what limits transfer.** Both criteria above are perfectly well-defined on
> CIFAR-100 classification, where they were established; what they require, a class
> posterior and a sparsity pre-training stage, simply is not on offer here. **Report this
> as a finding about what a protocol needs from a task, not as a gap in coverage**:
> filing it under "no time" throws away the strongest observation in this section and
> leaves you with nothing to say when someone asks whether you tried OBD-C.

### What the detection grid does and does not license

The paper's headline accuracy claim, accuracy essentially maintained to roughly 60%
parameter reduction, was established on **CIFAR-100 classification**. Everything below is
a different task on a different architecture, so what it can settle is the *range* of the
protocol, not whether the published result holds where it was published.

EfficientDet-Lite1, full-model parameter reduction, 40 epochs of recovery on full
COCO train2017, evaluated on val2017 (5000 images). Baseline mAP **0.3220**:

<!-- generated: ladder_lite1 -->
| reduction | `lamp` | of baseline | `magnitude_l2` | of baseline |
|---:|---:|---:|---:|---:|
| 10% | 0.3180 | 98.7% | 0.2624 | 81.5% |
| 20% | 0.3141 | 97.5% | 0.2355 | 73.1% |
| 30% | 0.3105 | 96.4% | 0.2048 | 63.6% |
| 40% | 0.3044 | 94.5% | 0.1793 | 55.7% |
| 50% | 0.2934 | 91.1% | 0.1525 | 47.4% |
| **60%** | **0.2769** | **86.0%** | **0.1421** | **44.1%** |
| 70% | 0.2462 | 76.5% | 0.1182 | 36.7% |
| 80% | 0.1900 | 59.0% | 0.1067 | 33.1% |
| 90%† | 0.0906 | 28.1% | 0.0641 | **19.9%** |
<!-- /generated -->

† Target 90%, **not realized**: the per-layer floor saturated with the backbone already at
−98.6%, so neither arm could cut that deep. `lamp` stopped at **88.5554%**, `magnitude_l2`
at **88.7362%**: two different models, and neither is the 90% the column header asks for.
Compare on realized reduction, never on the target.

On this detection grid:

* Under **`magnitude_l2`**, which is in the protocol's set, the claim **does not carry
  over**: 44.1% of baseline at 60%, and the ladder keeps falling to 19.9% at the top rung.
* Under **`lamp`**, which is **not** in the protocol's set, it **does**: 86.0% at 60%.
* Under `taylor` / `obdc` / `bn_scale`, the criteria that carry the CIFAR-100 result at
  high ratios, it is **untested here**, for the reasons above.

Evidence: [`results/detection/full_mapping_ladder_lite1.json`](../results/detection/full_mapping_ladder_lite1.json)
All 18 cells above, plus the dense baseline, with the int8 column and the operator
mapping alongside. The grid this table was first read off,
[`results/detection/full_scope_grid.json`](../results/detection/full_scope_grid.json)
(36 of 36 rungs since 2026-08-26, the EfficientDet-Lite2 half below), keeps the
pre-fine-tuning column and marks itself superseded for Lite1.

**The same grid on EfficientDet-Lite2 (448 px, baseline mAP 0.3589), finished 2026-08-26.**
Same pruning code, same seed, same 40-epoch recipe, on a second model at a different input
resolution, run to all nine rungs under both criteria:

<!-- generated: ladder_lite2 -->
| realized reduction | `lamp` | of baseline | `magnitude_l2` | of baseline |
|---:|---:|---:|---:|---:|
| 10.2% | 0.3512 | 97.8% | 0.2931 | 81.7% |
| 20.0% | 0.3492 | 97.3% | 0.2685 | 74.8% |
| 30.1% | 0.3466 | 96.6% | 0.2378 | 66.3% |
| 40.0% | 0.3408 | 95.0% | 0.2174 | 60.6% |
| 50.0% | 0.3274 | 91.2% | 0.1933 | 53.9% |
| **60.0%** | **0.3044** | **84.8%** | **0.1587** | **44.2%** |
| 70.1% | 0.2669 | 74.4% | 0.1401 | 39.0% |
| 80.1% | 0.1874 | 52.2% | 0.1283 | 35.8% |
| 84.4%‡ | 0.1257 | 35.0% | 0.1046 | 29.1% |
<!-- /generated -->

‡ Same saturation as Lite1's bottom row, at a *different* reduction: Lite2 protects a
larger ignored fraction (0.170 of parameters vs Lite1's 0.124), so its floor binds at
**84.4%** where Lite1's binds at 88.6%. The two bottom rows are not the same cut and must
not be read as one.

**This is the strongest thing the detection grid now says.** Rung for rung on matched
realized reduction, the `lamp` arm tracks across models to within **2.1 points** of retained
mAP at every rung from 10% to 70%. The `magnitude_l2` arm is looser: it agrees at 10/20/60/70%
but Lite2 sits **3 to 6.5 points higher** through the middle of the ladder (30–50%), so the
criterion's damage is somewhat model-dependent in size. What is not model-dependent is the
ordering and its magnitude: `lamp` is far above `magnitude_l2` at every one of the nine rungs
on both models, and at 60% the two grids nearly coincide, `magnitude_l2` retaining **44.1%**
(Lite1) and **44.2%** (Lite2) of dense mAP, `lamp` retains **86.0%** and **84.8%**. The
pre-fine-tuning collapse signature replicates too: global `magnitude_l2` never leaves the
range **0.0002–0.0004** (Lite1) / **0.0001–0.0002** (Lite2) at *any* rung, −10% included,
while `lamp` still has **0.121** (Lite1) / **0.154** (Lite2) left at −10%, stays an order of
magnitude above the floor through −30%, and only reaches it at the deepest two rungs. Both
models decay through the same sequence. So "the protocol criterion does not carry
over and LAMP recovers it" is no longer a single-model observation.

Say what the replication does and does not span. Two models, one architecture family, one
criterion pair, one pruning implementation. It is a replication **across scale and input
resolution**, not across architectures.

**The two models are now finished to the same depth.** Every column exists on both:

| | Lite1 @384 | Lite2 @448 |
|---|---|---|
| fp32 ladder, 19 rungs | ✅ | ✅ 2026-08-26 |
| int8 + Edge TPU artifacts | ✅ 305/305 mapped | ✅ 2026-08-27, 340/340 mapped |
| **int8 COCO val2017 mAP** | ✅ 19 rungs | ✅ **2026-08-28, 19 rungs** |
| **on-device latency** | ✅ 19 rungs | ✅ **2026-08-28, 19 rungs** |
| **channel census / tile-width fit** | ✅ 2026-08-22 | ✅ **2026-08-28, 19 rungs** |

Both new columns went through the harness Lite1's went through, so the two models compare
directly. Latency is 15 warmup + 100 invokes on the same Coral, two independent passes that
agree rung for rung to within **0.7%**; the int8 evaluator reproduced **0.31748** exactly on
the Lite1 dense rung before the Lite2 column was run.

**And the int8 comparison retires a claim this document used to carry.** Lite2 does *not*
quantize far worse than Lite1: the dense int8 cost is **−1.65%** on Lite2 against
**−1.38%** on Lite1. The old −5.9% figure was measured on the defective `--pool-ceil`
export, a network whose P7 level was 3×3 instead of 4×4, so it never described
EfficientDet-Lite2 at all. The criterion split, by contrast, does survive quantization on
both models: averaged over the 10–60% rungs the int8 cost is **−1.89%** under `lamp`
against **−3.63%** under `magnitude_l2`, and in absolute terms `lamp`'s int8 mAP at a 50%
cut (0.3216) still beats `magnitude_l2`'s at a 10% cut (0.2839).

One asymmetry worth keeping, because it bears on what the two models can be compared on:
Lite1 streams 0 B off-chip at *every* rung, while Lite2 does not start on-chip. Its dense
rung streams **701.62 KiB** and `lamp` −10% streams 43.31 KiB, reaching 0 B from `lamp`
−20% and `magnitude_l2` −10% onward. On this model pruning buys on-chip residency that
Lite1 never needed.

What that residency is worth in milliseconds, what Lite2's own I/O floor says about the
account section 2 built on Lite1, and why equal MAC counts do not buy equal latency, are
all questions about latency rather than about criterion coverage. They are argued where
that account lives, in
[§2](#2-latency-on-a-coral-usb-cost-tracks-bytes-moved-a-steep-slope-not-a-step), under
*A second detector, and what it does to the account above*. The short of it: the residency
is worth nothing measurable; the floor scales with the output tensor to within 2%; and the
criterion does move latency at matched raw MACs, but a channel census shows most of that is
the padding the raw-MAC axis cannot see, not two different laws.

Evidence: [`results/detection/full_scope_grid.json`](../results/detection/full_scope_grid.json)
(fp32 ladder) and [`results/detection/full_mapping_ladder_lite2.json`](../results/detection/full_mapping_ladder_lite2.json)
(int8 mAP, operator mapping, on-chip/off-chip split, on-device latency, and the three latency
findings §2 argues from).

So the defensible statement is *"under the one protocol criterion we tested it does not
carry over on either of two models, and a criterion outside the protocol's set recovers
it on both"*, **not** *"the published claim holds on detection"*, and **not** *"pruning
does not transfer to detection"*. The criterion space
has not been characterised.

### How LAMP was selected, and how to say so

LAMP entered the grid on the basis of a **four-way control at a single operating point**
(EfficientDet-Lite1, backbone −10%, no fine-tuning): `global+magnitude_l2` 0.0003,
`global+random` 0.0674, `global+lamp` **0.1092**, `local+magnitude_l2` 0.0820. It won
there, so it was carried to the full ladder alongside the paper's own criterion.

State the basis, not a superlative:

> *"LAMP was selected for the full grid from a four-way control at backbone −10%; the
> remaining criteria were not swept (see the budget above)."*

Evidence: [`results/detection/full_scope_grid.json`](../results/detection/full_scope_grid.json)
(`lamp_selection`) and [`results/detection/pruning_matrix.json`](../results/detection/pruning_matrix.json)
(`allocation_control_lite1`). Three of the four rows are worker-produced; the `random` row
is not; see the provenance table below.

Do **not** write *"we only ran LAMP because it is the strongest criterion"*. That is
circular, since strongest is only known against what was tested, and the untested set
includes the three that carry the published result at high ratios.

### Do not read LAMP as "the better criterion"

LAMP's protection here is a **`1/n` width artifact**, not principled layer adaptation: it
happens to leave the narrow early layers alone on these backbones, which is exactly where
[§1](#1-allocation-global-ranking--mean-normalised-magnitude-guts-narrow-early-layers)
shows the damage is done. Two measured cautions:

* On Re-ID it repairs the stem (50/64 vs 1/64) and **still loses to local ranking**
  (cos 0.688 vs 0.766); it relocates the starvation into `layer4` instead.
* On detection its own margin peaks at 70% (+108%) and then falls at 80% (+78%, absolute
  0.1900 = 59.0% of baseline). LAMP breaks down too; its usable range here ends near 70%.

Write it as *"on this backbone LAMP happens to protect the information bottleneck"*, and
declare it as a deviation from the published protocol.

---

## 4. Quantization: the export path changed on 2026-08-22, and older numbers predate it

Every conversion in this repository now runs **litert-torch → `ai-edge-quantizer`**
(`static_wi8_ai8`: weights `CHANNELWISE`, activations `TENSORWISE`), which is the
configuration the DSD 2026 protocol specifies. Until 2026-08-22 it ran
**`torch.onnx.export` → onnx2tf**, whose built-in quantizer this repo additionally
overrode to **per-tensor** for two families. Numbers committed before that date came
from the old path. This section says what changed and which numbers still carry it.

**What the protocol does.** The paper's pipeline is *"exported to ONNX, converted to
TensorFlow via onnx2tf, quantized with **ai-edge-quantizer** … and finally compiled for
the Edge TPU"*. In the reference implementation the call is
`Quantizer.add_static_config(regex=".*", operation_name=ALL_SUPPORTED,
activation_num_bits=8, weight_num_bits=8)`, with **no `weight_granularity` argument**, so
it takes that parameter's default, which in `ai_edge_quantizer` 0.4.2 is
`QuantGranularity.CHANNELWISE`. The fallback path in the same file does not use onnx2tf's
built-in quantizer either; it goes through `TFLiteConverter.from_saved_model()` with a
representative dataset, under the comment *"onnx2tf's built-in quantization produces
broken models for some architectures."* Both paths give per-channel weights.

**What this repo did, and does.**

| | quantizer | weight granularity |
|---|---|---|
| DSD 2026 protocol | `ai-edge-quantizer` (fallback: `TFLiteConverter`) | per-channel |
| this repo, now, **every family** | `ai-edge-quantizer`, `static_wi8_ai8` | per-channel |
| before 2026-08-22: `clip_rn50`, `reid`, `line_seg` | onnx2tf built-in | per-channel |
| before 2026-08-22: `imagenet_backbones`, `ssdlite` | onnx2tf built-in | **per-tensor** |
| before 2026-08-22: `effdet` (published ladder) | `ai-edge-quantizer`, `static_wi8_ai8` | per-channel |

The old per-tensor default was not neutral: `onnx2tf.convert`'s own `quant_type` default
is `"per-channel"`, so the repo overrode it, and only four `family.yaml` files overrode it
back. There is no per-tensor knob any more; `ai-edge-quantizer`'s recipe is the only
path, so the granularity divergence cannot recur.

**What the granularity cost, and what that number is not.** Measured once, on
`efficientdet_lite1` at the `lamp` −10% rung, with the converter held fixed and
granularity the only variable: **per-tensor 0.29167 mAP against per-channel 0.31352**, so
+0.0218 absolute or +7.5% relative, for +16.5% on the intermediate `.tflite`. That is a
measurement on **detection**. It says the knob was not free; it does **not** say what
`imagenet_backbones` or `ssdlite` lost, and nobody measured that.

**Which committed numbers still come from the old path.** The switch changed the export
path, not the checkpoints, so every FP32 pruning-stage number is unaffected. That is
`results/detection/pruning_matrix.json` (post-prune / pre-finetune mAP),
`results/clip_rn50/global_allocation.json` (zero-shot 99.0 → 0.0 → 70.75),
`results/reid/sweep_results.json` (FP32 distillation mAP/rank1), and
`results/criterion_census.json`. What *was* measured on an onnx2tf-built artifact:

* `results/line_seg/param_ladder_oncar.json`: `int8_val_iou`, 30 rungs
* `results/clip_rn50/zero_shot.json`: the int8 rows
* `results/tflite_size_audit.json`: byte counts for every family
* `results/detection/coral_latency.json`: the partially-mapped detection builds

Each of those carries a `measured_on` marker. Where a family has since been re-converted,
the current JSON says so; where it has not, the accuracy figure is what the old path
produced and has not been re-verified.

**Two things this does not license.** It does not retroactively impugn a published
accuracy number: the two families that sat on the coarser setting published none,
`imagenet_backbones` had never been run at all before 2026-08-22, and `ssdlite`'s three rungs
were fine-tuned on `coco-val2017`, the split `scripts/pruning.sh` itself labels
*"smoke-test, leaky"*, so no mAP for it is committed either. And closing the granularity
gap was never the same as closing the **quantizer** gap: onnx2tf's per-channel and
`ai-edge-quantizer`'s CHANNELWISE are different calibrators with different activation
handling. Only the second change, moving to `ai-edge-quantizer` itself, closed both.

Evidence: paper `material/paper/experiments.tex:11`; reference implementation
`03_convert_to_tflite_int8.py` `_quantize_ai_edge` / `_quantize_via_onnx2tf`;
`ai_edge_quantizer/quantizer.py:311` (the `CHANNELWISE` default, in the 0.4.2 environment
the reference implementation shipped with);
[`src/int8_pruning/convert/tflite.py`](../src/int8_pruning/convert/tflite.py) (`quantize_int8`). The
reference implementation and the paper source are not in this repository; see the
provenance table below.

---

## Ratio semantics differ between families: read the filename carefully

`pruned<P>pct` does **not** mean the same thing everywhere:

| family | default meaning of `P` | switch |
|---|---|---|
| `imagenet_backbones`, `clip_rn50`, `reid` | full-model **parameter** reduction | — |
| `effdet`, `ssdlite` | **backbone** parameter reduction | `--scope full` → full-model, writes a `_full` suffix |
| `line_seg` | full-model **parameter** reduction, `_param` suffix | `--ratio_semantics channel` reproduces the [retired round-1 ladder](../results/line_seg/prune_sweep_10_90pct.json) |

`line_seg` was the sharpest case and is worth keeping on record: until 2026-08-22 it
numbered rungs by the upstream handoff's per-layer **channel** ratio, so `pruned90pct` was a
99.0% parameter cut against `pruned90pct_param`'s 90.0%, an order of magnitude apart under
names one character different. That ladder has been withdrawn (see
[`results/_deleted/`](../results/_deleted/)) and the family now reports parameters like
everyone else, but `effdet` and `ssdlite` still report a backbone ratio, so the rule stands:
every worker records the realised value (`param_reduction_pct`, `realized_param_pct`,
`param_reduction_pct_full`) in its JSON log;
**compare on those columns, never on the filename.**

---

## Provenance: what backs each number, and which claims are weaker

Every measured claim above names a committed file. These are the exceptions and the
corrections, kept visible.

### Not reproducible from this repository

| claim | where it lives | status |
|---|---|---|
| The reachable-ceiling table in [`results/detection/target_reachability.json`](../results/detection/target_reachability.json) | that file | Its own `produced_by` says it: the numbers were read off the worker logs under gitignored `outputs/pruning_logs/` **by hand**, and the reading script was never committed. Superseded for the ceiling question by [`results/detection/pruning_matrix.json`](../results/detection/pruning_matrix.json), whose `reachable_ceilings_full_model` measures the same thing from a committed script; the file is kept because it is what the 2026-08-21 statements rested on. |
| `global + random` mAP **0.0674**: §1's four-way control and §3's LAMP-selection basis | [`results/detection/pruning_matrix.json`](../results/detection/pruning_matrix.json) (`provenance: "audit adversary"`), [`results/detection/full_scope_grid.json`](../results/detection/full_scope_grid.json) | The producing script was never committed. [`results/protocol_audit/k2_cap_sweep.log`](../results/protocol_audit/k2_cap_sweep.log) prints the number, but that line is a **hard-coded literal re-quoting it**; that script only ever constructs `magnitude_l2`, and its loop variable is the cap. Promoting the audit directory did not make it checkable. On record, unverifiable. |
| Re-ID: stem 64 → 1, cos 0.77 → 0.39, LAMP 0.688 vs local 0.766, stem 50/64 | `families/re-identification/reid/prune.py:95-96`, `families/re-identification/reid/README.md:96-99`, `src/int8_pruning/prune/core.py:13` | Committed, but as **code comments**, not as a results file. Never re-derived by a script. |
| BiFPN lowering: 2,816 single-channel `CONV_2D`, 88 siblings, 2,454-char names | [`src/int8_pruning/report/tflite_size.py`](../src/int8_pruning/report/tflite_size.py) module docstring | Committed as prose in the module that measured it. `results/tflite_size_audit.json` carries the byte columns but not these three counts. |
| Fine-tune cost 5.16 h and 14.90 h per rung | [`results/detection/full_scope_grid.json`](../results/detection/full_scope_grid.json) (`cost_per_criterion`) | The JSON is committed and the figures are the mean of `duration_s.total` over all 36 per-rung logs, but those logs live under the gitignored `outputs/`, so a clone cannot recompute them. [`results/protocol_audit/ft_throughput.py`](../results/protocol_audit/ft_throughput.py) writes nothing and no log was kept. Weaker than a committed measurement, stronger than the 2026-08-22 edition, which projected the Lite2 half. |
| Re-ID: "per-tensor collapses discriminability" | `families/re-identification/reid/README.md:17`, `README.zh-CN.md:17` | **Never measured.** Written by analogy with `clip_rn50`, where the collapse *is* measured (94.17% → 18.83%). No per-tensor Re-ID rung is committed, and `families/re-identification/reid/eval_discrim.py:36` describes the comparison as an experiment a reader could run. Both READMEs now say so. |
| The full-mapping lite1 ladder: 305/305 operator mapping, the int8 mAP column, on-device latency | [`results/detection/full_mapping_ladder_lite1.json`](../results/detection/full_mapping_ladder_lite1.json) | The producing scripts **are** committed, at [`results/protocol_audit/export_litert/`](../results/protocol_audit/export_litert/), pinned to litert-torch 0.9.3 / torch 2.10 / ai-edge-quantizer 0.8.0 / edgetpu_compiler 16.0. They invoke `python -m etpu.convert.flatbuffer`, a module path the 2026-08-27 package rename retired (`int8_pruning.convert.flatbuffer` now); they are kept verbatim because they record what was run, and they were already unrunnable for the reason that follows. What a clone still cannot do is *run* them: every input is under gitignored `outputs/`, and the venv they need is not the repo's. Two of the three steps also monkeypatch effdet and timm internals, and a patch that stops applying fails **silently**: back to 83 of 479 operators, with no error. Check the operator count before trusting a re-run. The `fp32_mAP` column is unchanged from the earlier cluster runs; only the int8 column, the operator mapping and the latency are new. |
| The 26.96 ms head-on-CPU split | [`results/detection/full_mapping_ladder_lite1.json`](../results/detection/full_mapping_ladder_lite1.json) (`open[2]`) | Prose only. **No surviving measurement at all**: it is in no log and no bench output. Quote it as an unverified recollection or not at all. The 52.78 ms it is compared against is the `lamp` −10% rung from the pre-calibration-fix bench pass, not the unpruned rung (52.404 ms). |
| The I/O floor: return bandwidth, and the ~1.5 ms it does not account for | [`results/detection/io_floor_control.json`](../results/detection/io_floor_control.json) | **Re-measured 2026-08-22 and now committed**: three models, two independent passes, with the alternatives (weight streaming, operator partitioning, arithmetic) ruled out one at a time. Two caveats remain in the file's `not_established`: the host→TPU direction was never varied and is charged at the return rate, and the residual between 33.4 ms of I/O and the 34.95 ms fitted floor is unattributed. |

### Corrected during the 2026-08-21 audit

| was | is now | why |
|---|---|---|
| §1 table: `blocks.1.0.conv_dw` 96 → 17, kept **8.3%** | `blocks.1.2` 144 → 12, kept 8.3% | 17/96 is 17.7%, not 8.3%. The row had merged the damage-saturation point (96 → 17, `blocks.1.0.conv_pw`) with the allocation-control point (8.3%, `blocks.1.2`). Both survive, each in its own paragraph. |
| §1 raw L2 `blocks.1.0` 0.5130 vs `blocks.5.1` 0.2689 | 1.0126 vs 0.6121 | The published pair reproduces under none of five reductions tried. What the argument needs, raw scale running early above late, holds under all five. |
| §3 census: `magnitude_l2` **89** rungs, `lamp` **13** | 94 and 13 at the audit; **101 and 20** after the Lite2 ladder finished 2026-08-26 | Hand-counted over one directory; the script counts five. The `.pt` figures (44, 4) were right but were being read from two different directories under a single column. |
| §2 "the published **63-row** ResNet-50 table" | 64 rows | 63 pruned configurations plus the unpruned baseline. The fit reported as F(1,61) uses all 64. |
| §2 "−40.0% on the compiled one" | −39.8% | The compiled artifacts measure −39.8%. −40.0% was a rounding that had drifted into the text; `attic/README.md` already recorded −39.8%. |
| `coral_latency.json`: "396 of 479", "only 1.34x", "0.00 B at EVERY rung" | 389 of 479, 1.35×, ssdlite carries ~49 KiB at 30% and 50% | Each contradicted that file's own rows. 396/83 is a real number, but for a *different compile* of the same input than the one benchmarked. |
| the litert export's `--pool-ceil` rewrite described as bit-exact for every EfficientDet | bit-exact for **Lite1 only**; the 5 **Lite2** rungs built through it were a different network (rebuilt since; see the third column) | `ceil_mode=True` equals timm's SAME-pool only on even pool inputs. Lite1 @384 pools 48→24→12→6→3, all even. Lite2 @448 pools 56→28→14→**7**→4, and at 7 the two disagree (4×4 vs 3×3). Confirmed on the artifact: the Lite2 export emits levels 56/28/14/7/**3**. **Corrected 2026-08-27**: the rewrite is now chosen per site from the pooled extent's parity (`_same_pool_args`), not once per model. TF SAME pads by 1 on an even extent (asymmetric, `ceil_mode`) and by 2 on an odd one (symmetric, `padding=1`). All 19 Lite2 rungs are rebuilt at P7 4×4, 340/340 operators on the TPU; Lite1 rebuilds byte-identical. Simply dropping the flag was *not* an option: it leaves a `PADV2` the compiler declines, and a correct-but-unrewritten Lite2 maps only 83 of 375 operators. The retired `export_litert/` artifacts were not rebuilt. `run_ladder.sh` now gates the flag to Lite1. |
