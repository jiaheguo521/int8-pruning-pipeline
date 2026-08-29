# reid: person Re-ID appearance embedders on the Edge TPU

Two tracks: three Torchreid Market-1501 embedders shipped **without** pruning,
and the Youtu `reid_youtu_lite` pruning-ratio sweep. Both use **per-channel**
int8 for the same reason CLIP does: the output is a cosine embedding.

Input is 256x128 (H x W), the only non-square family here.

> Prerequisites: the venv and Phase 1–3 drivers in the [root README](../../../README.md).

## 6. Person Re-ID: Torchreid appearance embedders (no pruning)

```bash
# Three Torchreid person-ReID embedders with Market-1501 weights as int8 Edge-TPU
# models: osnet_x0_5, osnet_x0_75, mobilenetv2_x1_0. Straight from the pretrained
# checkpoint through Phase 2/3 — NO training, NO pruning. Like CLIP they emit a
# cosine-matched EMBEDDING (osnet: 512-d), so they use PER-CHANNEL int8 by analogy with
# clip_rn50, where per-tensor IS measured to collapse the metric (zero-shot top-1
# 94.17% -> 18.83%). NOTE: the equivalent has never been measured for Re-ID and no
# per-tensor rung is committed -- eval_discrim.py:36 describes it as an experiment a
# reader could run. Do not repeat 'per-tensor collapses Re-ID' as if it were measured. Input is 256x128 (H x W), ImageNet norm, RGB.

# Step 0 — build the baselines (Torchreid installed --no-deps to protect the
# torch/TF stack; Market-1501 weights from the Model Zoo via gdown). Each forward()
# is verified to return the EMBEDDING, not the 751-way ID head (asserts dim != 751):
./scripts/download_and_finetune_models.sh
# -> outputs/models/reid_{osnet_x0_5,osnet_x0_75,mobilenetv2_x1_0}_market1501.pt
MARKET1501_ENABLED=1 ./scripts/download_datasets.sh   # calibration + eval crops (~140 MB)

# Step 1 — int8 convert (per-channel, auto-detected) + compile. The baselines live
# in outputs/models/, so point Phase 2 there via PT_PRUNED_DIR:
PT_PRUNED_DIR=$PWD/outputs/models GLOB='reid_*_market1501.pt' \
  DATASET_DIR=$PWD/data/datasets ./scripts/convert.sh
GLOB='reid_*_market1501_int8.tflite' SHOW_OPS=1 ./scripts/compile_edgetpu.sh
# -> outputs/tflite_int8_litert/reid_*_int8.tflite  ->  *_int8_edgetpu.tflite

# Step 2 — discriminability check (same-person vs different-person cosine, before
# AND after quant, plus the fp32<->int8 direction gate). Confirms int8 kept the
# metric — the PINTO_model_zoo "same/different indistinguishable" trap:
python families/re-identification/reid/eval_discrim.py \
    --pt outputs/models/reid_osnet_x0_5_market1501.pt \
    --tflite outputs/tflite_int8_litert/reid_osnet_x0_5_market1501_int8.tflite \
    --market-dir data/datasets/market1501 --num-ids 60

# Step 3 — SRAM report (on-chip/off-chip; off-chip>0 => params stream over USB, the
# ~51ms trap) + CPU int8 latency. No Coral needed for the SRAM prediction:
python families/re-identification/reid/report.py --glob 'reid_*_market1501_int8.tflite' \
    --out-json outputs/pruning_logs/reid_edgetpu_report.json

# Rank-1 / mAP on Market-1501 (Torchreid Model Zoo, for reference):
#   osnet_x0_75  93.7% / 81.2%     osnet_x0_5  92.5% / 79.8%     mobilenetv2_x1_0  85.6% / 67.3%
#   (OSNet is Re-ID-specific: far fewer params than MobileNetV2, higher accuracy.)

# Step 4 (optional) — REAL-DEVICE single-embed latency on the Coral USB, via
# tflite_runtime + libedgetpu delegate inside an Edge-TPU docker (no pycoral needed):
docker run --rm --privileged -v /dev/bus/usb:/dev/bus/usb \
    -v $PWD/src/int8_pruning/backends/edgetpu:/src:ro -v $PWD/outputs/edgetpu:/models:ro \
    edgetpu_ros:latest \
    bash -c 'python3 /src/bench.py /models/single_*/reid_*_edgetpu.tflite'
# Measured on Coral USB — the ranking INVERTS vs params/accuracy:
#   mobilenetv2_x1_0  2.9 ms   (70 ops; FASTEST despite most params — all dense convs)
#   osnet_x0_5        8.1 ms   (290 ops)      osnet_x0_75  11.2 ms  (290 ops; SLOWEST)
# All off-chip=0 (fit the 8 MiB SRAM) => none hits the USB-streaming penalty. Choose
# by MEASURED latency, not params: OSNet is more accurate but 3-4x slower per embed.
```

## 7. Youtu Re-ID (`reid_youtu_lite`): the pruning ratio sweep

```bash
# The one reid_* family that IS pruned. Tencent YouReID youtu_reid_baseline_lite
# (== opencv_zoo person_reid_youtu_2021nov): ResNet50 + cat(gap,gmp) -> 1x1 conv ->
# 768-d, 26.66M params, int8 ~26.7 MB. Trained on FOUR datasets (market1501 + Duke +
# MSMT17 + CUHK03) — that multi-source generalization is the reason to keep it over a
# single-source OSNet, and the reason it is worth shrinking rather than replacing.
#
# Why a sweep: on a Coral the emb model shares 8 MiB of SRAM with the detector. With
# an SSD taking 5.41 MB of cache, emb gets ~2.4 MiB — so the 26.1 MiB baseline streams
# ~23.7 MiB over USB. Measured on the device, a streamed MiB costs 2.30 ms on this
# family (lat = 2.302*offchip + 0.223*onchip + 2.782, R^2 = 0.9976 over six rungs;
# results/reid/coral_latency.json) => ~58 ms per embed co-compiled, and 46.7 ms measured
# standalone. Pruning buys latency directly: the 80% rung is 5.6 ms standalone.

# Step 0 — baseline (YouReID Model-Zoo checkpoint via gdown, ~226 MB).
./scripts/download_and_finetune_models.sh
# -> outputs/models/reid_youtu_lite.pt   (UNSUFFIXED: the weights are four-source)
MARKET1501_ENABLED=1 ./scripts/download_datasets.sh

# Step 1 — prune + distillation recovery. Duke/MSMT17/CUHK03 are unobtainable (Duke
# withdrawn, MSMT17 licensed), so the original four-source label space is out of
# reach and CrossEntropy/triplet recovery is impossible. Distillation sidesteps it:
# the TEACHER is the four-source model, so its knowledge transfers with no labels.
DATASET_DIR=$PWD/data/datasets PARTS=A CLS_MODEL=reid_youtu_lite \
  CLS_DATASET=market1501 IMPORTANCES=magnitude_l2 CHECKPOINTS="20 50 70 80 90" \
  ./scripts/pruning.sh
# -> outputs/pytorch_pruned/reid_youtu_lite_market1501_pruned{20,50,70,80,90}pct_magnitude_l2.pt
#    (+ a _pruned0pct.pt symlink to the baseline)

# NOTE: REID_GLOBAL_PRUNING defaults to 0 here — unlike EVERY other family. A global
# magnitude compare ranks RAW filter norms across layers, and resnet50's deep filters
# out-norm the stem's, so it deletes the early layers wholesale. Measured at 90%,
# cos(student,teacher) after an identical 1 epoch of distillation:
#   global + magnitude_l2  0.390   stem  1/64 channels survive   head 64.7% of params
#   global + lamp          0.688   stem 50/64                    head  9.2%
#   local  + magnitude_l2  0.766   stem 17/64                    head 32.8%
#   local  + lamp          0.766   stem 17/64                    head 32.8%
# The two local rows matching to 4 decimals is the lesson: once a uniform per-layer
# ratio fixes the ALLOCATION, the importance criterion (which only ranks channels
# WITHIN a layer) barely matters. Global magnitude even loses to global RANDOM.

# Step 2 — int8. Per-channel, real pedestrian crops, ImageNet norm, no BGR<->RGB in
# the graph (a negative-stride reverse makes edgetpu_compiler fail on dwl.reverse).
DATASET_DIR=$PWD/data/datasets GLOB='reid_youtu_lite*_market1501_pruned*.pt' \
  ./scripts/convert.sh
# -> outputs/tflite_int8_litert/..._int8.tflite   <- the CPU int8 IS the deliverable.
# Do NOT ship *_edgetpu.tflite: it must be CO-COMPILED with the detector in ONE
# edgetpu_compiler invocation, or the two models evict each other's cache and the
# latency numbers are void.

# Step 3 — real Market-1501 mAP / rank-1 (torchreid's evaluate_rank; fp32 vs int8).
# eval_reid_discriminability.py is only a quantization gate — it cannot rank a sweep.
python families/re-identification/reid/eval_map.py \
    --pt outputs/models/reid_youtu_lite.pt \
    --tflite outputs/tflite_int8_litert/reid_youtu_lite_market1501_pruned90pct_magnitude_l2_int8.tflite \
    --market-dir data/datasets/market1501
# Baseline reproduces the published numbers: mAP 87.89 / rank-1 95.22 (published
# 87.86 / 95.01), which validates the vendored arch + split_bn collapse + protocol.

# Ablations (each is its own stem — same baseline, different model):
#   768-d + simmat loss  -> reid_youtu_lite_simmat_market1501   (loss control)
#   256-d + simmat loss  -> reid_youtu_lite_e256_market1501     (head vs backbone)
# The head is pinned to keep the 768-d output drop-in, but its in_channels track
# layer4's surviving width, so it grows from 11.8% of params at 0% to 32.8% at 90%.
# The e256 runs prune the head first and hand that budget to the backbone.
REID_EMBED_DIM=256 REID_DISTILL_LOSS=simmat DATASET_DIR=$PWD/data/datasets PARTS=A \
  CLS_MODEL=reid_youtu_lite CLS_DATASET=market1501 IMPORTANCES=magnitude_l2 \
  CHECKPOINTS="80 90 95" ./scripts/pruning.sh
# 95% == 1.3M params == osnet_x0_75's capacity: the point that separates "multi-source
# DATA" from "CAPACITY" as the reason Youtu beats OSNet on an unseen viewpoint.

# Distillation only pins the teacher WHERE YOU SAMPLE. Market-only images hold the
# student to the teacher on Market-like crops; at high ratios capacity is scarce
# enough that it will sacrifice unsampled domains. Mixing in unlabeled crops from the
# deployment camera costs nothing (the teacher supplies the targets) and is the
# highest-leverage knob for a viewpoint no Re-ID dataset contains:
#   REID_EXTRA_IMAGE_DIRS="/path/to/car_crops" ... ./scripts/pruning.sh
```
