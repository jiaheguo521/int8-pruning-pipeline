> English | [中文版](SETUP.zh-CN.md) · [← back to the README](../README.md)

# Setup

Building the environment, fetching data and baseline weights, and running the pipeline
end to end: structured pruning with recovery, then int8 quantization. Compiling for the
Edge TPU and timing on a Coral are an optional backend after that, in section 5. For what
this repository is and what it measured, start at the [README](../README.md).

## 1. Create the venvs

```bash
# Enter the project root (replace <project_root> with the actual path on your server)
cd <project_root>

# One script, both export paths. The default run creates ./pruning-env: torch from
# the CUDA-matched index first, then `pip install -e '.[all]'` -- every model family
# plus the litert int8 export path. That is the whole prune -> recover -> int8
# pipeline, and it is all most people need.
./scripts/setup_env.sh

# Add the second export path. It gets its OWN venv, ./onnx2tf-env, and that is a
# constraint rather than a preference: onnx2tf pins numpy==1.26.4 and tensorflow
# wants numpy<2.2, while the litert path runs on numpy>=2. Section 4 says what it
# is for -- reproducing artefacts built before 2026-08-22, nothing else. The run
# finishes by pinning numpy==2.1.3, tensorflow's ceiling: that violates onnx2tf's own
# numpy==1.26.4 with a pip warning and works, and the one numpy-2 workaround it then
# needs lives in int8_pruning.convert.export_onnx2tf.
ONNX2TF_ENABLED=1 ./scripts/setup_env.sh

source ./pruning-env/bin/activate

# Sanity check (the script prints the same versions itself when it finishes)
python -c "import torch, torchvision, torch_pruning, effdet, pycocotools; print(torch.__version__, torch.cuda.is_available())"
```

The script is env-var driven and idempotent: an existing venv is reused and pip
re-resolves in place, so re-running it after a `pyproject.toml` change is the
supported way to update one. It never activates anything and never needs to be run
from inside a venv: every install goes through `<venv>/bin/pip` by absolute path.

```bash
# Every toggle it takes:
#   PRUNING_ENV_ENABLED=1|0   build ./pruning-env                       (default 1)
#   ONNX2TF_ENABLED=1|0       build ./onnx2tf-env                       (default 0)
#   PY=python3.12             interpreter used to CREATE the venvs (default python3;
#                             needs >=3.10, and 3.12 is what every result was run on)
#   EXTRAS=all                extras for pruning-env, named after the directories
#                             under families/ (see "Installing only what you need")
#   TORCH_VERSION=2.10.0      >=2.10 is a hard floor for the litert path; section 4
#                             says what breaks below it
#   TORCH_INDEX=...           wheel index for torch/torchvision (default cu128)
#   VENV_DIR=...              where ./pruning-env goes     (gitignored by name)
#   ONNX2TF_VENV_DIR=...      where ./onnx2tf-env goes     (gitignored by name)

# The three that matter on a differently-shaped machine:
PY=python3.12 ./scripts/setup_env.sh                                     # pick the interpreter (>=3.10)
TORCH_INDEX=https://download.pytorch.org/whl/cpu ./scripts/setup_env.sh  # no CUDA on this box
PRUNING_ENV_ENABLED=0 ONNX2TF_ENABLED=1 ./scripts/setup_env.sh           # only the onnx2tf venv
```

**Installing only what you need.** `[all]` is the default because it always works.
If you are bringing up one model family, the extras are named after the directories
under `families/`, and each pulls only that family's third-party code:

```bash
EXTRAS='effdet,convert' ./scripts/setup_env.sh   # EfficientDet + the int8 export path
EXTRAS='reid,convert'   ./scripts/setup_env.sh   # person Re-ID + the int8 export path
EXTRAS='clip-rn50'      ./scripts/setup_env.sh   # open-vocabulary zero-shot, prune only
```

`imagenet_backbones` and `line_seg` have no extra on purpose; they need nothing beyond
the core and torchvision. `reid` additionally needs torchreid, which is installed
without its dependencies so it cannot downgrade torch; see section 2.

## 2. Download datasets and baseline weights

```bash
# Enter the project root and activate the virtualenv
cd <project_root>
source ./pruning-env/bin/activate

# Paths default to scripts/config.sh values (PROJECT_DIR, DATASET_DIR,
# MODELS_DIR, PT_PRUNED_DIR, TFLITE_DIR, EDGETPU_DIR, LOG_DIR) —
# export only to override, e.g. to point DATASET_DIR at another drive:
# export DATASET_DIR=/datasets
# export DATASET_DIR="$(pwd)/data/datasets"
# export MODELS_DIR="$(pwd)/outputs/models"


# Step 1 — pull datasets (opt-in; downloads NOTHING by default, just prints the toggles).
#   Enable each dataset explicitly with its *_ENABLED flag (all default 0):
#     COCO_ENABLED=1            COCO 2017 train/val images + annotations (~25 GB)
#     COCO_MINITRAIN_ENABLED=1  + coco-minitrain 25k JSON (needs COCO images; ~5x faster recovery FT)
#     FLOWERS102_ENABLED=1      Oxford Flowers-102 via torchvision (~330 MB; fast bring-up classifier)
#     INAT_ENABLED=1            iNaturalist 2017 Plantae (~198 GB stream, NOT resumable, ~30-40 GB on disk)
#     IMAGENET_ENABLED=1        ImageNet-1k: NO download (registered access) — only verifies the
#                               $DATASET_DIR/Imagenet_1k ImageFolder layout + writes its README
#                             NOTE: nothing guarantees what is in that directory. On
#                             the development machine it holds the 10-class Imagenette
#                             subset, which an ImageFolder loads without complaint --
#                             src/int8_pruning/data/classification.py now refuses a non-1000-class
#                             tree rather than silently training a 1000-way head on 10.
#     MARKET1501_ENABLED=1      Market-1501 person Re-ID benchmark (~140 MB Google-Drive zip)
#   Two knobs that are not toggles: FORCE_README=1 overwrites an existing dataset
#   README, and INAT_MIRROR_URL=<url> fetches a Plantae-only mirror instead of
#   streaming the 198 GB archive. The stream cannot resume; a mirror fetch can.
#   General-edge paper default: ImageNet-1k (classification) + COCO (detection):
COCO_ENABLED=1 COCO_MINITRAIN_ENABLED=1 IMAGENET_ENABLED=1 ./scripts/download_datasets.sh
# Step 2 — pull baseline weights + finetune. Downloads every prune baseline:
#   classification (ImageNet head=1000): mobilenetv2, efficientnet_lite0 (timm),
#                                        mnasnet1_0, squeezenet1_1, resnet50
#   detection (COCO): efficientdet_lite1 (effdet), ssdlite_mobilenetv3 (torchvision)
#   By default finetunes MobileNetV2 -> iNat 2017 Plantae (2101 classes, ~4-5h on A6000)
#     -> outputs/models/mobilenetv2_inat.pt
#   Toggle to the fast bring-up path: CLS_DATASET=flowers102 ./scripts/download_and_finetune_models.sh
#   Skips if already present; force retrain: FINETUNE_FORCE=1; download weights only: FINETUNE_ENABLED=0
#   FINETUNE_EPOCHS=N sets the finetune budget (default 20).
#   Naming rule: a raw download keeps the bare family name (mobilenetv2.pt,
#   efficientdet_lite1.pt); a finetune adds the dataset suffix (mobilenetv2_inat.pt,
#   mobilenetv2_flowers102.pt). The ImageNet path is a pure download, so it stays
#   mobilenetv2.pt.
./scripts/download_and_finetune_models.sh

# (Optional) Manually rerun the default iNat 2017 Plantae finetune (the download script already ran it; only use when changing epochs / forcing a retrain):
# python families/classification/imagenet_backbones/finetune.py \
#     --dataset inat_plant \
#     --imagenet_weights $MODELS_DIR/mobilenetv2.pt \
#     --inat_train_json  $DATASET_DIR/inat2017_plant/train2017.json \
#     --inat_val_json    $DATASET_DIR/inat2017_plant/val2017.json \
#     --inat_image_root  $DATASET_DIR/inat2017_plant \
#     --num_classes 2101 --epochs 20 --amp --force

# (Optional) Fast bring-up path on Oxford Flowers-102 instead (~30 min on A6000):
# python families/classification/imagenet_backbones/finetune.py \
#     --imagenet_weights $MODELS_DIR/mobilenetv2.pt \
#     --data_root $DATASET_DIR/flowers102 \
#     --epochs 20 --amp --force
```
## 3. Phase 1: structured pruning

The default recipe is 100 recovery epochs over 35 runs (5 criteria x 7 ratios) per
part. What that costs on an A6000:

| run | wall clock |
|---|---|
| Part A, Flowers-102 | ~3-5 h |
| Part A, ImageNet-1k (the default) | weeks to months: 1.28M images x 100 ep x 35 runs |
| Part A, iNat 2017 Plantae | ~6-8 days |
| Part B, detection | ~25 days, detection being 5-10x slower per epoch |

For a first pass: `FINAL_EPOCHS=5 PARTS=A ./scripts/pruning.sh`, then read the curve
and decide whether to extend. On the ImageNet default, always set FINAL_EPOCHS.

```bash
# Real sweep: run Part A (classification) and Part B (detection, ~5-10x slower) independently in two steps
# A full 100 ep run totals ~30+ days; recommended to first run 30 ep to see the curve before deciding whether to extend
#
# Tunable env vars (exposed in ./scripts/pruning.sh):
#   PARTS         A | B | AB                       (default AB; here we split into A / B)
#   CLS_MODEL     mobilenetv2 | efficientnet_lite0 | mnasnet1_0 | squeezenet1_1 | resnet50
#                                                  (Part A family, default mobilenetv2;
#                                                   non-mobilenetv2 = imagenet only)
#   DET_MODEL     efficientdet_lite1 | ssdlite_mobilenetv3
#                                                  (Part B family, default efficientdet_lite1)
#   DET_SCOPE     backbone | full                  (effdet only, default backbone. What
#                                                   CHECKPOINTS is measured against; 'full'
#                                                   is what the published ladder used and
#                                                   writes a separate '_full' stem.
#                                                   The high rungs do not land on their
#                                                   target: the per-layer floor saturates,
#                                                   so CHECKPOINTS=90 realized 88.5554%
#                                                   under lamp and 88.7362% under
#                                                   magnitude_l2, two different models.
#                                                   Read realized_full_pct out of the log,
#                                                   never the target, before comparing
#                                                   arms. See
#                                                   results/detection/target_reachability.json)
#   CHECKPOINTS   "10 20 30 40 50 60 70"           (pruning ratios %, space-separated)
#   IMPORTANCES   "magnitude_l1 magnitude_l2 fpgm random lamp"  (5 data-free importances)
#   FINAL_EPOCHS  N                                (override recovery FT epochs, default 100)
#   PROJECT_DIR   path/to/project                  (defaults to the script's parent; see scripts/config.sh)
#   DATASET_DIR   path/to/datasets                 (default /datasets;        see scripts/config.sh)
#   MODELS_DIR    path/to/baseline_weights         (default outputs/models;   see scripts/config.sh)
#   CLS_DATASET   imagenet | flowers102 | inat_plant | market1501 | line_seg
#                                                  (Part A dataset, default imagenet;
#                                                   clip_rn50, relu_clip, reid_* and
#                                                   line_seg_* pin their own)
#   COCO_SUBSET   train2017 | minitrain | val2017  (Part B split, default train2017.
#                                                   minitrain is ~5x faster and needs the
#                                                   instances_minitrain2017.json file;
#                                                   val2017 is a smoke test and leaks,
#                                                   because the eval loader reads it too)
#   PRUNE_MODE    independent | iterative          (default independent; see below)
#
# Part-B recovery FT, EfficientDet paper defaults (unset = the worker's own defaults):
#   DET_LR         1e-3                            (default 1e-3)
#   DET_WD         4e-5                            (default 4e-5)
#   DET_SCHED      cosine | multistep              (default cosine)
#   DET_WARMUP     1                               (epochs of linear warmup; 0 disables)
#   DET_MILESTONES "60 80"                         (multistep only; milestones past
#                                                   --final_epochs are dropped)
#
# Idempotent, and quietly so: an existing .pt is skipped by the Python worker, which
# prints "Already present, skip" and exits 0, so a run that changed nothing looks like
# one that rebuilt everything. Delete the .pt files you mean to redo. Output names:
#   Part A   <baseline_name>_pruned<P>pct_<imp>.pt
#   Part B   <baseline_name>_coco-<subset>_pruned<P>pct_<imp>.pt
# The subset token is in the Part-B name so two COCO splits cannot collide.
#
# Four families take knobs of their own, read only when that family is selected:
#   CLIP_TEXT_EMB / CLIP_VAL_DIR            (clip_rn50) zero-shot eval during distillation.
#                                           Unset = the worker tracks cosine-to-teacher only.
#   RELU_CLIP_TEXT_EMB / RELU_CLIP_VAL_DIR  (relu_clip) the same pair.
#   RELU_CLIP_GLOBAL_PRUNING=1              rank across the backbone instead of per-layer. On
#                                           clip_rn50 that left the stem at 3 of 32 channels,
#                                           recorded in results/clip_rn50/global_allocation.json.
#   RELU_CLIP_ROUND_TO=N                    round surviving channel counts up to a multiple of N
#                                           (0 = off). The Edge TPU's array is 64 wide, so below
#                                           64 a narrower layer buys no time. Untested here.
#   REID_EMBED_DIM=768                      pins the released head, a drop-in for a 768-d gallery.
#                                           Smaller prunes the head first and hands the budget to
#                                           the backbone; requires REID_DISTILL_LOSS=simmat.
#   REID_DISTILL_LOSS=cosine|simmat         cosine pins the embedding and needs matching dims;
#                                           simmat matches the batch cosine-similarity structure
#                                           and does not.
#   REID_GLOBAL_PRUNING=1                   compare across layers, which deletes resnet50's stem
#                                           (1/64 channels at 90%) and halves recovered fidelity.
#                                           The measured table is in that family's prune.py.
#   REID_EXTRA_IMAGE_DIRS="d1 d2"           extra UNLABELED crop dirs mixed into the distillation
#                                           set; the teacher supplies the targets.
#   LANE_SEG_UPSTREAM=...                   the upstream MicroROS-Pi5_Coral_TPU training/ dir. It
#                                           supplies lane_seg_data (data + the held-out eval
#                                           protocol) and the self-collected dataset; no public
#                                           substitute exists.
#   LANE_SEG_EPOCHS / LANE_SEG_SAMPLES      recovery FT budget. The defaults are the upstream mix
#                                           stage, 30 ep x 3000 samples, which is what the
#                                           baselines were trained with.
#   LANE_SEG_NO_FINETUNE=1                  prune and stop. Bytes and latency depend only on the
#                                           STRUCTURE, so this maps the on-chip/off-chip crossover
#                                           cheaply. Those models are measurement artifacts, NOT
#                                           deliverables, so point LANE_SEG_OUT_DIR outside
#                                           PT_PRUNED_DIR.
# CHECKPOINTS means something different for line_seg_*: per-layer CHANNEL ratios, not parameter
# ratios. The handoff's measured byte table is keyed to them, and params fall as ~ratio^2.

# Part A - MobileNetV2 on ImageNet-1k. This is the default (CLS_DATASET=imagenet)
# and needs NO finetune: the pretrained head=1000 download IS the prune baseline.
# For a fast bring-up curve use CLS_DATASET=flowers102 (102 classes) — that one
# DOES need its baseline finetuned first, via scripts/download_and_finetune_models.sh.
# PARTS=A FINAL_EPOCHS=30 CHECKPOINTS="50 60 70 80" IMPORTANCES="magnitude_l1 magnitude_l2 fpgm random lamp" ./scripts/pruning.sh
PARTS=A FINAL_EPOCHS=30 CHECKPOINTS="50 60 70 80" IMPORTANCES="magnitude_l2" ./scripts/pruning.sh

# Default: every rung is pruned from the dense baseline independently (the
# PruningBench protocol). PRUNE_MODE=iterative instead walks ONE trajectory --
# prune to the smallest ratio, recover, save, then keep pruning the recovered
# model to the next ratio -- LTH-style iterative magnitude pruning (Frankle &
# Carbin, ICLR 2019), the form the literature reports recovers high ratios better.
# NOT MEASURED HERE: every ladder under results/ was run in the default independent
# mode, so whether the trajectory beats it on these families is an open question,
# not a result of this repo. What IS tested is the mechanism -- fixed dense
# reference, resume, one pruner, `_iter` naming -- so a trajectory can never
# overwrite an independent ladder. It is resumable.
# PRUNE_MODE=iterative PARTS=A CHECKPOINTS="50 60 70 80" ./scripts/pruning.sh
#
# Which families implement it is declared in families/*/family.yaml (prune.prune_modes)
# and printed by:
#   python -m int8_pruning.manifest capabilities
# Asking for a mode a family has not implemented stops the run before any worker
# starts and tells you what to implement -- it is never silently dropped.
# On the two CLIP families -- where recovery is distillation, not labels -- the teacher
# is PINNED to the dense original for the whole trajectory: LTH warm-starts the student,
# never the supervision target, and clip's zero-shot is scored against text vectors
# computed offline, so a teacher that drifted would make the metric incomparable.

# Part B - EfficientDet-Lite1 backbone (detection), also 30 ep (~10-15 days)
PARTS=B FINAL_EPOCHS=30 CHECKPOINTS="10 20 30" IMPORTANCES="magnitude_l2" ./scripts/pruning.sh

# Breadth-axis classification families (ImageNet only; baselines come from
# download_and_finetune_models.sh, no per-family finetune):
CLS_MODEL=resnet50 PARTS=A FINAL_EPOCHS=5 CHECKPOINTS="30 50 70" \
    IMPORTANCES="magnitude_l2" ./scripts/pruning.sh

# Second detection family - SSDLite320-MobileNetV3-Large (torchvision, input 320):
DET_MODEL=ssdlite_mobilenetv3 COCO_SUBSET=minitrain PARTS=B FINAL_EPOCHS=15 \
    CHECKPOINTS="30 50 70" IMPORTANCES="magnitude_l2" ./scripts/pruning.sh

# The published EfficientDet-Lite1 ladder (18 rungs behind
# results/detection/full_mapping_ladder_lite1.json). Four fields override a default,
# so the plain invocation does not reproduce it. Two traps beyond the skip-if-exists
# one above: every log writes recipe_ft.epochs=100 regardless of FINAL_EPOCHS, that
# field being the recipe's declared default rather than what ran. len(ft_history) is
# what ran, and it is 40 for all 18 published rungs.
PARTS=B DET_MODEL=efficientdet_lite1 COCO_SUBSET=train2017 DET_SCOPE=full \
    CHECKPOINTS="10 20 30 40 50 60 70 80 90" IMPORTANCES="lamp magnitude_l2" \
    FINAL_EPOCHS=40 ./scripts/pruning.sh


# Two smaller budgets, for bring-up rather than for a published ladder. Neither of
# these produced a committed result: classification has no ladder under results/ at
# all (docs/PRUNING_HAZARDS.md section 3), and coco-minitrain is the reduced split.
PARTS=A FINAL_EPOCHS=100 CHECKPOINTS="40 50 60 70 80" IMPORTANCES="magnitude_l2" ./scripts/pruning.sh
COCO_SUBSET=minitrain PARTS=B FINAL_EPOCHS=30 \
    CHECKPOINTS="10 20 30" IMPORTANCES="magnitude_l2" \
    ./scripts/pruning.sh
```

## 4. Phase 2: PyTorch (.pt) → TFLite int8

This is the default export path, and `pip install -e '.[all]'` in section 1 already
installed it. litert-torch + ai-edge-quantizer, per-channel int8, no tensorflow and no
onnx: `litert_torch.convert` takes the `nn.Module` directly, so there is no ONNX step,
and the flatbuffer rewrites in `int8_pruning.convert.flatbuffer` read the schema from
`ai_edge_litert`.

```bash
source ./pruning-env/bin/activate
# Only if section 1 installed a narrow set rather than '.[all]':
EXTRAS='convert' ./scripts/setup_env.sh

# Sanity check:
python -c "import torch, litert_torch, ai_edge_quantizer, ai_edge_litert; print('torch', torch.__version__, '| litert_torch', litert_torch.__version__)"
# Expect torch 2.10.x+cu128 and litert-torch 0.9.1 or newer. torch >= 2.10 is a
# hard floor: on 2.9.1 the torchvision SSDLite export dies inside torch's own
# functionalization pass with "tensor does not have a device".

# Smoke-test one pruned model. Outputs land in outputs/tflite_int8_litert/:
#   <name>_int8.tflite             the final artefact (Edge TPU input)
#   <name>_int8.size.json          size + i/o dtype/shape + calib metadata
#   calib_cache/<family>_n100_seed42.npy   reusable calibration samples
python src/int8_pruning/convert/tflite.py \
    --input outputs/pytorch_pruned/mobilenetv2_flowers102_pruned50pct_magnitude_l2.pt

# Batch every .pt in outputs/pytorch_pruned/ (calibration is built once
# per family and reused across prune ratios):
./scripts/convert.sh

# Useful overrides (env-var driven, same style as pruning.sh):
#   GLOB='mobilenetv2_*pct_*.pt' ./scripts/convert.sh     # subset by family
#   EXTRA_ARGS='--skip-existing' ./scripts/convert.sh     # idempotent re-run
#   EXTRA_ARGS='--keep-intermediate' ./scripts/convert.sh # keep the fp32 .tflite
#   NUM_CALIB=200 SEED=7 EXTRA_ARGS='--rebuild-calib' ./scripts/convert.sh
# One input dir and one output dir per export path, all defaulted in scripts/config.sh:
#   PT_PRUNED_DIR       outputs/pytorch_pruned        (input, every path)
#   TFLITE_DIR          outputs/tflite_int8_litert    (litert)
#   TFLITE_ONNX2TF_DIR  outputs/tflite_int8_onnx2tf   (onnx2tf)
#   ONNX_FP32_DIR       outputs/onnx_fp32             (onnx)

# The other export path. litert-torch above is the default and is what every
# current artefact was built with; torch.onnx.export -> onnx2tf built everything
# dated before 2026-08-22 and is kept so an old artefact can be reproduced.
# litert-torch is the recommended path and the one every current artefact was built
# on; onnx2tf stays available because the two are not interchangeable -- through
# onnx2tf EfficientDet maps 83 of 479 operators to the Edge TPU against litert-torch's
# 305 of 305, so a pre-2026-08-22 artefact reproduces only on the path that built it.
# It needs its OWN venv -- its tensorflow/onnx pins do not coexist with
# pruning-env's (section 1 has the numpy pins that make that a hard constraint).
# The same setup script builds it, behind a toggle:
#   ONNX2TF_ENABLED=1 ./scripts/setup_env.sh
#   . ./onnx2tf-env/bin/activate && EXPORT_PATH=onnx2tf ./scripts/convert.sh
#   QUANT_TYPE=per-channel|per-tensor is that path's one extra knob (default
#   per-channel). It is read only there; what the two settings cost is section 4 of
#   docs/PRUNING_HAZARDS.md.
# -> outputs/tflite_int8_onnx2tf/ (a separate directory: byte counts and
#    packaging_frac are NOT comparable across the two paths).

# The third export path writes fp32 ONNX and stops. It does not quantize, so it
# needs no calibration set and no dataset, and no number under results/ came
# through it. Its purpose is the toolchains that do not read TFLite: TensorRT,
# OpenVINO, Core ML, RKNN. It runs in this same venv, with one extra package
# (already installed by '.[all]'; on a narrow install, EXTRAS='onnx').
EXPORT_PATH=onnx ./scripts/convert.sh
# -> outputs/onnx_fp32/<name>.onnx plus a <name>.onnx.json sidecar recording the
#    opset, the input shape and the output names. Detection models keep all ten
#    head tensors, named cls_0..4 / box_0..4, matching the onnx2tf path.

# Verify the int8 model loads under the Coral host's runtime
# (Feranick build: tflite_runtime 2.16.2 + Python 3.10):
python -c "
from tflite_runtime.interpreter import Interpreter
it = Interpreter('outputs/tflite_int8_litert/mobilenetv2_flowers102_pruned50pct_magnitude_l2_int8.tflite')
it.allocate_tensors()
print(it.get_input_details()[0])
print(it.get_output_details()[0])
"
# Expect: dtype is int8 or uint8 on both input and output, shape (1, 224, 224, 3) NHWC for cls.

# (Optional) sanity-check Edge TPU mapping locally if edgetpu_compiler is installed:
edgetpu_compiler --show_operations \
    outputs/tflite_int8_litert/mobilenetv2_flowers102_pruned50pct_magnitude_l2_int8.tflite
# Expect ALL ops mapped to Edge TPU, 1 subgraph, no CPU fallback block in the
# log. The converter already applies the three rewrites the compiler needs
# (INT64 paddings, PADV2 -> PAD, identity GATHER_ND) -- see
# src/int8_pruning/convert/flatbuffer.py.

# Detection int8 models expose the raw per-level head tensors (10 outputs for
# EfficientDet-Lite1, 12 for SSDLite); anchor decode + NMS run on CPU,
# reproducing the exact PyTorch-baseline decode. Measure COCO mAP / demo with:
python families/detection/effdet/eval.py eval \
    --tflite outputs/tflite_int8_litert/efficientdet_lite1_coco-minitrain_pruned0pct_int8.tflite \
    --coco-root "$DATASET_DIR/coco"
python families/detection/ssdlite/eval.py eval \
    --tflite outputs/tflite_int8_litert/ssdlite_mobilenetv3_coco-minitrain_pruned0pct_int8.tflite \
    --coco-root "$DATASET_DIR/coco"
# Single-image demo: swap `eval ...` for `predict --image some.jpg --draw out.jpg`
```

## 5. Phase 3 (optional): TFLite int8 → Edge TPU (Coral USB)

Everything above this point is the pipeline proper and needs no Coral and no
`edgetpu_compiler`: sections 3 and 4 take a checkpoint to a finished int8 `.tflite` on a
plain machine. This section is an optional backend. Nothing chains into it. `compile_edgetpu.sh` is a driver you invoke yourself, and it exits 1 if the
compiler is missing, because compiling is the only thing it does. Where the Edge TPU is
incidental rather than the goal, the absence is reported instead: `scripts/deliver.sh`
prints `[SKIPPED] edgetpu/` and packages the int8 artifacts alone, and
`families/re-identification/reid/report.py` records `skipped` for its compile leg.

Stop here if you do not have the hardware. The code that needs it is confined to
`src/int8_pruning/backends/edgetpu/`, and nothing else in the package imports from there.


```bash
# Install the Edge TPU compiler once (see https://coral.ai/docs/edgetpu/compiler/):
#   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
#   echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
#       | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
#   sudo apt-get update && sudo apt-get install edgetpu-compiler
edgetpu_compiler --version

# Batch-compile every *_int8.tflite under outputs/tflite_int8_litert/. Each invocation
# of compile_edgetpu.sh creates a fresh run dir under outputs/edgetpu/
# holding the *_edgetpu.tflite artefacts, the compiler .log files, and the
# corresponding *_int8.size.json sidecars (so the I/O dtype/shape stays
# discoverable without reopening the .tflite).
./scripts/compile_edgetpu.sh
# -> outputs/edgetpu/single_<TS>/<stem>_int8_edgetpu.tflite

# Filter a subset (e.g. only MobileNetV2 magnitude_l2 runs):
GLOB='mobilenetv2_*pct_magnitude_l2_int8.tflite' ./scripts/compile_edgetpu.sh

# Just inspect the op mapping without keeping artefacts? Add SHOW_OPS=1; the
# script dumps <stem>.show_ops.txt next to the compiled output.
SHOW_OPS=1 GLOB='mobilenetv2_flowers102_pruned50pct_magnitude_l2_int8.tflite' \
    ./scripts/compile_edgetpu.sh

# "Combinations of 2 models" path: co-compile two models in one invocation so
# they share the on-chip parameter cache on the Coral USB device. Measured pairs
# and their on/off-chip splits: docs/edgetpu_cocompile.md. Use the bare
# stems (no _int8.tflite suffix); both files must already live in TFLITE_DIR.
PAIR="mobilenetv2_flowers102_pruned50pct_magnitude_l2 efficientdet_lite1_coco-train2017_pruned10pct_magnitude_l2" \
    ./scripts/compile_edgetpu.sh
# -> outputs/edgetpu/pair_<stemA>__<stemB>_<TS>/

# Useful overrides (env-var driven, same style as convert.sh):
#   TFLITE_DIR=path/to/int8_models ./scripts/compile_edgetpu.sh
#   EXTRA_ARGS='-m 13' ./scripts/compile_edgetpu.sh       # pin runtime version
#   EXTRA_ARGS='-a'    ./scripts/compile_edgetpu.sh       # enable multi-subgraph
#   EDGETPU_COMPILER=/opt/edgetpu/bin/edgetpu_compiler ./scripts/compile_edgetpu.sh

# On-device sanity check (run on the Coral host, not here):
#   from pycoral.utils.edgetpu import make_interpreter
#   it = make_interpreter('mobilenetv2_..._int8_edgetpu.tflite')
#   it.allocate_tensors()
```

## 6. Fetching the published artifacts instead of building them

Everything above builds the models. `scripts/fetch_deliverables.sh` downloads the ones
already published, which is what the downstream MicroROS-Pi5_Coral_TPU repo does. Model
binaries never enter git: they live in a HuggingFace repo, and every file is verified
against [`results/deliverables.sha256`](../results/deliverables.sha256) on the way in.
Files whose sha256 already matches are skipped, so an interrupted download resumes by
re-running. It needs curl (or wget) and sha256sum, nothing else.

Two tiers, because the `.pt` checkpoints are 54% of the payload and nothing that merely
*runs* a model needs them:

| tier | contents | size |
|---|---|---|
| `deploy` (default) | `models_int8_tflite/` + `edgetpu/` + `reference/` | 162 files, 354 MB |
| `full` | everything, adds `checkpoints_pytorch/` | 202 files, 763 MB |

Take `full` to re-prune, fine-tune further, or inspect the pruned structure. Its 0% row
is the unpruned baseline, kept so the package plots a complete ratio curve on its own.
Both tiers cover both published families.

```bash
./scripts/fetch_deliverables.sh                 # deploy tier, both families
./scripts/fetch_deliverables.sh --tier full     # + the .pt checkpoints
./scripts/fetch_deliverables.sh effdet_lite1_pruning   # one family
./scripts/fetch_deliverables.sh --list
./scripts/fetch_deliverables.sh --verify        # check what is already on disk
./scripts/fetch_deliverables.sh --force         # re-download even if valid
```

While the HuggingFace repo is private a token is also required, from `hf auth login` or
`HF_TOKEN=...` in the environment. Without one every request returns 401.

The checkpoints are full-module pickles, so `torch.load(weights_only=False)` rebuilds the
pruned module graph and the classes have to be importable. They come from pip
(`effdet==0.4.1`, `timm==1.0.27`, `omegaconf==2.3.0`), not from this repository. The exact
pins and the load call are in `checkpoints_pytorch/READ_THIS_FIRST.md`, which the script
always fetches alongside them.

The manifest is a subset of `deliverables/`, not a hash of it: the other packages in that
tree are downstream handoffs and are not published. Which ones qualify, and why, is in
`scripts/manifest_deliverables.sh`.
