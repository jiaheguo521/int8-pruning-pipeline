> English | [中文版](SETUP.zh-CN.md) · [← back to the README](../README.md)

# Setup

Building the environment, fetching data, and running each example from the dense baseline
to a scored int8 model. Every flag of every example is listed in its `--help`; this page
covers how the pieces fit together. Compiling for the Edge TPU is an optional last step,
in section 5.

## 1. Environment

```bash
./scripts/setup_env.sh                       # ./pruning-env: torch, every example, the litert export path
source ./pruning-env/bin/activate
```

The script installs torch from the CUDA-matched wheel index first, then
`pip install -e '.[all]'`. It is idempotent: an existing venv is reused and pip
re-resolves in place, so re-running it after a `pyproject.toml` change updates the venv.
Every install goes through `<venv>/bin/pip`, so no venv needs to be active.

Environment variables, listed at the top of
[`scripts/setup_env.sh`](../scripts/setup_env.sh), configure it. The ones that matter on a
different machine:

```bash
PY=python3.12 ./scripts/setup_env.sh                                     # interpreter, >=3.10
TORCH_INDEX=https://download.pytorch.org/whl/cpu ./scripts/setup_env.sh  # no CUDA
EXTRAS='effdet,convert' ./scripts/setup_env.sh                           # one example only
```

The litert export path requires torch 2.10 or newer: on 2.9.1 the SSDLite export fails
inside torch's functionalization pass.

**Extras.** `[all]` is every example plus the litert and fp32 ONNX export paths. A narrower
install names the examples it needs: `effdet`, `ssdlite`, `clip-rn50`, `relu-clip` and
`youreid`, plus `convert` for the int8 export. `cnn_backbones.py` and `minimal.py` need
only the core dependencies, as do pruning and recovery on their own.

**torchreid.** `youreid.py` scores Re-ID mAP with torchreid, which is kept out of every
extra because its setup pins an old torch and would downgrade the stack. Install it without
its dependencies; the `youreid` extra carries what it imports at runtime.

```bash
pip install --no-deps 'git+https://github.com/KaiyangZhou/deep-person-reid.git'
```

**The onnx2tf export path** needs a venv of its own, because onnx2tf and tensorflow pin
numpy versions incompatible with the litert path. It is needed only by scripts that call
`int8_pruning.export_int8(..., backend="onnx2tf")`; the examples all use the litert path.

```bash
ONNX2TF_ENABLED=1 ./scripts/setup_env.sh     # also builds ./onnx2tf-env
```

## 2. Datasets

[`examples/download_datasets.py`](../examples/download_datasets.py) fetches the datasets the
examples read. `--dataset` selects what to fetch, and `--dataset-dir` (default
`data/datasets`, the same default every example uses) must already exist.

```bash
mkdir -p data/datasets
python examples/download_datasets.py --dataset flowers102            # ~330 MB
python examples/download_datasets.py --dataset coco coco_minitrain   # ~25 GB, for detection
python examples/download_datasets.py --dataset market1501            # ~140 MB, for youreid
```

| dataset | used by | note |
|---|---|---|
| `flowers102` | `cnn_backbones.py` | the smallest; the quick start uses it |
| `imagenet` | `cnn_backbones.py`, `clip_rn50.py`, `relu_clip.py` | checks the ImageFolder layout under `Imagenet_1k/`; ILSVRC2012 itself needs a registered account and a manual download |
| `inat_plant` | `cnn_backbones.py` | a ~198 GB stream without resume; `--inat-mirror-url` fetches a resumable Plantae-only mirror |
| `coco` | `efficientdet.py`, `ssdlite.py` | train2017, val2017 and annotations |
| `coco_minitrain` | `efficientdet.py`, `ssdlite.py` | a 25k-image subset of train2017, about 5x faster to recover on; needs `coco` |
| `market1501` | `youreid.py` | |

`cnn_backbones.py --dataset imagenet` requires exactly 1000 classes under `Imagenet_1k/`.
The two CLIP examples only distil on those images and accept any ImageFolder there.

## 3. Running an example

Every example is the same three steps, each a subcommand:

```bash
python examples/<example>.py fetch-baseline ...                  # -> outputs/models/
python examples/<example>.py prune --ratios 0.3 0.5 ...          # -> outputs/pytorch_pruned/, outputs/pruning_logs/
python examples/<example>.py int8 --pt outputs/pytorch_pruned/<stem>.pt   # -> outputs/tflite_int8_litert/
```

`fetch-baseline` writes the dense model as a full `nn.Module`. `prune` takes it to each
ratio in `--ratios` (fractions of the parameters, e.g. `0.5`), recovers it, and saves each
rung with a JSON log of its recovery history and realized ratio. `int8` exports a pruned
`.pt` to int8 and scores it; with `--tflite`, it scores an existing int8 or
`_edgetpu.tflite`. `--models-dir`, `--dataset-dir` and `--out-dir` move the
three roots; `--importance` picks one of `magnitude_l1`, `magnitude_l2` (the default),
`fpgm`, `random` and `lamp`.

**`cnn_backbones.py`** prunes MobileNetV2, EfficientNet-Lite0, MnasNet, SqueezeNet or
ResNet-50 (`--model`) on ImageNet, and MobileNetV2 also on Flowers-102 or iNat Plantae
(`--dataset`). On ImageNet the pretrained download is the baseline; on the other two,
`fetch-baseline` finetunes it onto the dataset first (`--epochs`, default 20). Recovery is
the PruningBench ImageNet recipe, 90 epochs; set `--epochs` lower for a first pass.
`--iterative` walks one trajectory through the ratios, each pruned from the previous
recovered rung, and adds `_iter` to the name.

```bash
python examples/cnn_backbones.py fetch-baseline --model resnet50
python examples/cnn_backbones.py prune --model resnet50 --ratios 0.3 0.5 0.7 --epochs 5
```

**`efficientdet.py`** prunes EfficientDet-Lite1 or Lite2 (`--model`) on COCO.
`--coco-subset minitrain` is the fast recovery set; `val2017` is a smoke test only, since
mAP is scored on it too. `--ratios` is measured over the backbone; `--scope full` measures
it over the whole model and adds `_full` to the name. A high target under `--scope full`
can land short of the ratio asked for, so read `actual_ratio` from the log. The int8 model
ends at the ten raw head tensors, and anchor decoding and NMS run on the CPU.

```bash
python examples/efficientdet.py fetch-baseline --model efficientdet_lite1
python examples/efficientdet.py prune --model efficientdet_lite1 --ratios 0.3 0.5 \
    --coco-subset minitrain --epochs 30
```

**`ssdlite.py`** is SSDLite320-MobileNetV3-Large on COCO, with the same subsets, the same
backbone ratio and the same CPU decoding as `efficientdet.py`.

**`clip_rn50.py`** prunes the CLIP RN50 image tower and recovers it by distilling against
the unpruned tower on unlabelled ImageNet train images. Zero-shot scoring needs text
embeddings of the label names, computed once with open_clip by the fourth subcommand:

```bash
python examples/clip_rn50.py fetch-baseline
python examples/clip_rn50.py text-embeddings --preset imagenette
python examples/clip_rn50.py prune --ratios 0.3 0.5 \
    --text-emb outputs/models/clip_rn50_text_imagenette.npz \
    --val-images-dir data/datasets/Imagenet_1k/train
```

Without `--text-emb` and `--val-images-dir`, `prune` distils without scoring.

**`relu_clip.py`** prunes relu-clip's EfficientNet-Lite4 student the same way, and tracks
the mean cosine to the unpruned model; `--text-emb` takes the published `.npz` for
zero-shot. `--global-pruning` ranks channels across layers rather than within each, and
can strip this backbone's narrow early layers. Its `int8` scores zero-shot when given the
same `--text-emb` and `--val-images-dir`, and otherwise only exports.

**`youreid.py`** prunes YouReID's `baseline_lite` on Market-1501 crops, distilling without
labels. Its 768-d embedding layer is protected by default; `--embed-dim` below 768 prunes
that layer first and gives the budget to the backbone, which needs
`--distill-loss simmat`. `--extra-image-dirs` adds unlabelled crops, from the deployment
camera for instance. Scoring mAP in `int8` needs torchreid (section 1).

## 4. Export paths

Every example exports with the default path, `litert-torch` and `ai-edge-quantizer`,
per-channel. The other two paths are available as library calls:

```python
int8_pruning.export_int8(model, calib, "model_int8.tflite", backend="onnx2tf")     # in ./onnx2tf-env
int8_pruning.export_int8(model, calib, "model_int8.tflite", backend="onnx2tf", quant="per-tensor")
int8_pruning.export_onnx(model, "model.onnx", (1, 3, 224, 224))                    # fp32, no calibration
```

The two int8 paths lower the same graph differently, so their files differ in bytes and in
how much of a model the Edge TPU maps; compare measurements only within one path. The fp32
ONNX path is unquantized and feeds runtimes that take ONNX, such as TensorRT, OpenVINO,
Core ML and RKNN. It needs the `onnx` extra (in `[all]`).

## 5. Edge TPU (optional)

Sections 1 to 4 need neither Coral hardware nor `edgetpu_compiler`. Compiling an int8
`.tflite` for the Edge TPU needs the compiler, installed once from Coral's apt repository
(see [coral.ai](https://coral.ai/docs/edgetpu/compiler/)):

```bash
sudo apt-get install edgetpu-compiler        # after adding Coral's apt source
python examples/cnn_backbones.py int8 --dataset flowers102 --edgetpu \
    --pt outputs/pytorch_pruned/mobilenetv2_flowers102_pruned50pct_magnitude_l2.pt
# -> outputs/edgetpu/<stem>_int8_edgetpu.tflite, plus the compiler's log
```

Every example's `int8` takes `--edgetpu`: it compiles the int8 model into
`<out-dir>/edgetpu/` and scores the compiled one on an attached Coral, through the
libedgetpu delegate. `--tflite <stem>_int8_edgetpu.tflite` scores an already compiled
model. From a script:

```python
int8_pruning.compile_edgetpu("a_int8.tflite", "out/", show_ops=True)       # also writes the operator mapping
int8_pruning.compile_edgetpu(["a_int8.tflite", "b_int8.tflite"], "out/")    # together: they share the on-chip cache
```

`extra_args` passes flags through to the compiler, such as `("-m", "13")` to pin the
runtime version.

`int8_pruning.latency(path)` times a compiled model (15 warm-up and 100 timed passes) and needs
`libedgetpu` with `tflite_runtime` or `ai_edge_litert`.
[`evaluation/_evaluation.py`](../src/int8_pruning/evaluation/_evaluation.py) is self-contained, so it also
runs by path inside a docker image that ships a runtime and `libedgetpu`, with the Coral
USB passed through; its header has the `docker run` line.
