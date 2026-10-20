# INT8 Pruning Pipeline

**English** | [简体中文](README.zh-CN.md)

[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

INT8 Pruning Pipeline is a Python library that takes a trained PyTorch model through
structured pruning, accuracy recovery and export, as plain functions called from a script.
An FP32 model goes in; an INT8 TFLite model, optionally compiled for the Edge TPU, or an
FP32 ONNX graph comes out.

![Structured pruning with torch_pruning, then recovery by supervised fine-tuning on the dataset's own labels or by distilling against the unpruned original FP32 model, frozen throughout, then three export exits: litert-torch with ai-edge-quantizer (recommended) and torch.onnx.export with onnx2tf both produce an int8 TFLite model, which edgetpu_compiler compiles for the Edge TPU, while a separate torch.onnx.export leg produces an fp32 ONNX model, unquantized](docs/fig_pipeline_en.svg)

Structured pruning is DepGraph through [`torch_pruning`](https://github.com/VainF/Torch-Pruning).
Recovery takes one of two routes: fine-tuning on the dataset's own labels, or distilling
against the unpruned network frozen as its own teacher. The default int8 export path is
`litert-torch` with `ai-edge-quantizer`; `torch.onnx.export` with `onnx2tf` is the
alternative, and a third path writes an fp32 ONNX graph. Six examples run the library end
to end on eleven published baselines, across classification, COCO detection, person Re-ID
and open-vocabulary recognition.

## Quick start

`examples/minimal.py` runs every stage on a small CNN and random data in seconds on a
CPU. The commands after it take MobileNetV2 through every stage on Flowers-102, the
smallest dataset the download script fetches, with one ratio and one recovery epoch to
keep the run short.

```bash
./scripts/setup_env.sh                       # creates ./pruning-env
source ./pruning-env/bin/activate
python examples/minimal.py
mkdir -p data/datasets
python examples/download_datasets.py --dataset flowers102   # ~330 MB
python examples/cnn_backbones.py fetch-baseline --model mobilenetv2 --dataset flowers102
python examples/cnn_backbones.py prune --model mobilenetv2 --dataset flowers102 \
    --ratios 0.5 --epochs 1
python examples/cnn_backbones.py int8 --dataset flowers102 \
    --pt outputs/pytorch_pruned/mobilenetv2_flowers102_pruned50pct_magnitude_l2.pt
```

Every example takes `--help`. Adding `--edgetpu` to `int8` also compiles the result for
the Edge TPU, which needs `edgetpu_compiler`. Environment, datasets and every example in
full: **[`docs/SETUP.md`](docs/SETUP.md)**.

## Using it on your own model

The calling script supplies the model, its data and its metric:

```python
import copy
import torch
import int8_pruning

int8_pruning.seed_everything(42)
model = ...                                    # any nn.Module, in eval mode
teacher = copy.deepcopy(model)                 # only for distillation
example = torch.randn(1, 3, 224, 224)

info = int8_pruning.prune(model, example, 0.5, ignored_layers=[model.classifier])
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)   # after pruning
int8_pruning.finetune(model, train_loader, 30, optimizer=opt,
                      scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30),
                      evaluate=lambda: {"top1": top1(model, val_loader)}, select="top1")
# with no labels:  int8_pruning.distill(model, teacher, unlabelled_loader, 10, optimizer=opt)

calib = int8_pruning.export.image_folder("calibration_images/", 100, 224, mean, std)
int8_pruning.export_int8(model, calib, "outputs/tflite_int8_litert/model_int8.tflite")
int8_pruning.compile_edgetpu("outputs/tflite_int8_litert/model_int8.tflite")   # optional
```

The public API is the set of names in `int8_pruning.__all__`; modules whose names start
with an underscore are internal. `ratio` is the fraction of parameters removed, measured
against the model as it was before the call. Models that need more than that are covered
by the options below, each shown in one of the examples:

| the model | pass | example |
|---|---|---|
| has a head whose output width must stay | `ignored_layers=[head]` | every one |
| trains on a loss other than CrossEntropy on (image, label) | `finetune(..., train_step=fn)`, `fn(model, batch) -> loss` | [`ssdlite.py`](examples/ssdlite.py), [`efficientdet.py`](examples/efficientdet.py) |
| has no labels to train on | `distill(student, teacher, loader, ...)` | [`clip_rn50.py`](examples/clip_rn50.py), [`relu_clip.py`](examples/relu_clip.py), [`youreid.py`](examples/youreid.py) |
| cannot be traced as it stands | `prune(..., traced=wrapper)`, a module sharing its submodules | [`ssdlite.py`](examples/ssdlite.py) |
| should count the ratio over part of the model | `prune(..., scope=lambda m: m.backbone)` | [`ssdlite.py`](examples/ssdlite.py), [`efficientdet.py`](examples/efficientdet.py) |
| was already cut down another way before pruning | `prune(..., dense_params=n)` | [`youreid.py`](examples/youreid.py) `--embed-dim` |
| is pruned in several rounds, recovering in between | `for ratio, info in int8_pruning.prune_trajectory(model, example, ratios)` | [`cnn_backbones.py`](examples/cnn_backbones.py) `--iterative` |
| lowers into operators the Edge TPU does not map | `export_int8(..., graph_patches=[...])` | [`ssdlite.py`](examples/ssdlite.py), [`efficientdet.py`](examples/efficientdet.py), [`youreid.py`](examples/youreid.py) |

A model whose structure is far from what these tools require (their repositories state
the requirements) can be distilled into a CNN student first, as in
[`jiaheguo521/relu-clip`](https://github.com/jiaheguo521/relu-clip).

## How the stages work

Pruning is data-free: every importance criterion is computed from the weights, and
`example_inputs` is only traced. Recovery is where the compute goes. Distillation needs no
annotation, because the teacher is a `copy.deepcopy` of the network taken before pruning.
A distillation step costs more than a labelled one, and the gap widens as pruning
progresses, because the teacher keeps its size while the student shrinks.

A ladder is a set of rungs: one model cut to one target parameter reduction, then
recovered. By default each rung is pruned from the dense model independently.
`prune_trajectory` walks one trajectory instead, feeding each recovered model into the
next round of pruning.

Quantization is per channel by default. `litert-torch` with `ai-edge-quantizer` is the
recommended path: on detection it maps the whole graph onto the accelerator. The two int8
paths lower the same graph differently, so the share of a model that reaches the
accelerator depends on the path.

Pruning, recovery and export run on any machine and end at the int8 `.tflite` or the fp32
`.onnx`. Compiling for the Edge TPU (`int8_pruning.compile_edgetpu`, which needs `edgetpu_compiler`)
and timing on a Coral (`int8_pruning.latency`) are optional steps after that.

## Examples

| example | baselines | task | recovery |
|---|---|---|---|
| [`cnn_backbones.py`](examples/cnn_backbones.py) | MobileNetV2, EfficientNet-Lite0, MnasNet, SqueezeNet, ResNet-50 | ImageNet / Flowers-102 / iNat Plantae top-1 | labelled |
| [`efficientdet.py`](examples/efficientdet.py) | EfficientDet-Lite1, Lite2 | COCO detection (mAP) | labelled |
| [`ssdlite.py`](examples/ssdlite.py) | SSDLite320-MobileNetV3-Large | COCO detection (mAP) | labelled |
| [`clip_rn50.py`](examples/clip_rn50.py) | CLIP RN50 image tower | open-vocabulary zero-shot | distilled |
| [`relu_clip.py`](examples/relu_clip.py) | [relu-clip](https://github.com/jiaheguo521/relu-clip)'s EfficientNet-Lite4 student | open-vocabulary zero-shot | distilled |
| [`youreid.py`](examples/youreid.py) | YouReID baseline_lite | person Re-ID (mAP / CMC) | distilled |

Each has the same three subcommands: `fetch-baseline` writes the dense model, `prune`
prunes it to each of `--ratios` and recovers it, and `int8` exports a pruned model and
scores it. `clip_rn50.py` has a fourth, `text-embeddings`, for its zero-shot classifier. A
pruned checkpoint is named

```
<stem>_pruned<P>pct_<criterion>[_iter][_full].pt
```

`<stem>` is the baseline plus the data it was recovered on (`mobilenetv2_imagenet`,
`efficientdet_lite1_coco-train2017`), `<P>` the percentage of parameters removed, and
`<criterion>` the importance criterion. `_iter` marks a trajectory, and `_full` an
EfficientDet ladder measured over the whole model (`--scope full`) rather than its backbone.
The two detection examples measure `<P>` over the backbone by default and every other
example over the whole model, so filenames are comparable only within one example; each
run's JSON log records the realized ratio.

## Related paper

INT8 Pruning Pipeline extends the toolchain used in the paper below:

> M. Zouhdi, J. Guo, R. Hammadi, B. Sun, P. Leleux, T. Kłoda, M. Caccamo.
> *Pruning of Deep Neural Networks for Real-Time Execution on Edge TPU.*
> 29th Euromicro Conference on Digital System Design (DSD), Kraków, Poland, 2026.

## Running the tests

```bash
python3 -m unittest discover -s tests
```

The suite runs pruning, recovery and export on a small CNN, every example as far as its
`--help`, and the document checks: links, images, en/zh parity, and a documentation
entry for every script in `scripts/`. It needs only the standard library; tests that need
torch or the export packages are skipped when those are absent.

## Repository layout

```
src/int8_pruning/          the library: import int8_pruning
  pruning/                 prune, prune_trajectory, count_params
  recovery/                finetune, distill
  export/
    _int8.py               export_int8: int8 TFLite, dispatching on backend=
    _litert.py             backend="litert" (default): litert-torch + ai-edge-quantizer
    _onnx2tf.py            backend="onnx2tf": torch.onnx.export + onnx2tf
    _onnx.py               export_onnx: fp32 ONNX
    _calibration.py        image_folder, normalize_nhwc: calibration sets
    _patches.py            se_mean_pool, adaptive_max_pool: module rewrites during export
    _edgetpu_fixups.py     .tflite rewrites applied after quantization
    _outputs.py            flattening and naming nested model outputs
    _tflite_size.py        .tflite byte budget
  compilation/             compile_edgetpu: runs edgetpu_compiler
  evaluation/              make_interpreter, run_tflite, latency; runs by path on its own
  _seed.py                 seed_everything
examples/                  one script per model family, shared dataset code, dataset downloader
scripts/setup_env.sh       builds the virtual environments
tests/                     library and example tests
docs/                      setup guide and the pipeline figure
```

Each stage of the pipeline is a package whose `__init__.py` only re-exports, from modules
whose names start with an underscore; everything else in the library is such a module.
[`tests/test_public_api.py`](tests/test_public_api.py) holds the list of public names.

`outputs/` (pipeline products) and `data/` (datasets) are generated and gitignored.

## License and acknowledgements

Apache-2.0; see [`LICENSE`](LICENSE). Published model weights remain subject to the terms
of their upstream sources.

This work was carried out at **LAAS-CNRS**, Toulouse, France, on LAAS-CNRS computing
infrastructure, under the supervision of **Tomasz Kłoda**. He recommended the research
direction this work followed and is a co-author of the DSD 2026 paper. During the
internship he gave me a great deal of research freedom, with his guidance available
whenever I needed it.

The library builds on `torch_pruning`, `litert-torch` and `ai-edge-quantizer`
(google-ai-edge), `timm`, `efficientdet-pytorch`, `open_clip`, `deep-person-reid` and
`google-coral`.
