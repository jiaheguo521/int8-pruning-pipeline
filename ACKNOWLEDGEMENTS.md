# Acknowledgements

## Institutional context

This work was carried out at **LAAS-CNRS**, Toulouse, France, under the supervision of
**Tomasz Kłoda**. The pruning and recovery fine-tuning runs reported in
[`results/`](results/) were executed on LAAS-CNRS computing infrastructure.

He recommended the research direction this work followed, supervised the downstream on-car
segmentation work behind the `line_seg` family, and is a co-author of the DSD 2026 paper
whose evaluation protocol is carried over here.

I am grateful for the freedom he gave me in deciding what to measure, which families to
open and which negative results to publish, with his guidance available throughout.

## Upstream projects

This pipeline is glue around other people's work:

- [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning) (MIT): the structured
  pruning engine every family's worker is built on.
- [google-ai-edge/litert-torch](https://github.com/google-ai-edge/litert-torch) and
  [ai-edge-quantizer](https://github.com/google-ai-edge/ai-edge-quantizer) (Apache-2.0):
  the PyTorch → int8 TFLite path this repo converts through. `litert_torch.convert`
  takes the `nn.Module` directly, so there is no ONNX step and no TFLiteConverter
  fallback.
- [PINTO0309/onnx2tf](https://github.com/PINTO0309/onnx2tf) (MIT): the ONNX → int8 TFLite
  path this repo used until 2026-08-22. Every number committed before that date came
  through it; see `docs/PRUNING_HAZARDS.md` §4.
- [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
  (Apache-2.0), [rwightman/efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch)
  (Apache-2.0), [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
  (MIT), and
  [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
  (MIT): model definitions and pretrained weights.
- [google-coral](https://github.com/google-coral): `edgetpu_compiler`,
  `libedgetpu`, and the conventions this repo follows for compiled artifacts
  (the `_edgetpu.tflite` suffix, keeping the pre-compile `.tflite` alongside).

## Contributions

The `imagenet_backbones` worker and the Edge TPU benchmark path build on an earlier
CIFAR-100 study by **Mouad Zouhdi**, whose *evaluation* protocol this repository follows;
he is listed as an author in [`CITATION.cff`](CITATION.cff) for that reason. That study is
the DSD 2026 paper cited in the [README](README.md#citation). Cite that paper, not this
repository, when referring to the CIFAR-100 results themselves.

To be precise about what belongs to whom: **neither that paper nor this repository
proposes a pruning method.** The structured-pruning machinery is DepGraph
(Fang et al. 2023), used through its reference implementation `torch_pruning`, and every
importance criterion comes from its own prior paper (see below). What DSD 2026 contributes
is an *evaluation* of that machinery on the Edge TPU, on CIFAR-100 classification:
which models, which ratios, which criteria, which recovery schedule, which quantization
configuration, and how latency is measured. What this repository contributes is carrying that same evaluation
onto detection, segmentation, person Re-ID and open-vocabulary classification, and
reporting where it holds and where it does not.

## References

The methods this pipeline applies, each from its own source: the papers a reader needs
in order to know what the code is doing.

**Structured pruning engine**

- G. Fang, X. Ma, M. Song, M. B. Mi, X. Wang. *DepGraph: Towards Any Structural Pruning.*
  CVPR 2023. [arXiv:2301.12900](https://arxiv.org/abs/2301.12900): the dependency-graph
  formulation behind [`torch_pruning`](https://github.com/VainF/Torch-Pruning), which is
  what actually removes channels in every family here.

**Importance criteria implemented in [`src/int8_pruning/prune/core.py`](src/int8_pruning/prune/core.py)**

| flag | `torch_pruning` class | source |
|---|---|---|
| `magnitude_l1`, `magnitude_l2` | `MagnitudeImportance(p=1\|2)` | H. Li, A. Kadav, I. Durdanovic, H. Samet, H. P. Graf. *Pruning Filters for Efficient ConvNets.* ICLR 2017. [arXiv:1608.08710](https://arxiv.org/abs/1608.08710) |
| `fpgm` | `FPGMImportance()` | Y. He, P. Liu, Z. Wang, Z. Hu, Y. Yang. *Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration.* CVPR 2019. [doi:10.1109/CVPR.2019.00447](https://doi.org/10.1109/CVPR.2019.00447) |
| `lamp` | `LAMPImportance(p=2)` | J. Lee, S. Park, S. Mo, S. Ahn, J. Shin. *Layer-adaptive Sparsity for the Magnitude-based Pruning.* ICLR 2021. [arXiv:2010.07611](https://arxiv.org/abs/2010.07611) |
| `random` | `RandomImportance()` | control |

Criteria named by the DSD 2026 protocol but **not** run here, and the reason for each,
are in [§3 of `docs/PRUNING_HAZARDS.md`](docs/PRUNING_HAZARDS.md): `bn_scale`
(Z. Liu et al., *Learning Efficient Convolutional Networks through Network Slimming*,
ICCV 2017), `taylor` (P. Molchanov et al., *Importance Estimation for Neural Network
Pruning*, CVPR 2019) and `obdc` (Y. LeCun et al., *Optimal Brain Damage*, NeurIPS 1989;
C. Wang et al., *EigenDamage*, ICML 2019).

**Model and dataset behind the published artifacts**

- M. Tan, R. Pang, Q. V. Le. *EfficientDet: Scalable and Efficient Object Detection.*
  CVPR 2020. [arXiv:1911.09070](https://arxiv.org/abs/1911.09070)
- T.-Y. Lin et al. *Microsoft COCO: Common Objects in Context.* ECCV 2014.
  [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)

**Edge TPU**

- Coral. *USB Accelerator Datasheet.* https://coral.ai/docs/accelerator/datasheet/
  Source for the 8 MiB on-chip parameter cache every latency claim here turns on.
- K. Seshadri, B. Akin, J. Laudon, R. Narayanaswami, A. Yazdanbakhsh. *An Evaluation of
  Edge TPU Accelerators for Convolutional Neural Networks.*
  [arXiv:2102.10423](https://arxiv.org/abs/2102.10423)
- H. T. Kung. *Why Systolic Architectures?* IEEE Computer 15(1), 1982.
  [doi:10.1109/MC.1982.1653825](https://doi.org/10.1109/MC.1982.1653825). Background for
  why padding channel counts to the array width (W = 64 here) predicts latency better than
  raw MACs do.
- H. Gao, H. Choi, Y. Wang. *Work-In-Progress: Modeling and Analysis of Inference Latency
  on USB Edge TPUs.* RTSS 2025. [doi:10.1109/RTSS66672.2025.00058](https://doi.org/10.1109/RTSS66672.2025.00058)

**On recovery fine-tuning**

- Z. Liu, M. Sun, T. Zhou, G. Huang, T. Darrell. *Rethinking the Value of Network Pruning.*
  ICLR 2019. [arXiv:1810.05270](https://arxiv.org/abs/1810.05270). Why the pruned
  *structure* plus retraining, rather than the inherited weights, is what this pipeline's
  recovery stage is really exploiting.
