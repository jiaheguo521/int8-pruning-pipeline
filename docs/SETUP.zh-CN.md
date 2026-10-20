> 中文版 | [English](SETUP.md) · [← 回到 README](../README.zh-CN.md)

# 环境搭建与运行

搭环境、取数据，并把每个示例从稠密基线跑到一个评测过的 int8 模型。每个示例的全部参数列在它的
`--help` 里；本页说明各部分如何配合。编译成 Edge TPU 产物是可选的最后一步，见第 5 节。

## 1. 环境

```bash
./scripts/setup_env.sh                       # ./pruning-env：torch、全部示例、litert 导出路径
source ./pruning-env/bin/activate
```

脚本先从与 CUDA 匹配的 wheel 源装 torch，再 `pip install -e '.[all]'`。它是幂等的：已有的 venv
会被复用，pip 原地重新解析依赖，所以改了 `pyproject.toml` 之后重跑一遍即可更新 venv。
所有安装都走 `<venv>/bin/pip`，无需先激活 venv。

脚本由环境变量配置，完整列表在 [`scripts/setup_env.sh`](../scripts/setup_env.sh) 开头。
换一台配置不同的机器时常用的是这几个：

```bash
PY=python3.12 ./scripts/setup_env.sh                                     # 解释器，>=3.10
TORCH_INDEX=https://download.pytorch.org/whl/cpu ./scripts/setup_env.sh  # 没有 CUDA
EXTRAS='effdet,convert' ./scripts/setup_env.sh                           # 只装一个示例
```

litert 导出路径要求 torch 2.10 或更新：在 2.9.1 上 SSDLite 的导出会在 torch 的 functionalization
阶段失败。

**Extras。** `[all]` 是全部示例加上 litert 与 fp32 ONNX 两条导出路径。装得更窄时，按需要的示例
选 `effdet`、`ssdlite`、`clip-rn50`、`relu-clip`、`youreid`，int8 导出再加 `convert`。
`cnn_backbones.py`、`minimal.py` 以及单独的剪枝和恢复只需要核心依赖。

**torchreid。** `youreid.py` 用 torchreid 算 Re-ID mAP。torchreid 的 setup 锁了旧版 torch，
会把整套依赖降级，所以不放进任何 extra，而是不带依赖地单独安装；它运行时 import 的包由
`youreid` extra 提供。

```bash
pip install --no-deps 'git+https://github.com/KaiyangZhou/deep-person-reid.git'
```

**onnx2tf 导出路径**需要单独的 venv，因为 onnx2tf 和 tensorflow 锁定的 numpy 版本与 litert 路径不兼容。
只有调用 `int8_pruning.export_int8(..., backend="onnx2tf")` 的脚本才需要它；示例全部走 litert 路径。

```bash
ONNX2TF_ENABLED=1 ./scripts/setup_env.sh     # 另建 ./onnx2tf-env
```

## 2. 数据集

[`examples/download_datasets.py`](../examples/download_datasets.py) 下载示例要读的数据集。
`--dataset` 选择要取的数据集；`--dataset-dir`（默认 `data/datasets`，与所有示例的默认值相同）
必须事先存在。

```bash
mkdir -p data/datasets
python examples/download_datasets.py --dataset flowers102            # 约 330 MB
python examples/download_datasets.py --dataset coco coco_minitrain   # 约 25 GB，检测用
python examples/download_datasets.py --dataset market1501            # 约 140 MB，youreid 用
```

| 数据集 | 使用者 | 说明 |
|---|---|---|
| `flowers102` | `cnn_backbones.py` | 最小的一个，快速上手用的就是它 |
| `imagenet` | `cnn_backbones.py`、`clip_rn50.py`、`relu_clip.py` | 检查 `Imagenet_1k/` 下的 ImageFolder 布局；ILSVRC2012 本身需要注册账号后手动下载 |
| `inat_plant` | `cnn_backbones.py` | 约 198 GB 的流式下载，不支持续传；`--inat-mirror-url` 改取只含 Plantae、可续传的镜像 |
| `coco` | `efficientdet.py`、`ssdlite.py` | train2017、val2017 与标注 |
| `coco_minitrain` | `efficientdet.py`、`ssdlite.py` | train2017 的 25k 张子集，恢复快约 5 倍；需要先有 `coco` |
| `market1501` | `youreid.py` | |

`cnn_backbones.py --dataset imagenet` 要求 `Imagenet_1k/` 下恰好 1000 类。两个 CLIP 示例只在这些图上
蒸馏，那里放任何 ImageFolder 都可以。

## 3. 运行示例

每个示例都是同样三步，各是一个子命令：

```bash
python examples/<example>.py fetch-baseline ...                  # -> outputs/models/
python examples/<example>.py prune --ratios 0.3 0.5 ...          # -> outputs/pytorch_pruned/、outputs/pruning_logs/
python examples/<example>.py int8 --pt outputs/pytorch_pruned/<stem>.pt   # -> outputs/tflite_int8_litert/
```

`fetch-baseline` 把稠密模型存成完整的 `nn.Module`。`prune` 把它剪到 `--ratios` 里的每个比例
（参数比例的小数，如 `0.5`），恢复，然后保存每一档，并附一份记录恢复过程和实际达成比例的
JSON 日志。`int8` 把一个剪枝后的 `.pt` 导出为 int8 并评测；给 `--tflite` 时，评测一个现成的
int8 或 `_edgetpu.tflite`。`--models-dir`、`--dataset-dir`、`--out-dir` 改三个根目录；
`--importance` 在 `magnitude_l1`、`magnitude_l2`（默认）、`fpgm`、`random`、`lamp` 里选一个。

**`cnn_backbones.py`** 在 ImageNet 上剪 MobileNetV2、EfficientNet-Lite0、MnasNet、SqueezeNet
或 ResNet-50（`--model`），MobileNetV2 还可以在 Flowers-102 或 iNat Plantae 上剪（`--dataset`）。
ImageNet 上预训练权重本身就是基线；另两个数据集上，`fetch-baseline` 先把它微调到该数据集
（`--epochs`，默认 20）。恢复用 PruningBench 的 ImageNet 配方，90 个 epoch；第一次跑时把
`--epochs` 调小。`--iterative` 沿各比例走一条轨迹，每一档从上一档恢复后的模型继续剪，
文件名加 `_iter`。

```bash
python examples/cnn_backbones.py fetch-baseline --model resnet50
python examples/cnn_backbones.py prune --model resnet50 --ratios 0.3 0.5 0.7 --epochs 5
```

**`efficientdet.py`** 在 COCO 上剪 EfficientDet-Lite1 或 Lite2（`--model`）。
`--coco-subset minitrain` 是快的恢复集；`val2017` 只能做冒烟测试，因为 mAP 也在它上面算。
`--ratios` 按 backbone 计算；`--scope full` 改按全模型计算，文件名加 `_full`。`--scope full`
下目标较高时实际比例可能达不到，所以以日志里的 `actual_ratio` 为准。int8 模型的输出是十个
原始检测头张量，anchor 解码和 NMS 在 CPU 上跑。

```bash
python examples/efficientdet.py fetch-baseline --model efficientdet_lite1
python examples/efficientdet.py prune --model efficientdet_lite1 --ratios 0.3 0.5 \
    --coco-subset minitrain --epochs 30
```

**`ssdlite.py`** 是 COCO 上的 SSDLite320-MobileNetV3-Large，子集、backbone 比例和 CPU 解码都与
`efficientdet.py` 相同。

**`clip_rn50.py`** 剪 CLIP RN50 图像塔，在无标注的 ImageNet 训练图上向未剪枝的塔蒸馏来恢复。
zero-shot 评测需要类别名的文本嵌入，由第四个子命令用 open_clip 一次算好：

```bash
python examples/clip_rn50.py fetch-baseline
python examples/clip_rn50.py text-embeddings --preset imagenette
python examples/clip_rn50.py prune --ratios 0.3 0.5 \
    --text-emb outputs/models/clip_rn50_text_imagenette.npz \
    --val-images-dir data/datasets/Imagenet_1k/train
```

不给 `--text-emb` 和 `--val-images-dir` 时，`prune` 只蒸馏不评测。

**`relu_clip.py`** 用同样的方式剪 relu-clip 的 EfficientNet-Lite4 学生，并跟踪它与未剪枝模型的
平均余弦；`--text-emb` 接已发布的 `.npz` 做 zero-shot。`--global-pruning` 跨层而不是在层内
给通道排序，可能把这个骨干里窄的早期层整层剪掉。它的 `int8` 给了同样的
`--text-emb` 和 `--val-images-dir` 就评测 zero-shot，否则只导出。

**`youreid.py`** 在 Market-1501 裁剪图上剪 YouReID 的 `baseline_lite`，无标签蒸馏。768 维嵌入层
默认受保护；`--embed-dim` 小于 768 时先剪这一层，把预算让给 backbone，此时需要
`--distill-loss simmat`。`--extra-image-dirs` 加入无标注的裁剪图，比如来自部署相机的。
`int8` 评测 mAP 需要 torchreid（第 1 节）。

## 4. 导出路径

所有示例都用默认路径导出：`litert-torch` 配 `ai-edge-quantizer`，per-channel。另两条路径以库函数提供：

```python
int8_pruning.export_int8(model, calib, "model_int8.tflite", backend="onnx2tf")     # 在 ./onnx2tf-env 里
int8_pruning.export_int8(model, calib, "model_int8.tflite", backend="onnx2tf", quant="per-tensor")
int8_pruning.export_onnx(model, "model.onnx", (1, 3, 224, 224))                    # fp32，不需要校准集
```

两条 int8 路径对同一张图的下降方式不同，所以文件字节数不同，Edge TPU 能映射的部分也不同；
测量结果只在同一路径内比较。fp32 ONNX 路径不量化，供读取 ONNX 的运行时（TensorRT、OpenVINO、
Core ML、RKNN 等）使用，需要 `onnx` extra（`[all]` 里已含）。

## 5. Edge TPU（可选）

第 1 至 4 节既不需要 Coral 硬件，也不需要 `edgetpu_compiler`。把 int8 `.tflite` 编译成 Edge TPU
产物需要这个编译器，从 Coral 的 apt 源装一次即可（见 [coral.ai](https://coral.ai/docs/edgetpu/compiler/)）：

```bash
sudo apt-get install edgetpu-compiler        # 先添加 Coral 的 apt 源
python examples/cnn_backbones.py int8 --dataset flowers102 --edgetpu \
    --pt outputs/pytorch_pruned/mobilenetv2_flowers102_pruned50pct_magnitude_l2.pt
# -> outputs/edgetpu/<stem>_int8_edgetpu.tflite，以及编译器日志
```

每个示例的 `int8` 都支持 `--edgetpu`：把 int8 模型编译到 `<out-dir>/edgetpu/`，并通过 libedgetpu
delegate 在接上的 Coral 上评测编译后的模型。
`--tflite <stem>_int8_edgetpu.tflite` 评测已编译好的模型。在脚本里：

```python
int8_pruning.compile_edgetpu("a_int8.tflite", "out/", show_ops=True)       # 同时写出算子映射
int8_pruning.compile_edgetpu(["a_int8.tflite", "b_int8.tflite"], "out/")    # 一起编译，共享片上缓存
```

`extra_args` 把参数透传给编译器，比如 `("-m", "13")` 固定运行时版本。

`int8_pruning.latency(path)` 测编译后模型的延迟（15 次预热、100 次计时），需要 `libedgetpu`，以及
`tflite_runtime` 或 `ai_edge_litert` 之一。[`evaluation/_evaluation.py`](../src/int8_pruning/evaluation/_evaluation.py)
自成一体，所以也能在一个自带运行时与 `libedgetpu` 的 docker 镜像里按文件路径运行，Coral USB
透传进容器；`docker run` 命令写在它的文件头里。
