#!/usr/bin/env python3
"""Vendored YouReID `baseline_lite` architecture (reid_youtu_lite).

Inference-path definition of Tencent YouTu's `youtu_reid_baseline_lite`
(TencentYoutuResearch/PersonReID-YouReID, `example/baseline/baseline_lite_multidataset.yaml`
-> `models/baseline.py::Baseline`), plus a loader for the released checkpoint.

Why vendored instead of pip-installing YouReID:
YouReID is a research repo with no setup.py; `models/baseline.py` starts with
`from core.loss import *`, which drags in the whole training stack. We only need
the eval-mode forward, which is ~30 lines. The torchvision ResNet-50 has
parameter names IDENTICAL to YouReID's `models/backbones/resnet.py`
(`conv1`/`bn1`/`layer1..4`, standard Bottleneck), so checkpoint keys line up 1:1.

Architecture (verified against the upstream source + the released checkpoint):
    resnet50 (last_stride=1)  ->  [B, 2048, 16, 8]   for a 256x128 input
    cat([gap(x), gmp(x)], 1)  ->  [B, 4096, 1, 1]    <- why embedding_layer is 4096-in
    embedding_layer 1x1 conv  ->  [B,  768, 1, 1]
    bn + squeeze              ->  [B,  768]
54 convs (53 resnet + embedding_layer), 26.66M params, int8 ~26.7MB.

Two upstream quirks this file deliberately flattens:

1. `split_bn: true` — the trainer runs `convert_dsbnConstBatch(batch_size=128,
   constant_batch=32)`, replacing EVERY BatchNorm2d with a 4-way `bn_list` (one
   per source dataset, 4x32=128). Its eval path uses `bn_list[0]` ONLY, so
   collapsing to `bn_list[0]` is numerically EXACT, not an approximation.
2. `Baseline.forward` returns `[y], [x]` in train mode but the bare embedding `x`
   in eval mode. Our forward ALWAYS returns the embedding, which kills the repo's
   "wrong tensor" trap by construction and makes `.train()` safe during
   distillation (BN stats update; no output-shape swap).

`fc_layer` (Linear(768, 3246) — the merged four-dataset ID head) is NOT built:
eval never uses it, and recovery here is distillation, not CrossEntropy.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50

EMBED_DIM = 768  # YouReID Baseline reduce_dim default; the released lite weights use it
POOL_CAT_DIM = 4096  # gap ++ gmp over resnet50's 2048 channels


class _ResNetTrunk(nn.Module):
    """torchvision resnet50 with YouReID's last_stride=1; forward returns the
    layer4 featuremap (no avgpool/fc). Attribute names match YouReID's
    `models/backbones/resnet.py`, so `resnet.*` checkpoint keys load 1:1."""

    def __init__(self):
        super().__init__()
        r = resnet50(weights=None)
        # last_stride=1 (Baseline default): layer4 keeps 16x8 for a 256x128 input instead of 8x4. Both the
        # 3x3 and the downsample conv carry the stride in torchvision's V1.5 layout, as YouReID's _make_layer does.
        r.layer4[0].conv2.stride = (1, 1)
        r.layer4[0].downsample[0].stride = (1, 1)
        self.conv1 = r.conv1
        self.bn1 = r.bn1
        self.relu = r.relu
        self.maxpool = r.maxpool
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        self.layer4 = r.layer4

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class YoutuReIDLite(nn.Module):
    """YouReID baseline_lite eval path. Input [B,3,256,128] -> output [B,768]."""

    def __init__(self):
        super().__init__()
        self.resnet = _ResNetTrunk()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.embedding_layer = nn.Conv2d(POOL_CAT_DIM, EMBED_DIM, kernel_size=1,
                                         stride=1, bias=False)
        self.bn = nn.Sequential(nn.BatchNorm2d(EMBED_DIM))

    def forward(self, x):
        x = self.resnet(x)
        x = torch.cat([self.gap(x), self.gmp(x)], 1)
        x = self.embedding_layer(x)
        return self.bn(x).squeeze(dim=3).squeeze(dim=2)


def load_youtu_checkpoint(model, ckpt_path):
    """Load the released `youtu_reid_baseline_lite` checkpoint into `model`.

    Rewrites three upstream artifacts, then loads **strict**: any leftover
    mismatch means our architecture drifted from the checkpoint and must explode
    rather than silently load partial weights.
      - `module.` prefix        (saved from a DataParallel/DDP wrapper)
      - `*.bn_list.{0..3}.*`    (split_bn; keep 0, which is what eval uses)
      - `fc_layer.*`            (train-only ID head; we do not build it)
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    clean = {}
    for key, val in sd.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith(("fc_layer.", "ce_loss.", "tri_loss.")):
            continue
        if ".bn_list." in key:
            if ".bn_list.0." not in key:
                continue
            key = key.replace(".bn_list.0.", ".")
        clean[key] = val

    model.load_state_dict(clean, strict=True)
    return model


def build_youtu_lite(ckpt_path):
    """Built + loaded + eval-mode YoutuReIDLite. The one entry point callers need."""
    model = YoutuReIDLite()
    load_youtu_checkpoint(model, ckpt_path)
    model.eval()
    return model
