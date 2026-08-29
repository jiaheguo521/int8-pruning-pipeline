"""Bigger lane-seg model ladder for the dual-model pruning experiment
(downstream repo: int8-pruning-pipeline).

SELF-CONTAINED ON PURPOSE: the downstream pruning repo loads delivered .pt files with
torch.load(weights_only=False) (full-module pickle, per CoralCLIP A4 handoff contract),
which needs the defining module importable under the same name. This file has zero
intra-repo imports and a collision-unlikely module name, so the handoff is "copy this
one file"; torchvision encoders unpickle from torchvision itself.

Op discipline (Edge TPU gate, same as models.TinySegNet): conv/BN/ReLU6, MaxPool,
nearest Resize, Concat, residual Add only. DLv3LiteMBV2 additionally tries
global-avg-pool + nearest broadcast (ASPP image pooling) -- gate-experiment only.
All variants output a single [B,1,H/2,W/2] mask-logit tensor (input stride-2), so
they slot into the existing EdgeTPUSeg / follow_lane_tpu contract via meta.json.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 255-scale normalization stats (meta.json mean255/std255 convention)
NORM_05_255 = [127.5, 127.5, 127.5]
IMAGENET_MEAN255 = [123.675, 116.28, 103.53]
IMAGENET_STD255 = [58.395, 57.12, 57.375]


class ConvBNAct(nn.Module):
    def __init__(self, ci, co, k=3, s=1):
        super().__init__()
        self.c = nn.Conv2d(ci, co, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(co)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.c(x)))


class LaneSegLadder(nn.Module):
    """models.TinySegNet scaled by width: same 8-conv encoder-decoder as the production
    345KB model (width=(16,32,64,96)), so pruning curves compare within one family.
    Redefined here (not imported) to keep this module self-contained for the handoff."""

    def __init__(self, width):
        super().__init__()
        c1, c2, c3, c4 = width
        self.e1 = ConvBNAct(3, c1, 3, 2)      # 1/2
        self.e2 = ConvBNAct(c1, c2, 3, 2)     # 1/4
        self.e3 = ConvBNAct(c2, c3, 3, 2)     # 1/8
        self.e4 = ConvBNAct(c3, c4, 3, 2)     # 1/16
        self.bott = ConvBNAct(c4, c4, 3, 1)
        self.d1 = ConvBNAct(c4, c3, 3, 1)
        self.d2 = ConvBNAct(c3, c2, 3, 1)
        self.d3 = ConvBNAct(c2, c1, 3, 1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        x = self.e4(self.e3(self.e2(self.e1(x))))
        x = self.bott(x)
        x = self.d1(F.interpolate(x, scale_factor=2, mode="nearest"))
        x = self.d2(F.interpolate(x, scale_factor=2, mode="nearest"))
        x = self.d3(F.interpolate(x, scale_factor=2, mode="nearest"))
        return self.head(x)                    # [B,1,H/2,W/2]


class _MBV2Enc(nn.Module):
    """torchvision MobileNetV2 features split at strides 2/4/8/16/32 (drops the final
    1x1->1280 conv). All ReLU6+BN, TPU-safe."""
    chs = (16, 24, 32, 96, 320)

    def __init__(self, pretrained):
        super().__init__()
        from torchvision.models import mobilenet_v2
        f = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None).features
        self.s2, self.s4, self.s8 = f[0:2], f[2:4], f[4:7]
        self.s16, self.s32 = f[7:14], f[14:18]

    def forward(self, x):
        s2 = self.s2(x); s4 = self.s4(s2); s8 = self.s8(s4)
        s16 = self.s16(s8); s32 = self.s32(s16)
        return s2, s4, s8, s16, s32


class _ResNetEnc(nn.Module):
    """torchvision resnet18/34 as a stride-2/4/8/16/32 feature pyramid. ReLU+BN+MaxPool."""

    def __init__(self, arch, pretrained):
        super().__init__()
        import torchvision.models as tvm
        m = getattr(tvm, arch)(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)
        self.pool = m.maxpool
        self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4
        e = 4 if arch in ("resnet50", "resnet101") else 1
        self.chs = (64, 64 * e, 128 * e, 256 * e, 512 * e)

    def forward(self, x):
        s2 = self.stem(x)
        s4 = self.l1(self.pool(s2))
        s8 = self.l2(s4); s16 = self.l3(s8); s32 = self.l4(s16)
        return s2, s4, s8, s16, s32


class DecBlock(nn.Module):
    """Upsample -> 1x1-project the deep tensor to EXACTLY the skip's channel count ->
    concat -> 2x conv. Equal-channel concat keeps onnx2tf on the clean NHWC path;
    channel-unequal concats get wrapped in NCHW TRANSPOSE sandwiches that break the
    Edge TPU single-subgraph gate (seen on the first ladder gate run)."""

    def __init__(self, c_deep, c_skip, co):
        super().__init__()
        self.pre = ConvBNAct(c_deep, c_skip, 1)
        self.c1 = ConvBNAct(2 * c_skip, co)
        self.c2 = ConvBNAct(co, co)

    def forward(self, x, skip):
        x = self.pre(F.interpolate(x, scale_factor=2, mode="nearest"))
        return self.c2(self.c1(torch.cat([x, skip], 1)))


class LinkDecBlock(nn.Module):
    """LinkNet-style additive skip: upsample -> conv, + 1x1-projected skip -> conv.
    Add has no concat-axis ambiguity at all -- fallback if equal-channel concat
    still fails the gate."""

    def __init__(self, c_deep, c_skip, co):
        super().__init__()
        self.c1 = ConvBNAct(c_deep, co)
        self.sp = ConvBNAct(c_skip, co, 1)
        self.c2 = ConvBNAct(co, co)

    def forward(self, x, skip):
        x = self.c1(F.interpolate(x, scale_factor=2, mode="nearest"))
        return self.c2(x + self.sp(skip))


class UNetTV(nn.Module):
    """UNet on a torchvision ImageNet encoder, nearest-upsample decoder, head at
    stride 2 -> [B,1,H/2,W/2] logits. skip='cat' (UNet) or 'add' (LinkNet)."""

    def __init__(self, encoder="resnet18", dec=(256, 128, 64, 32), pretrained=True, skip="cat"):
        super().__init__()
        self.enc = _MBV2Enc(pretrained) if encoder == "mobilenet_v2" else _ResNetEnc(encoder, pretrained)
        c = self.enc.chs
        blk = DecBlock if skip == "cat" else LinkDecBlock
        deeps = (c[4], dec[0], dec[1], dec[2])
        skips = (c[3], c[2], c[1], c[0])
        self.blocks = nn.ModuleList(blk(cd, cs, co) for cd, cs, co in zip(deeps, skips, dec))
        self.head = nn.Conv2d(dec[-1], 1, 1)

    def forward(self, x):
        s2, s4, s8, s16, s32 = self.enc(x)
        x = self.blocks[0](s32, s16)
        x = self.blocks[1](x, s8)
        x = self.blocks[2](x, s4)
        x = self.blocks[3](x, s2)
        return self.head(x)                    # [B,1,H/2,W/2]


class DLv3LiteMBV2(nn.Module):
    """DeepLab-flavored gate experiment: MBV2 encoder + ASPP-lite at stride 32
    (1x1 / 3x3 / stacked-3x3 / image-pool branches, no atrous), plain conv+nearest
    decoder to stride 2. The image-pool branch (global avg pool -> 1x1 -> nearest
    broadcast back to 1/32 grid) is the op under test."""

    def __init__(self, c=256, pretrained=True):
        super().__init__()
        self.enc = _MBV2Enc(pretrained)
        ce = self.enc.chs[4]
        self.b0 = ConvBNAct(ce, c, 1)
        self.b1 = ConvBNAct(ce, c, 3)
        self.b2 = nn.Sequential(ConvBNAct(ce, c, 3), ConvBNAct(c, c, 3))
        self.pool_proj = ConvBNAct(ce, c, 1)
        self.proj = ConvBNAct(4 * c, c, 1)
        self.d1 = ConvBNAct(c, 128)
        self.d2 = ConvBNAct(128, 96)
        self.d3 = ConvBNAct(96, 64)
        self.d4 = ConvBNAct(64, 48)
        self.head = nn.Conv2d(48, 1, 1)

    def forward(self, x):
        s32 = self.enc(x)[4]
        hw = s32.shape[2:]
        p = self.pool_proj(F.adaptive_avg_pool2d(s32, 1))
        p = F.interpolate(p, size=hw, mode="nearest")
        x = self.proj(torch.cat([self.b0(s32), self.b1(s32), self.b2(s32), p], 1))
        x = self.d1(F.interpolate(x, scale_factor=2, mode="nearest"))
        x = self.d2(F.interpolate(x, scale_factor=2, mode="nearest"))
        x = self.d3(F.interpolate(x, scale_factor=2, mode="nearest"))
        x = self.d4(F.interpolate(x, scale_factor=2, mode="nearest"))
        return self.head(x)                    # [B,1,H/2,W/2]


# name -> build(pretrained), input/mask size, meta.json norm. int8 bytes ~= params; targets ~2/4/8/16/32 MB.
VARIANTS = {
    # Reference row: the deployed 345KB models.TinySegNet architecture, retrained through THIS
    # script/split/recipe. The deployed model's held-out numbers are optimistic (trained on all 187
    # frames, this split's val block included), so only this row is an apples-to-apples floor.
    "line_seg_base128": dict(
        build=lambda pretrained=False: LaneSegLadder((16, 32, 64, 96)),
        input=128, mask=64, mean255=NORM_05_255, std255=NORM_05_255),
    "line_seg_w48": dict(
        build=lambda pretrained=False: LaneSegLadder((48, 96, 192, 288)),
        input=128, mask=64, mean255=NORM_05_255, std255=NORM_05_255),
    "line_seg_w64": dict(
        build=lambda pretrained=False: LaneSegLadder((64, 128, 256, 384)),
        input=160, mask=80, mean255=NORM_05_255, std255=NORM_05_255),
    "line_seg_w96": dict(
        build=lambda pretrained=False: LaneSegLadder((96, 192, 384, 576)),
        input=192, mask=96, mean255=NORM_05_255, std255=NORM_05_255),
    # B family uses ADD skips (LinkNet style): concat skips fail the compile gate even with equal channels.
    "line_seg_link_mbv2": dict(
        build=lambda pretrained=False: UNetTV("mobilenet_v2", dec=(256, 160, 112, 80), pretrained=pretrained, skip="add"),
        input=192, mask=96, mean255=IMAGENET_MEAN255, std255=IMAGENET_STD255),
    "line_seg_link_r18": dict(
        build=lambda pretrained=False: UNetTV("resnet18", dec=(256, 128, 64, 32), pretrained=pretrained, skip="add"),
        input=192, mask=96, mean255=IMAGENET_MEAN255, std255=IMAGENET_STD255),
    "line_seg_link_r34": dict(
        build=lambda pretrained=False: UNetTV("resnet34", dec=(512, 256, 128, 64), pretrained=pretrained, skip="add"),
        input=192, mask=96, mean255=IMAGENET_MEAN255, std255=IMAGENET_STD255),
    "line_seg_dlv3_mbv2": dict(
        build=lambda pretrained=False: DLv3LiteMBV2(pretrained=pretrained),
        input=192, mask=96, mean255=IMAGENET_MEAN255, std255=IMAGENET_STD255),
}

# Negative result (gate run 2026-08-04): concat-skip UNets fail the Edge TPU gate.
# onnx2tf keeps decoder CONCATENATION in NCHW with TRANSPOSE sandwiches -- multi-subgraph
# split on r18/r34, "large activation tensors" compiler crash on mbv2 -- and equal-channel
# concat does not avoid it. Out of VARIANTS, so the gate and train loops skip them.
GATE_FAILED = {
    "line_seg_unet_mbv2": dict(
        build=lambda pretrained=False: UNetTV("mobilenet_v2", dec=(256, 160, 112, 80), pretrained=pretrained),
        input=192, mask=96, mean255=IMAGENET_MEAN255, std255=IMAGENET_STD255),
    "line_seg_unet_r18": dict(
        build=lambda pretrained=False: UNetTV("resnet18", dec=(256, 128, 64, 32), pretrained=pretrained),
        input=192, mask=96, mean255=IMAGENET_MEAN255, std255=IMAGENET_STD255),
    "line_seg_unet_r34": dict(
        build=lambda pretrained=False: UNetTV("resnet34", dec=(512, 256, 128, 64), pretrained=pretrained),
        input=192, mask=96, mean255=IMAGENET_MEAN255, std255=IMAGENET_STD255),
}
