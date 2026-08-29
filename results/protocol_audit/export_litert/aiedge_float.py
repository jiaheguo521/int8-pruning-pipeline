"""litert-torch float export of a pruned EfficientDet .pt (no PT2E).

  python aiedge_float.py [--patch-fpn] [--pad-zero] [--nhwc-out] out.tflite

--patch-fpn  effdet's FpnCombine does torch.stack(nodes,-1).sum(-1); rewrite as
             sequential adds. weight_method is 'sum' for the lite configs, so
             this is arithmetically identical, and it removes the rank-5
             CONCAT/RESHAPE/SUM the Edge TPU cannot map.
--pad-zero   timm's SAME-pool pads with -inf -> PADV2 ("Operation not
             supported"). Padding with 0 -> PAD, which the TPU maps. Same
             window alignment; differs only where every real value in an edge
             window is negative.
--nhwc-out   also emit the 10 head outputs channel-last, like the onnx2tf
             pipeline does.
"""
import sys
from pathlib import Path
import torch

import os as _os
from pathlib import Path as _Path
# Repo root: four levels up from results/protocol_audit/export_litert/, or $ETPU_REPO.
R = _Path(_os.environ.get("ETPU_REPO") or _Path(__file__).resolve().parents[3])
PT = Path(next((a for a in sys.argv[1:] if a.endswith(".pt")),
               R / "outputs/pytorch_pruned/efficientdet_lite1_coco-train2017_pruned10pct_lamp_full.pt"))
out = Path([a for a in sys.argv[1:] if a.endswith(".tflite")][0])

import effdet.efficientdet as E
if "--patch-fpn" in sys.argv:
    _orig = E.FpnCombine.forward
    def _fwd(self, x):
        nodes = [rs(x[o]) for o, rs in zip(self.inputs_offsets, self.resample.values())]
        if self.weight_method == "sum":
            o = nodes[0]
            for n in nodes[1:]:
                o = o + n
            return o
        return _orig(self, x)
    E.FpnCombine.forward = _fwd

if "--pool-ceil" in sys.argv:
    # timm's SAME-pool is F.pad(-inf) + VALID pool, and litert-torch keeps that PAD in NCHW
    # wrapped in 3 transposes per site, which cuts the Edge TPU subgraph. ceil_mode=True with
    # no padding is EXACTLY the same computation for these shapes (48->24->12->6->3: the last
    # window is clamped to the input, which is what -inf padding achieves) and lowers to a
    # plain MAX_POOL_2D with SAME padding.
    import torch.nn.functional as _F
    import timm.layers.pool2d_same as _P
    def _pool_fwd(self, x):
        return _F.max_pool2d(x, self.kernel_size, self.stride, (0, 0),
                             self.dilation, ceil_mode=True)
    _P.MaxPool2dSame.forward = _pool_fwd

if "--pad-zero" in sys.argv:
    import torch.nn.functional as _F
    import timm.layers.pool2d_same as _P
    def _pool_fwd(self, x):
        x = _P.pad_same(x, self.kernel_size, self.stride, value=0.0)
        return _F.max_pool2d(x, self.kernel_size, self.stride, (0, 0),
                             self.dilation, self.ceil_mode)
    _P.MaxPool2dSame.forward = _pool_fwd

model = torch.load(str(PT), map_location="cpu", weights_only=False).eval()

class Flat(torch.nn.Module):
    """EfficientDet returns (cls_list, box_list); flatten to 10 tensors."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        cls, box = self.m(x)
        return tuple(list(cls) + list(box))

class FatOutput(torch.nn.Module):
    """Backbone + BiFPN, but one output is nearest-upsampled so the model
    returns ~2 MB instead of 0.26 MB. Compute is essentially unchanged
    (RESIZE_NEAREST_NEIGHBOR is nearly free), so any latency delta is the cost
    of moving bytes back across USB, not of computing them."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        f = self.m.fpn(self.m.backbone(x))
        big = torch.nn.functional.interpolate(f[0], scale_factor=3, mode="nearest")
        return (big,) + tuple(f[1:])

class FatConv(torch.nn.Module):
    """Backbone + BiFPN + ONE 1x1 conv 88->810 on P3, so the TPU has to return
    1.78 MiB instead of 0.26 MiB. The extra compute is 164 MMACs (~0.1 ms at
    4 TOPS), so a large latency delta can only be the cost of moving the bytes
    off the accelerator."""
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.wide = torch.nn.Conv2d(88, 810, 1)
    def forward(self, x):
        f = self.m.fpn(self.m.backbone(x))
        return (self.wide(f[0]),) + tuple(f[1:])

class FeaturesOnly(torch.nn.Module):
    """Everything up to and including the BiFPN; the 5 head-input features.
    Used to price the detection heads separately from the rest."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        return tuple(self.m.fpn(self.m.backbone(x)))

import litert_torch
size = 448 if "lite2" in PT.name else 384
feat = "--features-only" in sys.argv
fat = "--fat-output" in sys.argv
fatconv = "--fat-conv" in sys.argv
n_out = 5 if (feat or fat or fatconv) else 10
wrapper = FatConv if fatconv else (FatOutput if fat else (FeaturesOnly if feat else Flat))
net = litert_torch.to_channel_last_io(
    wrapper(model), args=[0],
    outputs=list(range(n_out)) if "--nhwc-out" in sys.argv else None).eval()
edge = litert_torch.convert(net, (torch.randn(1, size, size, 3),))
edge.export(str(out))
print(f"[float] {out.name} {out.stat().st_size/2**20:.2f} MiB  src={PT.name} size={size} "
      f"flags={[a for a in sys.argv[1:] if a.startswith('--')]}")
