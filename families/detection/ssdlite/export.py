#!/usr/bin/env python3
"""Graph wrapper for torchvision SSDLite320-MobileNetV3-Large.

The full torchvision SSD module is not a plain tensor->tensor graph: its
forward runs GeneralizedRCNNTransform (resize + 0.5/0.5 normalize) on a list
of images and, in eval mode, ends in data-dependent postprocessing (decode +
NMS). Neither torch_pruning's dependency graph nor torch.onnx.export can
trace that. This wrapper exposes the traceable core — feature extractor +
per-level head convolutions — and is shared by:

  - families/detection/ssdlite/prune.py : builds the MetaPruner over the wrapper
    so the dependency graph sees the head convs consume backbone features
    (head convs go in ignored_layers -> their interface channels stay fixed).
  - src/int8_pruning/convert/tflite.py : wraps the loaded SSD before ONNX
    export; the (cls_list, box_list) return mirrors effdet's convention so
    the existing multi-output detection export path (cls_i/box_i naming)
    works unchanged.

The wrapper shares module objects with the wrapped SSD, so pruning through it
mutates the SSD in place; the saved .pt is always the full torchvision SSD
module (unpickling needs only torchvision).
"""
import torch
import torch.nn as nn


class SSDLitePerLevelHeads(nn.Module):
    """forward(x [N,3,320,320], already 0.5/0.5-normalized) ->
    (cls_list, box_list): 6 + 6 raw NCHW head-conv outputs
    (cls [N,546,s,s], box [N,24,s,s], s = 20,10,5,3,2,1 for COCO-91).
    Per-level decode order (H,W,anchor-major reshape + concat) is reproduced
    on CPU by families/detection/ssdlite/eval.py."""

    def __init__(self, ssd):
        super().__init__()
        self.backbone = ssd.backbone  # SSDLiteFeatureExtractorMobileNet
        self.cls_heads = ssd.head.classification_head.module_list
        self.box_heads = ssd.head.regression_head.module_list

    def forward(self, x):
        feats = list(self.backbone(x).values())
        cls = [m(f) for m, f in zip(self.cls_heads, feats)]
        box = [m(f) for m, f in zip(self.box_heads, feats)]
        return cls, box


class _HardSigmoidDecomposed(nn.Module):
    """relu6(x + 3) / 6 spelled out as ADD + RELU6 + MUL. The RELU6 ends up
    folded into the ADD's fused activation on the int8 path — fully
    TPU-mappable."""

    def forward(self, x):
        return torch.clamp(x + 3.0, 0.0, 6.0) * (1.0 / 6.0)


class _HardSwishDecomposed(nn.Module):
    """x * relu6(x + 3) / 6 rewritten as x * clamp(0.5x + 1.5, 0, 3) / 3, with the
    clamp itself spelled out of RELUs.

    Mathematically identical (clamp((x+3)/2, 0, 3) == relu6(x+3)/2). Two separate
    converter hazards shape this spelling:

    1. The naive ADD/RELU6/MUL(1/6) form — and even a MIN/MAX(0,6) form — gets
       canonicalized and pattern-matched by the TFLite MLIR converter right back
       into HARD_SWISH, which edgetpu_compiler cannot map (a single HARD_SWISH at
       the stem pushes the whole graph onto the CPU; verified: 2/159 ops mapped).
       The (0,3) range has no RELU6 lowering, so the matcher — which needs relu6
       and a 1/6 constant — never fires.

    2. The upper clamp must not be a MINIMUM. ai-edge-quantizer leaves MINIMUM in
       float, so each site comes out as DEQUANTIZE -> MINIMUM -> QUANTIZE and cuts
       the Edge TPU subgraph: mobilenetv3-large has 20 hardswish sites, and with
       torch.minimum the whole SSDLite mapped 4 of 305 operators. Writing
       min(t, 3) as t - relu(t - 3) keeps everything in int8 — RELU folds into the
       preceding op's fused activation and SUB is a mapped operator.

    So: y = relu(t) - relu(relu(t) - 3) with t = 0.5x + 1.5, then x * y / 3."""

    def __init__(self):
        super().__init__()
        self.register_buffer("three", torch.full((), 3.0))

    def forward(self, x):
        t = torch.relu(x * 0.5 + 1.5)           # lower clamp; folds into the ADD
        y = t - torch.relu(t - self.three)      # upper clamp without a MINIMUM
        return x * (y * (1.0 / 3.0))


def decompose_hard_activations(module):
    """Replace nn.Hardswish / nn.Hardsigmoid with explicit ADD+RELU6+MUL
    compositions, recursively and in place.

    Why: with the native modules the converter emits HARD_SWISH (builtin 117) and
    RELU_0_TO_1 (builtin 152) in the int8 tflite. edgetpu_compiler 16.0 can't
    even parse opcode 152 ("Op builtin_code out of range"), and HARD_SWISH is
    not an Edge TPU op anyway — MobileNetV3's stem uses it, so the whole graph
    would fall back to CPU. The decomposition is mathematically identical and
    lowers to operators the TPU maps. First established against onnx2tf; it is
    still needed on the litert-torch path, where _HardSwishDecomposed also has to
    avoid MINIMUM (see its docstring).

    Export-path only: never call this in the pruning worker — the saved .pt
    must keep stock torchvision modules so unpickling needs only torchvision.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(module, name, _HardSwishDecomposed())
        elif isinstance(child, nn.Hardsigmoid):
            setattr(module, name, _HardSigmoidDecomposed())
        else:
            decompose_hard_activations(child)
    return module


def is_torchvision_ssd(model) -> bool:
    try:
        from torchvision.models.detection.ssd import SSD
    except ImportError:
        return False
    return isinstance(model, SSD)


def maybe_wrap_ssd(model):
    """Wrap a torchvision SSD for export/tracing; pass anything else through.

    Also decomposes Hardswish/Hardsigmoid (see decompose_hard_activations) —
    safe here because the converter calls this on a freshly-loaded throwaway
    copy that is exported to ONNX and discarded, never re-pickled."""
    if not is_torchvision_ssd(model):
        return model
    return SSDLitePerLevelHeads(decompose_hard_activations(model))
