#!/usr/bin/env python3
"""The default export path: `.pt` -> float `.tflite` -> per-channel int8.

    .pt (full model pickle)
    -> float .tflite            litert_torch.convert, straight from the nn.Module
    -> int8 .tflite             ai-edge-quantizer, static_wi8_ai8 (per-channel),
                                calibrated with pre-normalized NHWC samples
    -> buffers inlined          int8_pruning.convert.flatbuffer.inline_buffers
    -> Edge TPU fixups          int8_pruning.convert.flatbuffer.etpu_fixups

There is no ONNX step here: litert_torch takes the module directly. The
alternative path is `int8_pruning.convert.export_onnx2tf`, which is kept for reproducing
artifacts built before 2026-08-22 and is not the default — see that module and
docs/PRUNING_HAZARDS.md section 4 for what the switch bought.

The five monkeypatches below are litert-specific. They exist because
litert-torch's lowering of SE-pool, timm's SAME-pool, effdet's BiFPN sum and
AdaptiveMaxPool2d each cut the Edge TPU subgraph; onnx2tf lowered those same
constructs differently (and broke on others), so it does not want them.
"""
from pathlib import Path

import numpy as np
import torch

from int8_pruning.convert.common import (
    FlattenOutputs, ModelFamily, load_pt_module, probe_detection_levels,
    wrap_ssd_if_needed,
)


# Model rewrites for the Edge TPU mapping

def _patch_effdet_fpn() -> None:
    """effdet's FpnCombine does torch.stack(nodes, -1).sum(-1); rewrite as adds.

    `weight_method` is 'sum' for every lite config, so this is arithmetically
    identical, and it removes the rank-5 CONCAT/RESHAPE/SUM the Edge TPU cannot map.
    Without it the TPU subgraph stops at the backbone.
    """
    import effdet.efficientdet as E
    orig = E.FpnCombine.forward

    def fwd(self, x):
        nodes = [rs(x[o]) for o, rs in zip(self.inputs_offsets, self.resample.values())]
        if self.weight_method == "sum":
            out = nodes[0]
            for n in nodes[1:]:
                out = out + n
            return out
        return orig(self, x)

    E.FpnCombine.forward = fwd


def _same_pool_args(n: int, k: int, s: int):
    """(padding, ceil_mode) for a torch max_pool2d that equals TF SAME on extent n.

    TF SAME pads by `pad_total = (ceil(n/s)-1)*s + k - n`, split before/after as
    (pad_total//2, pad_total-pad_total//2). torch's max_pool2d only takes a
    symmetric padding, so the split decides which form reproduces it exactly:

        pad_total even -> symmetric already; padding=pad_total//2, ceil_mode off
        pad_total odd  -> one extra on the right, which is what ceil_mode adds;
                          padding=pad_total//2, ceil_mode on

    Both lower to a plain MAX_POOL_2D with SAME padding -- no PAD op either way.
    For effdet's k=3, s=2 the parity is just the extent's: an even extent gives
    pad_total 1 (ceil_mode), an odd one gives 2 (symmetric padding 1).

        lite1 @384 pools 48->24->12->6, all even -> ceil_mode at every site
        lite2 @448 pools 56->28->14->7 ->4; the 7 is odd -> padding=1 there

    Getting that one site wrong is what made the pre-2026-08-27 lite2 exports a
    different network (P7 3x3 instead of 4x4); see docs/PRUNING_HAZARDS.md.

    Returns None when torch cannot express the padding (it requires padding <= k/2),
    leaving that site on timm's original SAME path.
    """
    pad_total = max((-(-n // s) - 1) * s + k - n, 0)
    pad = pad_total // 2
    return None if pad * 2 > k else (pad, bool(pad_total % 2))


def _patch_timm_pool_ceil() -> None:
    """timm's SAME-pool is F.pad(-inf) + VALID pool; rewrite it as a padded pool.

    litert-torch keeps that PAD in NCHW and wraps it in three transposes per site,
    which cut the Edge TPU subgraph -- and the PAD itself is a PADV2 the compiler
    declines, because its -inf fill never equals the tensor's zero point. The
    rewrite is decided per site from the actual input extent by _same_pool_args,
    not once per model: on EfficientDet-Lite2 three of the four pool sites need
    ceil_mode and the fourth needs symmetric padding, so no single form works.
    """
    import torch.nn.functional as F
    import timm.layers.pool2d_same as P

    orig = P.MaxPool2dSame.forward

    def fwd(self, x):
        k = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size,) * 2
        s = self.stride if isinstance(self.stride, tuple) else (self.stride,) * 2
        args = [_same_pool_args(n, ki, si) for n, ki, si in zip(x.shape[-2:], k, s)]
        if any(a is None for a in args):
            return orig(self, x)
        (ph, ch), (pw, cw) = args
        if ch != cw:
            return orig(self, x)
        return F.max_pool2d(x, k, s, (ph, pw), self.dilation, ceil_mode=ch)

    P.MaxPool2dSame.forward = fwd


def _patch_squeeze_excitation_pool() -> None:
    """torchvision SqueezeExcitation pools with nn.AdaptiveAvgPool2d(1).

    litert-torch's channel-last propagation stops at that op, so every SE block
    comes out wrapped in an NHWC->NCHW->NHWC transpose pair, and edgetpu_compiler
    declines the channel-moving TRANSPOSE ("otherwise supported, but not mapped
    due to some unspecified limitation"). Each pair cuts the subgraph: SSDLite's
    five SE blocks left 161 of 285 operators on the CPU.

    torch.mean over the spatial axes is the same computation for a static input
    and stays channel-last. Only the pooling step is replaced; fc1/fc2 and both
    activations are untouched.
    """
    from torchvision.ops.misc import SqueezeExcitation

    def _scale(self, input):
        scale = input.mean(dim=(2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    SqueezeExcitation._scale = _scale


def _patch_adaptive_max_pool() -> None:
    """litert-torch has no lowering for aten.adaptive_max_pool2d.

    With output_size=1 and a static input it is exactly a max_pool over the whole
    spatial extent, which lowers to MAX_POOL_2D (verified bit-exact, max abs diff
    0.0). AdaptiveAvgPool2d lowers fine, so only the max variant needs this. Used by
    families/re-identification/reid/youreid_model.py.
    """
    import torch.nn.functional as F

    def fwd(self, x):
        if not (self.output_size == 1 or tuple(self.output_size) == (1, 1)):
            raise NotImplementedError(
                f"AdaptiveMaxPool2d(output_size={self.output_size}) has no static "
                f"equivalent here; only output_size=1 is rewritten")
        return F.max_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))

    torch.nn.AdaptiveMaxPool2d.forward = fwd


# Float export (litert-torch)

def export_float_tflite(pt_path: Path, f32_path: Path, family: ModelFamily) -> dict:
    """Load the .pt full pickle and export a float .tflite via litert-torch.

    There is no ONNX step: litert_torch.convert takes the nn.Module directly.
    Returns {"n_outputs", "channel_last_outputs", "model_patches"}.
    """
    import litert_torch

    print(f"  [float] loading {pt_path.name}")
    _patch_adaptive_max_pool()
    patches = ["adaptive_max_pool"]

    model = load_pt_module(pt_path)

    input_h = family.input_size
    input_w = family.input_w or family.input_size

    if family.detection:
        model, wrapped = wrap_ssd_if_needed(model)
        if wrapped:
            _patch_squeeze_excitation_pool()
            patches += ["ssd_per_level_heads", "se_mean_pool"]

        # BiFPN and SAME-pool are EfficientDet's rewrites; SSDLite has neither, so skip the imports.
        if "efficientdet" in family.name:
            _patch_effdet_fpn()
            patches.append("effdet_fpn_sum")
            _patch_timm_pool_ceil()
            patches.append("timm_pool_same")

        n_levels = probe_detection_levels(
            model, torch.randn(1, family.input_channels, input_h, input_w),
            family.name)
        n_out = 2 * n_levels
        model = FlattenOutputs(model)
        print(f"  [float] detection model: {n_out} head outputs "
              f"({n_levels} levels x cls/box)")
    else:
        n_out = 1

    # to_channel_last_io can only transpose rank>=3 outputs. Detection heads and segmentation
    # maps are spatial; classification logits and Re-ID / CLIP embeddings are 2-D and stay put.
    with torch.no_grad():
        probe = model(torch.randn(1, family.input_channels, input_h, input_w))
    outs = list(probe) if isinstance(probe, (list, tuple)) else [probe]
    spatial = all(getattr(o, "ndim", 0) >= 3 for o in outs)

    print(f"  [float] litert-torch convert shape=(1,{input_h},{input_w},"
          f"{family.input_channels}) outputs={n_out} channel_last={spatial}")
    f32_path.parent.mkdir(parents=True, exist_ok=True)
    net = litert_torch.to_channel_last_io(
        model, args=[0], outputs=list(range(n_out)) if spatial else None).eval()
    edge = litert_torch.convert(
        net, (torch.randn(1, input_h, input_w, family.input_channels),))
    edge.export(str(f32_path))
    print(f"  [float] {f32_path.name} {f32_path.stat().st_size / 2**20:.2f} MiB")
    return {"n_outputs": n_out, "channel_last_outputs": spatial,
            "model_patches": patches}


# int8 quantization (ai-edge-quantizer) + the two flatbuffer rewrites

def _patch_qsv_true_minmax() -> None:
    """Replace ai-edge-quantizer's smoothed activation ranges with the true min/max.

    ai-edge-quantizer moving-averages every tensor's min/max across calibration
    batches (qsv_utils.moving_average_update, smoothing_factor=0.95), so every
    activation range comes out narrower than the data really is. TFLite's own
    converter takes the true min/max, and that is worth +0.020 mAP on
    efficientdet_lite1 (0.2932 -> 0.3134). Every committed litert number in this
    repo was produced with this on, so it is the default here, not a flag.

    The live call site is calibrator.py -> algorithm_manager.get_update_qsv_func(),
    NOT the qsv_utils module attribute, so patch there.
    """
    def _minmax_update(qsv, new_qsv, *a, **kw):
        if not qsv:
            return new_qsv
        return {"min": np.minimum(qsv["min"], new_qsv["min"]),
                "max": np.maximum(qsv["max"], new_qsv["max"])}

    from ai_edge_quantizer import algorithm_manager as _am
    _am.get_update_qsv_func = lambda *a, **kw: _minmax_update
    import ai_edge_quantizer.calibrator as _c
    _c.algorithm_manager.get_update_qsv_func = lambda *a, **kw: _minmax_update
    from ai_edge_quantizer.utils import qsv_utils as _q
    _q.moving_average_update = _minmax_update


def quantize_int8(f32_path: Path, calib_npy: Path, int8_path: Path,
                  num_calib: int, work_dir: Path) -> dict:
    """Float .tflite -> int8, then inline buffers and apply the Edge TPU fixups.

    Weights are per-channel (ai-edge-quantizer's `static_wi8_ai8` default,
    CHANNELWISE), which is what the DSD 2026 protocol specifies. There is no
    per-tensor knob on this path: direction-sensitive cosine embeddings collapse
    outright under per-tensor (CLIP zero-shot 94.17% -> 18.83%) and detection
    loses 0.0218 mAP, and nothing in this repo wants that. The onnx2tf path
    exposes `--quant-type` only because reproducing its retired artifacts
    requires it. See docs/PRUNING_HAZARDS.md.
    """
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import Quantizer, recipe

    from int8_pruning.convert.flatbuffer import inline_buffers, etpu_fixups

    sigs = Interpreter(model_path=str(f32_path)).get_signature_list()
    sig = list(sigs)[0]
    in_name = sigs[sig]["inputs"][0]

    calib = np.load(str(calib_npy))[:num_calib]
    data = {sig: [{in_name: calib[i: i + 1]} for i in range(len(calib))]}

    _patch_qsv_true_minmax()
    print(f"  [int8] ai-edge-quantizer static_wi8_ai8 (per-channel), "
          f"calib={len(calib)} from {calib_npy.name}")
    qt = Quantizer(str(f32_path))
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    raw_path = work_dir / f"{int8_path.stem}_raw.tflite"
    qt.quantize(qt.calibrate(data)).export_model(str(raw_path))

    # ORDER MATTERS: inline before fixups, or every large constant is silently dropped.
    inl_path = work_dir / f"{int8_path.stem}_inl.tflite"
    moved = inline_buffers(raw_path, inl_path)
    counts = etpu_fixups(inl_path, int8_path)
    return {"external_buffers_inlined": moved, "etpu_fixups": counts,
            "num_calib": len(calib)}
