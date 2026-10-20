"""The default export path: nn.Module -> float `.tflite` -> per-channel int8.

    nn.Module
    -> float .tflite            litert_torch.convert, straight from the module
    -> int8 .tflite             ai-edge-quantizer, static_wi8_ai8 (per-channel),
                                calibrated with pre-normalized NHWC samples
    -> buffers inlined          _edgetpu_fixups.inline_buffers
    -> Edge TPU fixups          _edgetpu_fixups.etpu_fixups

Graph patches are litert-specific: onnx2tf lowers the constructs they rewrite differently.
"""
import contextlib
import logging
from pathlib import Path

import numpy as np
import torch

from int8_pruning.export._outputs import FlattenOutputs

logger = logging.getLogger(__name__)


def float_tflite(model, f32_path: Path, input_shape) -> dict:
    """Export `model` (eval mode) to a float .tflite with NHWC input, via litert-torch.

    `input_shape` is the NCHW shape the model takes. A forward that returns nested
    lists or tuples (a detector's per-level heads) is flattened, so every leaf tensor
    becomes an output of its own. Returns {"n_outputs", "channel_last_outputs"}.
    """
    # Imported here, as in the rest of this module: the `convert` extra is optional, and
    # pruning and recovery run without it.
    litert_torch = _require_convert()

    _, c, input_h, input_w = input_shape
    with torch.no_grad():
        probe = model(torch.randn(*input_shape))
    if isinstance(probe, (list, tuple)):
        model = FlattenOutputs(model)
        with torch.no_grad():
            probe = model(torch.randn(*input_shape))
    outs = list(probe) if isinstance(probe, (list, tuple)) else [probe]
    n_out = len(outs)
    # to_channel_last_io can only transpose rank>=3 outputs. Detection heads and segmentation
    # maps are spatial; classification logits and Re-ID / CLIP embeddings are 2-D and stay put.
    spatial = all(getattr(o, "ndim", 0) >= 3 for o in outs)

    logger.info(f"  [float] litert-torch convert shape=(1,{input_h},{input_w},"
                f"{c}) outputs={n_out} channel_last={spatial}")
    f32_path.parent.mkdir(parents=True, exist_ok=True)
    net = litert_torch.to_channel_last_io(
        model, args=[0], outputs=list(range(n_out)) if spatial else None).eval()
    edge = litert_torch.convert(net, (torch.randn(1, input_h, input_w, c),))
    edge.export(str(f32_path))
    logger.info(f"  [float] {f32_path.name} {f32_path.stat().st_size / 2**20:.2f} MiB")
    return {"n_outputs": n_out, "channel_last_outputs": spatial}


def quantize_int8(f32_path: Path, calib: np.ndarray, int8_path: Path,
                  work_dir: Path) -> dict:
    """Float .tflite -> int8, then inline buffers and apply the Edge TPU fixups.

    Weights are per-channel (ai-edge-quantizer's `static_wi8_ai8` default,
    CHANNELWISE), the only granularity on this path: under per-tensor, direction-sensitive
    cosine embeddings collapse and detection loses mAP. The onnx2tf path offers per-tensor.
    """
    # Imported here, like litert_torch above: _int8.py imports this module for every export
    # path, and an env that prunes only, or exports through onnx2tf, lacks `convert`.
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import Quantizer, recipe

    from int8_pruning.export._edgetpu_fixups import inline_buffers, etpu_fixups

    sigs = Interpreter(model_path=str(f32_path)).get_signature_list()
    sig = list(sigs)[0]
    in_name = sigs[sig]["inputs"][0]

    data = {sig: [{in_name: calib[i: i + 1]} for i in range(len(calib))]}

    logger.info(f"  [int8] ai-edge-quantizer static_wi8_ai8 (per-channel), calib={len(calib)}")
    with _true_minmax():
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


def _require_convert():
    """Import litert_torch with an actionable message when the extra is not installed.

    The quantizer and the runtime are checked here too, so that a missing one fails before
    the float conversion rather than after it.
    """
    try:
        import ai_edge_litert  # noqa: F401
        import ai_edge_quantizer  # noqa: F401
        import litert_torch
    except ImportError as e:
        raise ImportError(
            f"the litert export path (the default) needs the optional dependencies; "
            f"{e.name} is missing:\n"
            "    pip install -e '.[convert]'"
        ) from e
    return litert_torch


@contextlib.contextmanager
def _true_minmax():
    """Replace ai-edge-quantizer's smoothed activation ranges with the true min/max.

    ai-edge-quantizer moving-averages every tensor's min/max across calibration
    batches (qsv_utils.moving_average_update, smoothing_factor=0.95), so every
    activation range comes out narrower than the data really is. TFLite's own
    converter takes the true min/max, which is worth real mAP on a detector. The patch
    applies to every export and is undone on exit.

    The live call site is calibrator.py -> algorithm_manager.get_update_qsv_func(),
    so the patch covers it as well as the qsv_utils module attribute.
    """
    def _minmax_update(qsv, new_qsv, *a, **kw):
        if not qsv:
            return new_qsv
        return {"min": np.minimum(qsv["min"], new_qsv["min"]),
                "max": np.maximum(qsv["max"], new_qsv["max"])}

    from ai_edge_quantizer import algorithm_manager as _am
    import ai_edge_quantizer.calibrator as _c
    from ai_edge_quantizer.utils import qsv_utils as _q

    undo = []
    try:
        for owner, attribute, value in (
                (_am, "get_update_qsv_func", lambda *a, **kw: _minmax_update),
                (_c.algorithm_manager, "get_update_qsv_func", lambda *a, **kw: _minmax_update),
                (_q, "moving_average_update", _minmax_update)):
            undo.append((owner, attribute, getattr(owner, attribute)))
            setattr(owner, attribute, value)
        yield
    finally:
        for owner, attribute, original in reversed(undo):
            setattr(owner, attribute, original)
