"""int8_pruning.convert.export_onnx

The third export path: `.pt` -> fp32 `.onnx`, and nothing after it.

    .pt (full-module pickle)
      -> fp32 .onnx           torch.onnx.export, legacy exporter (dynamo=False)

This path does NOT quantize. It exists because the two int8 paths both end in a
TFLite flatbuffer, which is a terminal format: TFLite is what the mobile and
Coral toolchains consume, and almost nothing else reads it. ONNX is the
interchange format the other vendor toolchains take as input -- TensorRT,
OpenVINO, Core ML via coremltools, RKNN, Hailo, QNN. Exporting fp32 ONNX is what
lets a reader of this repository take a pruned checkpoint somewhere this
repository never measured.

Read the scope twice, because the file it writes cannot state it:

  * **fp32, not int8.** Quantizing ONNX is a different recipe (ONNX QDQ, its own
    calibration, its own numerics) and the result would NOT be comparable to the
    int8 TFLite artifacts this repository measured. Nothing here claims it would
    be. If you want int8 on another runtime, quantize on that runtime's own
    toolchain from this graph.
  * **Not measured.** No latency, no accuracy and no operator-mapping number in
    `results/` was produced through this path. It is an exit, not a result.

Why this file duplicates the `torch.onnx.export` call in `export_onnx2tf.py`
instead of sharing it: that module reproduces artifacts built before
2026-08-22, and the one property it has to keep is that its output does not move.
Sharing a helper would mean every change made for this path is also a change to
that one. The duplication is the cheaper of the two risks, and it is deliberate.

`onnx` is a hard requirement, not an optional checker: torch 2.10's exporter
raises `OnnxExporterError: Module onnx is not installed!` before writing anything,
with `dynamo=False` as much as with the dynamo exporter. It is its own extra
(`pip install -e '.[onnx]'`) rather than part of `convert`, because the litert
path genuinely does not need it and the point of that path is that it pulls
neither onnx nor tensorflow.
"""
from pathlib import Path

import torch


def _require_onnx():
    """Import onnx with an actionable message when the extra is not installed.

    torch.onnx.export raises `Module onnx is not installed!` on its own, but from
    inside the exporter and without saying which extra provides it.
    """
    try:
        import onnx
    except ImportError as e:
        raise ImportError(
            "the fp32 ONNX export path needs the optional dependency:\n"
            "    pip install -e '.[onnx]'\n"
            "It is not in the default `convert` extra because the litert path "
            "does not build an ONNX graph at all."
        ) from e
    return onnx


from int8_pruning.convert.common import (
    FlattenOutputs, ModelFamily, load_pt_module, probe_detection_levels,
    wrap_ssd_if_needed,
)


def export_fp32_onnx(pt_path: Path, onnx_path: Path, family: ModelFamily) -> dict:
    """Load a `.pt` full pickle, write fp32 ONNX. Returns sidecar fields."""
    onnx = _require_onnx()
    print(f"  [onnx] loading {pt_path.name}")
    model = load_pt_module(pt_path)

    input_h = family.input_size
    input_w = family.input_w or family.input_size
    dummy = torch.randn(1, family.input_channels, input_h, input_w)
    input_name = "input"
    output_names = ["output"]

    if family.detection:
        # Same as the onnx2tf path: flatten the per-level head tensors and name them, so a consumer
        # can map an output back to its FPN level. No SE / BiFPN / SAME-pool rewrites -- those exist
        # for litert-torch's lowering and the Edge TPU compiler, neither of which is downstream here.
        model, _ = wrap_ssd_if_needed(model)
        n_levels = probe_detection_levels(model, dummy, family.name)
        # Same names as export_onnx2tf, so the two graphs can be diffed.
        output_names = ([f"cls_{i}" for i in range(n_levels)]
                        + [f"box_{i}" for i in range(n_levels)])
        # Not redundant even though torch 2.10 happens to flatten the nested tuple on its own: that
        # collapsed once before, leaving the EfficientDet graph with a single [1,3,3,36] output
        # instead of the ten-tensor pyramid. Do not rely on the exporter's mood.
        model = FlattenOutputs(model)
        print(f"  [onnx] detection model: {2 * n_levels} head outputs "
              f"({n_levels} levels x cls/box)")

    print(f"  [onnx] exporting opset {family.onnx_opset} "
          f"shape=(1,{family.input_channels},{input_h},{input_w})")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=[input_name], output_names=output_names,
        dynamic_axes=None,
        opset_version=family.onnx_opset,
        do_constant_folding=True,
        dynamo=False,
    )

    info = {
        "onnx_opset": family.onnx_opset,
        "precision": "fp32",
        "input_name": input_name,
        "output_names": output_names,
        "input_shape": [1, family.input_channels, input_h, input_w],
        "size_bytes": onnx_path.stat().st_size,
        "size_mib": round(onnx_path.stat().st_size / (1024 * 1024), 4),
    }

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    info["ir_version"] = graph.ir_version
    info["n_nodes"] = len(graph.graph.node)
    print(f"  [onnx] checker OK, {info['n_nodes']} nodes")

    return info
