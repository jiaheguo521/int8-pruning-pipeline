"""fp32 ONNX through torch.onnx.export's legacy exporter (dynamo=False).

The call is kept separate from the one in `_onnx2tf.py`, so this path can change without
changing what onnx2tf receives. torch raises on a missing `onnx` only from inside the
exporter, hence `_require_onnx`.
"""
import logging
from pathlib import Path
from typing import NamedTuple

import torch

from int8_pruning.export._outputs import FlattenOutputs, head_names

__all__ = ["export_onnx", "ExportOnnxResult"]

logger = logging.getLogger(__name__)


class ExportOnnxResult(NamedTuple):
    """What `export_onnx` recorded.

    Attributes
    ----------
    onnx_opset, ir_version, n_nodes, size_bytes : int
    precision, input_name : str
    output_names : list of str
    input_shape : list of int
    size_mib : float
    """
    onnx_opset: int
    precision: str
    input_name: str
    output_names: list
    input_shape: list
    size_bytes: int
    size_mib: float
    ir_version: int
    n_nodes: int


def export_onnx(model, out_path, input_shape, opset=18):
    """Write `model` as fp32 ONNX taking `input_shape` (NCHW).

    For TensorRT, OpenVINO and other ONNX toolchains; quantize with their own tools. Nested
    outputs are named as on the onnx2tf path: ``cls_i`` / ``box_i`` for a detector's pair
    of lists, ``output_i`` otherwise.

    Parameters
    ----------
    model : torch.nn.Module
    out_path : str or pathlib.Path
    input_shape : sequence of int
    opset : int, default 18

    Returns
    -------
    ExportOnnxResult

    Raises
    ------
    ImportError
        If `onnx` is not installed.
    """
    onnx = _require_onnx()
    model.eval()
    onnx_path, input_shape = Path(out_path), tuple(input_shape)
    dummy = torch.randn(*input_shape)
    input_name = "input"
    with torch.no_grad():
        probe = model(dummy)
    output_names = head_names(probe)
    if isinstance(probe, (list, tuple)):
        # torch 2.10 flattens the nested tuple itself, but can collapse it into a single output;
        # the wrapper keeps every level of a detector's pyramid.
        model = FlattenOutputs(model)
        logger.info(f"  [onnx] {len(output_names)} outputs: {' '.join(output_names)}")

    logger.info(f"  [onnx] exporting opset {opset} shape={tuple(input_shape)}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=[input_name], output_names=output_names,
        dynamic_axes=None,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    result = ExportOnnxResult(
        onnx_opset=opset,
        precision="fp32",
        input_name=input_name,
        output_names=output_names,
        input_shape=list(input_shape),
        size_bytes=onnx_path.stat().st_size,
        size_mib=round(onnx_path.stat().st_size / (1024 * 1024), 4),
        ir_version=graph.ir_version,
        n_nodes=len(graph.graph.node),
    )
    logger.info(f"  [onnx] checker OK, {result.n_nodes} nodes")
    return result


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
