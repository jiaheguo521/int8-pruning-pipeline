"""`export_int8`: a recovered model to an int8 .tflite, through litert-torch or onnx2tf."""
import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

from int8_pruning.evaluation import make_interpreter
from int8_pruning.export._litert import float_tflite, quantize_int8
from int8_pruning.export._onnx2tf import onnx_graph, run_onnx2tf
from int8_pruning.export._tflite_size import tflite_size

__all__ = ["export_int8", "ExportInt8Result"]


class ExportInt8Result(NamedTuple):
    """What `export_int8` recorded; fields that do not apply are None.

    Attributes
    ----------
    path, backend, quant_type : str
    size_bytes, file_bytes : int
    const_bytes, name_bytes : int
        Constant data and tensor names; None without the flatbuffer schema.
    const_mib, packaging_frac : float
        packaging_frac is the share of the file that is not constant data.
    num_tensors, num_operators, num_subgraphs : int
    input_dtype, output_dtype : str
    input_shape, output_shape : list of int
        Read on the CPU; None without a TFLite runtime. Output fields describe output 0.
    num_outputs : int
    outputs : list of dict
        Name, dtype and shape of each output, when there is more than one.
    num_calib : int
    n_outputs, channel_last_outputs, external_buffers_inlined, etpu_fixups, model_patches
        litert path only.
    onnx_opset : int
        onnx2tf path only.
    """
    path: str
    backend: str
    quant_type: str
    size_bytes: int
    file_bytes: int
    const_bytes: int
    name_bytes: int
    const_mib: float
    packaging_frac: float
    num_tensors: int
    num_operators: int
    num_subgraphs: int
    input_dtype: str
    input_shape: list
    output_dtype: str
    output_shape: list
    num_outputs: int
    outputs: list
    num_calib: int
    n_outputs: int
    channel_last_outputs: bool
    external_buffers_inlined: int
    etpu_fixups: dict
    model_patches: list
    onnx_opset: int


def export_int8(model, calib, out_path, *, backend="litert", quant="per-channel",
                graph_patches=(), opset=18):
    """Quantize `model` to an int8 .tflite at `out_path`, calibrated on `calib`.

    Parameters
    ----------
    model : torch.nn.Module
        Takes one NCHW image tensor; nested list/tuple outputs become separate outputs.
    calib : array_like
        (N, H, W, 3) float32, normalized for the model; sets the input size.
    out_path : str or pathlib.Path
    backend : {"litert", "onnx2tf"}, default "litert"
        litert-torch + ai-edge-quantizer, or torch.onnx.export + onnx2tf (own venv).
    quant : {"per-channel", "per-tensor"}, default "per-channel"
        "per-tensor" on the onnx2tf path only.
    graph_patches : sequence of callable, default ()
        Context-manager factories in force during the litert export, e.g. `se_mean_pool`.
    opset : int, default 18
        ONNX opset of the onnx2tf path.

    Returns
    -------
    ExportInt8Result

    Raises
    ------
    ValueError
        On an unknown `backend`, or "per-tensor" with litert.
    """
    calib = np.asarray(calib, dtype=np.float32)
    _, h, w, c = calib.shape
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with tempfile.TemporaryDirectory(dir=out_path.parent) as tmp:
        tmp = Path(tmp)
        if backend == "litert":
            if quant != "per-channel":
                raise ValueError("the litert path is per-channel only; per-tensor is onnx2tf's")
            with contextlib.ExitStack() as patches:
                for patch in graph_patches:
                    patches.enter_context(patch())
                f32_path = tmp / f"{out_path.stem}_f32.tflite"
                info = float_tflite(model, f32_path, (1, c, h, w))
            info.update(quantize_int8(f32_path, calib, out_path, tmp))
            info["model_patches"] = [p.__name__ for p in graph_patches]
        elif backend == "onnx2tf":
            onnx_path = tmp / f"{out_path.stem}.onnx"
            input_name = onnx_graph(model, onnx_path, (1, c, h, w), opset)
            calib_npy = tmp / "calib.npy"
            np.save(calib_npy, calib)
            produced = run_onnx2tf(onnx_path, calib_npy, input_name, tmp, quant)
            shutil.move(str(produced), str(out_path))
            info = {"onnx_opset": opset, "num_calib": len(calib)}
        else:
            raise ValueError(f"backend must be 'litert' or 'onnx2tf', got {backend!r}")
    record = {"path": str(out_path), "backend": backend, "quant_type": quant,
              "size_bytes": out_path.stat().st_size, **tflite_size(out_path),
              **info, **_io_summary(out_path)}
    # Unrecorded fields are None; a key with no field raises, rather than being dropped.
    return ExportInt8Result(**{**dict.fromkeys(ExportInt8Result._fields), **record})


def _io_summary(int8_path):
    """Input/output dtype and shape of an int8 file, read on the CPU; None without a runtime."""
    info = {"input_dtype": None, "output_dtype": None,
            "input_shape": None, "output_shape": None}
    try:
        it = make_interpreter(int8_path)
    except ImportError:
        return info
    i = it.get_input_details()[0]
    outs = it.get_output_details()
    info["input_dtype"] = np.dtype(i["dtype"]).name
    info["input_shape"] = list(map(int, i["shape"]))
    # The single-output fields describe the first output; `outputs` lists every one.
    o0 = outs[0]
    info["output_dtype"] = np.dtype(o0["dtype"]).name
    info["output_shape"] = list(map(int, o0["shape"]))
    info["num_outputs"] = len(outs)
    if len(outs) > 1:
        info["outputs"] = [
            {"name": o["name"],
             "dtype": np.dtype(o["dtype"]).name,
             "shape": list(map(int, o["shape"]))}
            for o in outs
        ]
    return info
