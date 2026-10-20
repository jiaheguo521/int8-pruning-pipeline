#!/usr/bin/env python3
"""The second export path: nn.Module -> ONNX -> int8 `.tflite` via onnx2tf.

    nn.Module
    -> .onnx                    torch.onnx.export
    -> int8 .tflite             onnx2tf.convert(output_integer_quantized_tflite=True)

Its dependencies are the `onnx2tf` extra, imported lazily so the litert path runs without
them.

Do not rename this module to `onnx2tf.py`. Running a file in this directory by path
puts `src/int8_pruning/export/` at `sys.path[0]`; a sibling named `onnx2tf.py` then shadows
the real package and the `import onnx2tf` below resolves to itself, absolute import or not.
"""
import logging
import os
from pathlib import Path

import numpy as np
import torch

from int8_pruning.export._outputs import FlattenOutputs, head_names

logger = logging.getLogger(__name__)


def onnx_graph(model, onnx_path: Path, input_shape, opset: int) -> str:
    """Export `model` (eval mode) to ONNX for onnx2tf. Returns the ONNX input name.

    Nested outputs are flattened and named by `head_names`.
    """
    dummy = torch.randn(*input_shape)
    input_name = "input"
    with torch.no_grad():
        probe = model(dummy)
    output_names = head_names(probe)
    nested = isinstance(probe, (list, tuple))
    if nested:
        model = FlattenOutputs(model)
        logger.info(f"  [onnx] {len(output_names)} outputs: {' '.join(output_names)}")

    logger.info(f"  [onnx] exporting opset {opset} (legacy, dynamo=False) "
                f"shape={tuple(input_shape)}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=[input_name], output_names=output_names,
        dynamic_axes=None,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # Optional simplify; a failure only skips it. Multi-output graphs skip it, because
    # onnxsim's output pruning can drop a head.
    if not nested:
        try:
            # Imported here: onnxsim is optional, and the simplify is skipped without it.
            import onnx
            import onnxsim
            m = onnx.load(str(onnx_path))
            m_simp, ok = onnxsim.simplify(m)
            if ok:
                onnx.save(m_simp, str(onnx_path))
                logger.info("  [onnx] simplified OK")
        except Exception as e:
            logger.warning(f"  [onnx] simplify skipped: {type(e).__name__}: {e}")

    return input_name


def run_onnx2tf(onnx_path: Path, calib_npy: Path, input_name: str,
                tmp_dir: Path, quant_type: str = "per-channel") -> Path:
    """Run onnx2tf with int8 output. Returns the `_full_integer_quant.tflite` path.

    `quant_type` defaults to per-channel, onnx2tf's own default; "per-tensor" is
    selectable, and its accuracy is not comparable with a per-channel file's.
    """
    onnx2tf = _require_onnx2tf()

    _ensure_onnx2tf_test_data()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  [onnx2tf] converting -> int8 ({quant_type}, calib={calib_npy.name})")
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(tmp_dir),
        output_integer_quantized_tflite=True,
        quant_type=quant_type,
        custom_input_op_name_np_data_path=[
            [input_name, str(calib_npy),
             [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ],
        batch_size=1,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )

    # Find the *_full_integer_quant.tflite output.
    candidates = sorted(tmp_dir.glob("*_full_integer_quant.tflite"))
    if not candidates:
        # Fall back to any int8 variant; otherwise list what was produced.
        candidates = sorted(tmp_dir.glob("*_integer_quant.tflite"))
    if not candidates:
        existing = sorted(p.name for p in tmp_dir.glob("*.tflite"))
        raise RuntimeError(
            f"onnx2tf produced no *_full_integer_quant.tflite in {tmp_dir}. "
            f"Found: {existing or '(no .tflite at all)'}"
        )
    return candidates[0]


def _require_onnx2tf():
    """Import onnx2tf with an actionable message when the extra is not installed."""
    try:
        import onnx2tf  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "the onnx2tf export path needs the optional dependencies:\n"
            "    pip install -e '.[onnx2tf]'\n"
            "They are a separate extra from `convert` because they pull tensorflow "
            "and onnx. Use backend=\"litert\" (the default) unless you need this "
            "path's lowering."
        ) from e
    return onnx2tf


def _ensure_onnx2tf_test_data():
    """Place onnx2tf's float-check sample in cwd, so onnx2tf 1.29.x runs under numpy >= 2.

    onnx2tf.utils.common_functions.download_test_image_data() fetches a pickled .npy from
    GitHub releases, used only to compare ONNX and TF float outputs. numpy >= 2 refuses to
    load it (`allow_pickle=False`) and convert() aborts. A synthetic file in cwd satisfies
    onnx2tf's `os.path.isfile(LOCAL_FILE_PATH)` check and skips the download; random data
    serves, since the file feeds only that float check.
    """
    name = "calibration_image_sample_data_20x128x128x3_float32.npy"
    path = Path(os.getcwd()) / name
    if path.exists():
        return
    rng = np.random.RandomState(0)
    fake = rng.rand(20, 128, 128, 3).astype(np.float32)
    np.save(path, fake)
    logger.info(f"  [workaround] wrote dummy {name} to cwd (numpy 2.x compat)")
