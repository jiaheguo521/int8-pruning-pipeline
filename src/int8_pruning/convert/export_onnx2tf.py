#!/usr/bin/env python3
"""The second export path: `.pt` -> ONNX -> int8 `.tflite` via onnx2tf.

    .pt (full model pickle)
    -> .onnx                    torch.onnx.export, opset from ModelFamily
    -> int8 .tflite             onnx2tf.convert(output_integer_quantized_tflite=True)

This is the path every artifact in this repo dated before **2026-08-22** was
built with. It was removed on that date and is restored here as an explicit,
opt-in alternative, because comparing an old artifact against a new one is
otherwise impossible: `results/tflite_size_audit.json` carries 77 onnx2tf rows
against 63 litert rows and `packaging_frac` is not comparable across the two.

`int8_pruning.convert.export_litert` is the default, and should stay the default. Through
onnx2tf, EfficientDet mapped only **83 of 479 operators** to the Edge TPU --
onnx2tf lowered effdet's BiFPN fusion to a rank-5 CONCAT/SUM and timm's
SAME-pool to PADV2, and neither is mappable, so the TPU subgraph stopped at the
backbone. Through litert-torch the same network maps **305 of 305**, every rung,
nothing on the CPU. litert-torch is the recommended path; this one stays for the
artifacts it built.

Dependencies (`onnx`, `onnx2tf`, `onnxsim`, `tensorflow`) are NOT in the
`convert` extra and are imported lazily, so having them absent costs the litert
path nothing:

    pip install -e '.[onnx2tf]'

Do not rename this module to `onnx2tf.py`. `tflite.py` is invoked BY PATH
(`python src/int8_pruning/convert/tflite.py`, as scripts/convert.sh does), which puts
`src/int8_pruning/convert/` at `sys.path[0]`; a sibling named `onnx2tf.py` then shadows
the real package and the `import onnx2tf` below resolves to itself. Absolute
imports do not help -- the shadowing module IS the absolute one.
"""
import os
from pathlib import Path

import numpy as np
import torch

from int8_pruning.convert.common import (
    FlattenOutputs, ModelFamily, load_pt_module, probe_detection_levels,
    wrap_ssd_if_needed,
)


def _require_onnx2tf():
    """Import onnx2tf with an actionable message when the extra is not installed."""
    try:
        import onnx2tf  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "the onnx2tf export path needs the optional dependencies:\n"
            "    pip install -e '.[onnx2tf]'\n"
            "They are deliberately not in the default `convert` extra -- they "
            "pull tensorflow and onnx, which the litert path does not need. "
            "Use --export-path litert (the default) if you do not need to "
            "reproduce a pre-2026-08-22 artifact."
        ) from e
    return onnx2tf


# ONNX export

def export_onnx(pt_path: Path, onnx_path: Path, family: ModelFamily) -> str:
    """Load .pt full pickle, export to ONNX. Returns the ONNX input name."""
    print(f"  [onnx] loading {pt_path.name}")
    model = load_pt_module(pt_path)

    input_h = family.input_size
    input_w = family.input_w or family.input_size
    dummy = torch.randn(1, family.input_channels, input_h, input_w)
    input_name = "input"

    if family.detection:
        # No SE / BiFPN / SAME-pool rewrites here: those are litert-torch's lowering problems, and
        # applying them would change what this path produces, the one thing it exists to keep stable.
        model, _ = wrap_ssd_if_needed(model)
        # Name each head tensor. EfficientDet returns (class_out[0..L-1], box_out[0..L-1]); keep that
        # ordering so the post-processor can map outputs back to FPN levels.
        n_levels = probe_detection_levels(model, dummy, family.name)
        output_names = ([f"cls_{i}" for i in range(n_levels)]
                        + [f"box_{i}" for i in range(n_levels)])
        model = FlattenOutputs(model)
        print(f"  [onnx] detection model: {2 * n_levels} head outputs "
              f"({n_levels} levels x cls/box)")
    else:
        output_names = ["output"]

    print(f"  [onnx] exporting opset {family.onnx_opset} (legacy, dynamo=False) "
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

    # Optional simplify; failure is benign. Skipped for detection: a multi-output graph is more
    # fragile under onnxsim's output pruning, and constant folding is not worth a dropped head.
    if not family.detection:
        try:
            import onnx
            import onnxsim
            m = onnx.load(str(onnx_path))
            m_simp, ok = onnxsim.simplify(m)
            if ok:
                onnx.save(m_simp, str(onnx_path))
                print(f"  [onnx] simplified OK")
        except Exception as e:
            print(f"  [onnx] simplify skipped: {type(e).__name__}: {e}")

    return input_name


# onnx2tf

def _ensure_onnx2tf_test_data():
    """Workaround for onnx2tf 1.29.x + numpy >= 2 incompatibility.

    onnx2tf.utils.common_functions.download_test_image_data() fetches an
    older-numpy-pickled .npy from GitHub releases (used internally to
    compare ONNX vs TF float outputs for sanity, NOT for our int8
    calibration). numpy >= 2 refuses to load it (`allow_pickle=False`
    default), so the whole convert() call aborts before reaching our
    calibration data. We pre-place a clean synthetic file in cwd so
    onnx2tf's `os.path.isfile(LOCAL_FILE_PATH)` check finds it and skips
    the download. Random data is fine — it's only used for the float
    equivalence check, which we don't depend on.
    """
    name = "calibration_image_sample_data_20x128x128x3_float32.npy"
    path = Path(os.getcwd()) / name
    if path.exists():
        return
    rng = np.random.RandomState(0)
    fake = rng.rand(20, 128, 128, 3).astype(np.float32)
    np.save(path, fake)
    print(f"  [workaround] wrote dummy {name} to cwd (numpy 2.x compat)")


def run_onnx2tf(onnx_path: Path, calib_npy: Path, input_name: str,
                tmp_dir: Path, quant_type: str = "per-channel") -> Path:
    """Run onnx2tf with int8 output. Returns the `_full_integer_quant.tflite` path.

    `quant_type` defaults to per-channel, which is both onnx2tf's own default and
    what the DSD 2026 protocol specifies. The retired `imagenet_backbones` and
    `ssdlite` artifacts were built per-TENSOR (it was the ModelFamily default
    until 2026-08-22); pass --quant-type per-tensor to reproduce those, and read
    docs/PRUNING_HAZARDS.md section 4 before quoting anything measured that way.
    """
    onnx2tf = _require_onnx2tf()

    _ensure_onnx2tf_test_data()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [onnx2tf] converting -> int8 ({quant_type}, calib={calib_npy.name})")
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
        # Fall back to any int8 variant; otherwise list what we got.
        candidates = sorted(tmp_dir.glob("*_integer_quant.tflite"))
    if not candidates:
        existing = sorted(p.name for p in tmp_dir.glob("*.tflite"))
        raise RuntimeError(
            f"onnx2tf produced no *_full_integer_quant.tflite in {tmp_dir}. "
            f"Found: {existing or '(no .tflite at all)'}"
        )
    return candidates[0]
