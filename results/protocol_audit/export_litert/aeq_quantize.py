"""Static int8 PTQ of a float .tflite with ai-edge-quantizer."""
import sys, os
from pathlib import Path
import numpy as np

import os as _os
from pathlib import Path as _Path
# Repo root: four levels up from results/protocol_audit/export_litert/, or $ETPU_REPO.
R = _Path(_os.environ.get("ETPU_REPO") or _Path(__file__).resolve().parents[3])
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
calib_npy = Path(os.environ.get(
    "CALIB_NPY", R / "outputs/tflite_int8/calib_cache/efficientdet_lite1_coco_n100_seed42.npy"))

from ai_edge_litert.interpreter import Interpreter
sigs = Interpreter(model_path=str(src)).get_signature_list()
sig = list(sigs)[0]
in_name = sigs[sig]["inputs"][0]

calib = np.load(str(calib_npy))[: int(os.environ.get("CALIB_N", 100))]
if os.environ.get("ANCHOR_RANGE", "0") == "1":
    # ai-edge-quantizer's min/max calibrator comes out narrower than the data: it clipped 2.6% of effdet
    # inputs and 12.1% of line_seg_w96 inputs. Two constant frames at the true bounds pin the real range.
    full = np.load(str(calib_npy))
    lo = np.full_like(calib[:1], full.min())
    hi = np.full_like(calib[:1], full.max())
    calib = np.concatenate([calib, lo, hi], axis=0)
    print(f"[aeq] anchored input range to [{full.min():.3f}, {full.max():.3f}]")
data = {sig: [{in_name: calib[i : i + 1]} for i in range(len(calib))]}

if os.environ.get("QSV_MINMAX", "0") == "1":
    # ai-edge-quantizer smooths every tensor's min/max across calibration batches
    # (qsv_utils.moving_average_update, smoothing_factor=0.95), so every activation range
    # comes out narrower than the data really is; TFLite's own converter takes the true
    # min/max, worth ~0.02 mAP on effdet. The live call site is calibrator.py:574 ->
    # algorithm_manager.get_update_qsv_func(), NOT the qsv_utils module attribute.
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
    print("[aeq] QSV update -> true min/max (patched algorithm_manager.get_update_qsv_func)")

from ai_edge_quantizer import Quantizer, recipe
from ai_edge_quantizer.algorithm_manager import AlgorithmName
algo = getattr(AlgorithmName, os.environ.get("AEQ_ALGO", "MIN_MAX_UNIFORM_QUANT"))
qt = Quantizer(str(src))
qt.load_quantization_recipe(recipe.static_wi8_ai8(algorithm_key=algo))
print(f"[aeq] algorithm={algo}")
cal = qt.calibrate(data)
if os.environ.get("FIX_INPUT_QSV", "0") == "1":
    # Same smoothing on the INPUT tensor (calibrator.py: qsv_utils.moving_average_update):
    # 2.6% of effdet inputs and 11% of line_seg_w96 inputs were being clipped. The input
    # range is the one range we know exactly, so pin it.
    full = np.load(str(calib_npy))
    lo, hi = float(full.min()), float(full.max())
    from ai_edge_litert.interpreter import Interpreter as _I
    it2 = _I(model_path=str(src)); it2.allocate_tensors()
    in_name = it2.get_input_details()[0]["name"]
    hit = [k for k in cal if k == in_name or k.endswith("/" + in_name)]
    for k in hit or [k for k in cal if "arg" in k.lower() or "input" in k.lower()]:
        before = dict(cal[k])
        for f, v in (("min", lo), ("max", hi)):
            if f in cal[k]:
                cal[k][f] = np.array(v, dtype=np.float32).reshape(np.shape(cal[k][f]))
        print(f"[aeq] pinned {k}: {before} -> {cal[k]}")
res = qt.quantize(cal)
res.export_model(str(dst))
print(f"[int8] {dst.name} {dst.stat().st_size/2**20:.2f} MiB  calib={len(calib)}")
