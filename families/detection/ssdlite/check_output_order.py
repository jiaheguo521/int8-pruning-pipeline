#!/usr/bin/env python3
"""Do the int8 tflite's 12 head tensors correspond, one for one, to the model's?

    python families/detection/ssdlite/check_output_order.py <int8.tflite> [<checkpoint.pt>]

This is a CORRESPONDENCE gate, not an accuracy measurement. It answers the one
question eval.py cannot answer for itself: it groups the 12 outputs by channel
count and sorts by feature-map size rather than trusting the exporter's output
order, and nothing checked that the tensors it groups are the tensors it thinks.

What it checks:
  1. every output shape is unique among the 12, so the shape-based grouping is
     forced rather than merely plausible;
  2. each PyTorch head tensor's best-correlating tflite tensor is its own;
  3. eval.py's grouping recovers torchvision's level order (20,10,5,3,2,1);
  4. the concatenation reaches torchvision's [1,3234,91] / [1,3234,4].

Run 2026-08-22 on ssdlite_mobilenetv3_coco-val2017_pruned0pct_magnitude_l2: all
four hold. Head fidelity against float32 came out at cos 0.92-0.97 (class) and
0.96-0.97 (box) over four calibration frames, with the argmax class agreeing on
99.66-100% of the 3234 anchors. The same script on the retired onnx2tf build
gives 0.82-0.95 and 0.94-0.96 -- worse on every frame, which is the direction
per-channel weights predict.
"""
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "families" / "detection" / "ssdlite"))

from int8_pruning.convert.common import enable_full_module_unpickling, FlattenOutputs
from int8_pruning.convert.export_litert import _patch_squeeze_excitation_pool
from ai_edge_litert.interpreter import Interpreter
from export import maybe_wrap_ssd

NUM_CLASSES, NUM_ANCHORS = 91, 6
DEFAULT_PT = (_ROOT / "outputs/pytorch_pruned"
              / "ssdlite_mobilenetv3_coco-val2017_pruned0pct_magnitude_l2.pt")
CALIB = (_ROOT / "outputs/tflite_int8_litert/calib_cache"
         / "ssdlite_mobilenetv3_coco_n100_seed42.npy")


def _pack_torch(tensors, k):
    """torchvision's packing: [N, A*K, H, W] -> [N, H*W*A, K]."""
    out = []
    for t in tensors:
        n, _, h, w = t.shape
        out.append(t.view(n, -1, k, h, w).permute(0, 3, 4, 1, 2).reshape(n, -1, k))
    return torch.cat(out, dim=1)


def _pack_tflite(arrays, k):
    return torch.cat([torch.from_numpy(np.ascontiguousarray(a)).reshape(1, -1, k)
                      for a in arrays], dim=1)


def main(tflite_path, pt_path=DEFAULT_PT, n_frames=4):
    enable_full_module_unpickling()
    _patch_squeeze_excitation_pool()
    model = FlattenOutputs(maybe_wrap_ssd(
        torch.load(str(pt_path), map_location="cpu", weights_only=False).eval()
    ).eval()).eval()

    calib = np.load(str(CALIB))[:n_frames]
    it = Interpreter(model_path=str(tflite_path))
    it.allocate_tensors()
    inp, outs = it.get_input_details()[0], it.get_output_details()

    def run_tflite(frame_nhwc):
        scale, zp = inp["quantization"]
        info = np.iinfo(inp["dtype"])
        q = np.clip(np.round(frame_nhwc / scale + zp), info.min, info.max)
        it.set_tensor(inp["index"], q.astype(inp["dtype"]))
        it.invoke()
        got = []
        for o in outs:
            a = it.get_tensor(o["index"])
            s, z = o["quantization"]
            got.append(((a.astype(np.float32) - z) * s) if s else a.astype(np.float32))
        return got

    frame = calib[0:1]
    tfl = run_tflite(frame)
    with torch.no_grad():
        raw = model(torch.from_numpy(frame.transpose(0, 3, 1, 2).copy()))
    ref = [t.numpy().transpose(0, 2, 3, 1) for t in raw]          # -> NHWC
    ok = True

    # 1. shapes unique -> the grouping is forced, not merely plausible
    shapes = [a.shape for a in tfl]
    unique = len(set(shapes)) == len(shapes)
    print(f"[1] all 12 output shapes unique: {unique}")
    ok &= unique

    # 2. each reference tensor's best match is its own
    def corr(a, b):
        if a.shape != b.shape:
            return -2.0
        a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d else -2.0

    pairs = [int(np.argmax([corr(ref[i], tfl[j]) for j in range(12)])) for i in range(12)]
    identity = pairs == list(range(12))
    print(f"[2] identity pairing torch[i] <-> tflite[i]: {identity}  ({pairs})")
    ok &= identity

    # 3./4. eval.py's own grouping, verbatim, then the concatenation
    cls_ch = NUM_ANCHORS * NUM_CLASSES
    cls = sorted([a for a in tfl if a.shape[-1] == cls_ch],
                 key=lambda a: a.shape[1], reverse=True)
    box = sorted([a for a in tfl if a.shape[-1] != cls_ch],
                 key=lambda a: a.shape[1], reverse=True)
    order_ok = ([a.shape[1] for a in cls] == [t.shape[2] for t in raw[:6]]
                and [a.shape[1] for a in box] == [t.shape[2] for t in raw[6:]])
    print(f"[3] eval.py grouping recovers torchvision's level order: {order_ok}  "
          f"(cls {[a.shape[1] for a in cls]})")
    ok &= order_ok

    tc, tb = _pack_tflite(cls, NUM_CLASSES), _pack_tflite(box, 4)
    pc, pb = _pack_torch(raw[:6], NUM_CLASSES), _pack_torch(raw[6:], 4)
    shape_ok = tc.shape == pc.shape and tb.shape == pb.shape
    print(f"[4] concatenated shapes match torchvision: {shape_ok}  "
          f"cls {tuple(tc.shape)} box {tuple(tb.shape)}")
    ok &= shape_ok

    # fidelity, reported but not asserted -- this is a correspondence gate
    print(f"\n    {'frame':>5} {'cls cos':>9} {'box cos':>9} {'argmax agree':>13}")
    for i in range(len(calib)):
        t = run_tflite(calib[i:i + 1])
        with torch.no_grad():
            r = model(torch.from_numpy(calib[i:i + 1].transpose(0, 3, 1, 2).copy()))
        c = sorted([a for a in t if a.shape[-1] == cls_ch],
                   key=lambda a: a.shape[1], reverse=True)
        b = sorted([a for a in t if a.shape[-1] != cls_ch],
                   key=lambda a: a.shape[1], reverse=True)
        tc, tb = _pack_tflite(c, NUM_CLASSES), _pack_tflite(b, 4)
        pc, pb = _pack_torch(r[:6], NUM_CLASSES), _pack_torch(r[6:], 4)
        cos = lambda x, y: float(torch.nn.functional.cosine_similarity(
            x.reshape(1, -1), y.reshape(1, -1)).item())
        agree = float((tc.argmax(-1) == pc.argmax(-1)).float().mean())
        print(f"    {i:>5} {cos(tc, pc):>9.5f} {cos(tb, pb):>9.5f} {100 * agree:>12.2f}%")

    print(f"\n{'PASS' if ok else 'FAIL'}: output correspondence")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1]),
                  Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PT))
