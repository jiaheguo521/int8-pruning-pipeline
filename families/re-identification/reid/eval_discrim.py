#!/usr/bin/env python3
"""Quantization sanity gate for the reid_* embedders: same- vs different-person cosine separation.

Sanity-check that int8 quantization did NOT destroy a Re-ID embedder's
discriminative power. Re-ID matches by COSINE similarity, so a bad int8
calibration (the classic PINTO_model_zoo failure) can leave same-person and
different-person pairs indistinguishable even when the model "looks" converted.

For each backend it embeds a small identity-labeled sample from Market-1501 and
reports:
  * mean cosine of SAME-person pairs (same identity, different camera)  -> high
  * mean cosine of DIFFERENT-person pairs                               -> low
  * separation margin = same - diff                                     -> big
When BOTH --pt (fp32) and --tflite (int8) are given it also reports the
per-image direction-preservation gate  mean cos(fp32_emb, int8_emb)  (the CLIP
de-risk number; ~>0.8 means quantization kept the embedding direction).

Backends (either or both):
  --pt     <reid_*_market1501.pt>          fp32 reference (torch)
  --tflite <..._int8.tflite>               int8 on CPU (tflite_runtime / tf.lite)
  --tflite <..._int8_edgetpu.tflite>       int8 on the Coral (pycoral), if present

Reuses make_interpreter / quantize_input / _input_quant from the classifier eval
worker (same path the int8 model was calibrated for), so int8-vs-fp32 numbers
are directly comparable.

Usage:
    python families/re-identification/reid/eval_discrim.py \
        --pt outputs/models/reid_osnet_x0_5_market1501.pt \
        --tflite outputs/tflite_int8_litert/reid_osnet_x0_5_market1501_int8.tflite \
        --market-dir data/datasets/market1501 \
        --num-ids 60 --out-json outputs/pruning_logs/reid_osnet_x0_5_discrim.json

    # To demonstrate the per-tensor collapse, convert once with
    #   --quant-type per-tensor  and point --tflite at that file.
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

from int8_pruning.runtime.interpreter import (
    _input_quant, embed_tflite, make_interpreter, quantize_input,
)

# Torchreid's default test transform: 256x128 (H x W), ImageNet mean/std, RGB.
REID_H, REID_W = 256, 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Market-1501 filename: PID_cCAM sSEQ_FRAME_BBOX.jpg, e.g. 0002_c1s1_000451_03.jpg
_FNAME_RE = re.compile(r"^(-?\d+)_c(\d+)")


def parse_pid_cam(path):
    m = _FNAME_RE.match(Path(path).name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def preprocess(pil):
    """PIL -> normalized HWC float32 (256x128, ImageNet norm, RGB)."""
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil = pil.resize((REID_W, REID_H), Image.BILINEAR)  # PIL takes (W, H)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def collect_gallery(market_dir):
    """-> dict pid -> list of (Path, cam), from bounding_box_test (+ query)."""
    root = Path(market_dir)
    img_dirs = []
    for sub in ("bounding_box_test", "query"):
        img_dirs += sorted(root.glob(f"**/{sub}"))
    if not img_dirs:
        raise FileNotFoundError(
            f"No bounding_box_test/query under {root}. Expected "
            f"<market-dir>/**/Market-1501-v15.09.15/{{bounding_box_test,query}}/*.jpg")
    by_pid = {}
    for d in img_dirs:
        for p in sorted(d.glob("*.jpg")):
            pc = parse_pid_cam(p)
            if pc is None:
                continue
            pid, cam = pc
            if pid <= 0:          # -1 = junk, 0000 = distractor
                continue
            by_pid.setdefault(pid, []).append((p, cam))
    return by_pid


def build_pairs(by_pid, num_ids, seed):
    """Pick num_ids identities with >=2 cameras; return (images, same, diff)."""
    rng = random.Random(seed)
    usable = sorted(pid for pid, items in by_pid.items()
                    if len({c for _, c in items}) >= 2)
    rng.shuffle(usable)
    picked = usable[:num_ids]
    if len(picked) < 2:
        raise RuntimeError("need >=2 identities with multi-camera shots")

    reps = {}   # pid -> (pathA, pathB): two shots from different cameras
    for pid in picked:
        items = sorted(by_pid[pid], key=lambda t: (t[1], t[0].name))
        first = items[0]
        second = next(it for it in items if it[1] != first[1])
        reps[pid] = (first[0], second[0])

    same = [reps[pid] for pid in picked]
    # different-person: each picked pid's A-shot vs the next pid's A-shot
    diff = [(reps[a][0], reps[b][0])
            for a, b in zip(picked, picked[1:] + picked[:1]) if a != b]

    images, seen = [], set()
    for a, b in same:
        for p in (a, b):
            if p not in seen:
                seen.add(p)
                images.append(p)
    return images, same, diff


def l2(v):
    return v / (np.linalg.norm(v) + 1e-12)


def embed_all(backend, src, images):
    """-> dict Path -> L2-normalized embedding."""
    embs = {}
    if backend == "tflite":
        it = make_interpreter(src)
        for p in images:
            embs[p] = l2(embed_tflite(it, preprocess(Image.open(p))))
    else:  # fp32 torch
        import torch
        model = torch.load(src, map_location="cpu", weights_only=False).eval()
        for p in images:
            x = torch.from_numpy(preprocess(Image.open(p))).permute(2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                embs[p] = l2(model(x).flatten().numpy().astype(np.float32))
    return embs


def cos_stats(embs, pairs):
    a = np.array([float(np.dot(embs[x], embs[y])) for x, y in pairs], np.float32)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


def run(args):
    by_pid = collect_gallery(args.market_dir)
    images, same, diff = build_pairs(by_pid, args.num_ids, args.seed)
    print(f"[reid-discrim] identities={len(same)}  images={len(images)}  "
          f"same-pairs={len(same)}  diff-pairs={len(diff)}")

    backends = []
    if args.pt:
        backends.append(("pt", args.pt))
    if args.tflite:
        backends.append(("tflite", args.tflite))
    if not backends:
        raise SystemExit("give at least one of --pt / --tflite")

    report = {"num_ids": len(same), "n_images": len(images), "backends": {}}
    emb_by = {}
    for backend, src in backends:
        embs = embed_all(backend, src, images)
        emb_by[backend] = embs
        s, d = cos_stats(embs, same), cos_stats(embs, diff)
        sep = s["mean"] - d["mean"]
        report["backends"][Path(src).name] = {
            "backend": backend, "same": s, "diff": d, "separation": sep}
        print(f"\n[{backend}] {Path(src).name}")
        print(f"    same-person cos : mean={s['mean']:.4f}  std={s['std']:.4f}")
        print(f"    diff-person cos : mean={d['mean']:.4f}  std={d['std']:.4f}")
        print(f"    separation      : {sep:.4f}  "
              f"({'OK' if sep > 0.1 else 'WEAK — check calibration'})")

    # Direction-preservation gate: fp32 vs int8 on identical images.
    if "pt" in emb_by and "tflite" in emb_by:
        fp32, int8 = emb_by["pt"], emb_by["tflite"]
        cw = np.array([float(np.dot(fp32[p], int8[p])) for p in images], np.float32)
        report["cos_fp32_int8"] = {"mean": float(cw.mean()), "std": float(cw.std()),
                                   "min": float(cw.min())}
        print(f"\n[gate] cos(fp32_emb, int8_emb): mean={cw.mean():.4f}  min={cw.min():.4f}  "
              f"({'OK' if cw.mean() > 0.8 else 'LOW — quant shifted embedding directions'})")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"\n[reid-discrim] wrote {args.out_json}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pt", help="fp32 torch baseline (reid_*_market1501.pt)")
    p.add_argument("--tflite", help="int8 tflite (CPU) or *_edgetpu.tflite (Coral)")
    p.add_argument("--market-dir", required=True,
                   help="Market-1501 root (contains **/query, **/bounding_box_test)")
    p.add_argument("--num-ids", type=int, default=60,
                   help="identities sampled for the pairs (default 60)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", default=None)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
