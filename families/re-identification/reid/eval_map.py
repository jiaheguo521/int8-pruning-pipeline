#!/usr/bin/env python3
"""Market-1501 mAP / CMC rank-k for the reid_* families.

The real Re-ID accuracy metric, for ranking a pruning sweep.

Why this exists next to eval_discrim.py:
`eval_discrim.py` is a QUANTIZATION SANITY GATE: ~60 same-person
and ~60 different-person cosine pairs, pass/fail on a separation threshold. It
answers "did int8 destroy the embedding?" — it has no query/gallery protocol, no
ranking, no AP, so it CANNOT order a pruning sweep by accuracy. This worker does
the standard Market-1501 evaluation instead.

Protocol (torchreid's, which reproduces published numbers):
  query   = query/*.jpg              (3368 imgs, pid > 0)
  gallery = bounding_box_test/*.jpg  (19732 imgs; pid == 0 distractors KEPT,
                                      pid == -1 junk dropped)
  cosine distance, then torchreid.metrics.evaluate_rank, which removes
  same-pid-same-camera gallery hits per the official protocol.

Metric + ranking come from torchreid (its rank_cylib Cython ext is built in
pruning-env) — we do NOT reimplement AP.

Backends (either or both; give both to get the fp32-vs-int8 accuracy delta):
  --pt     <*.pt>              fp32 torch (batched; GPU if available)
  --tflite <*_int8.tflite>     int8 on CPU (tflite_runtime / tf.lite)
  --tflite <*_int8_edgetpu.tflite>  int8 on the Coral (pycoral), if present

Preprocessing is imported from eval_discrim so fp32 and int8 go
through the byte-identical path the int8 model was calibrated for.

Usage:
    python families/re-identification/reid/eval_map.py \
        --pt outputs/models/reid_youtu_lite.pt \
        --market-dir data/datasets/market1501 \
        --out-json outputs/pruning_logs/reid_youtu_lite_map.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from eval_discrim import parse_pid_cam, preprocess
from int8_pruning.runtime.interpreter import embed_tflite, make_interpreter

RANKS = (1, 5, 10)


def collect_split(market_dir, subdir):
    """-> (paths, pids, camids) for `subdir` under the Market root.

    Mirrors torchreid's process_dir: pid == -1 (junk) is dropped, pid == 0
    (background distractors) is KEPT — dropping the distractors would inflate mAP.
    """
    dirs = sorted(Path(market_dir).glob(f"**/{subdir}"))
    if not dirs:
        raise FileNotFoundError(
            f"No {subdir}/ under {market_dir}. Expected "
            f"<market-dir>/**/Market-1501-v15.09.15/{subdir}/*.jpg")
    paths, pids, camids = [], [], []
    for p in sorted(dirs[0].glob("*.jpg")):
        pc = parse_pid_cam(p)
        if pc is None:
            continue
        pid, cam = pc
        if pid == -1:
            continue
        paths.append(p)
        pids.append(pid)
        camids.append(cam)
    return paths, np.asarray(pids, np.int64), np.asarray(camids, np.int64)


class _CropDataset:
    """Lazily preprocessed crops. Streaming, because holding query+gallery as
    float32 at once would be ~9 GB."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch
        hwc = preprocess(Image.open(self.paths[i]))
        return torch.from_numpy(hwc).permute(2, 0, 1).float()


def embed_pt(src, paths, batch_size, device=None):
    """fp32 torch, batched -> [N, D] float32."""
    import torch

    dev = torch.device(device) if device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(src, map_location="cpu", weights_only=False).eval().to(dev)
    loader = torch.utils.data.DataLoader(
        _CropDataset(paths), batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=(dev.type == "cuda"))
    out = []
    with torch.no_grad():
        for batch in loader:
            emb = model(batch.to(dev, non_blocking=True))
            out.append(emb.flatten(1).float().cpu())
    return torch.cat(out).numpy()


def embed_tf(src, paths):
    """int8 tflite -> [N, D] float32. Batch-1: that is how Phase 2 exports."""
    it = make_interpreter(src)
    out = None
    t0 = time.time()
    for i, p in enumerate(paths):
        emb = embed_tflite(it, preprocess(Image.open(p)))
        if out is None:
            out = np.zeros((len(paths), emb.size), np.float32)
        out[i] = emb
        if (i + 1) % 2000 == 0:
            print(f"      {i+1}/{len(paths)}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def score(qf, gf, q_pids, g_pids, q_camids, g_camids):
    """-> (mAP, {rank: acc}) via torchreid (cosine distance)."""
    import torch
    from torchreid.metrics.distance import compute_distance_matrix
    from torchreid.metrics.rank import evaluate_rank

    distmat = compute_distance_matrix(torch.from_numpy(qf), torch.from_numpy(gf),
                                      metric="cosine").numpy()
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=max(RANKS))
    return float(mAP) * 100.0, {r: float(cmc[r - 1]) * 100.0 for r in RANKS}


def run(args):
    q_paths, q_pids, q_camids = collect_split(args.market_dir, "query")
    g_paths, g_pids, g_camids = collect_split(args.market_dir, "bounding_box_test")
    print(f"[reid-map] query={len(q_paths)}  gallery={len(g_paths)}  "
          f"ids(query)={len(set(q_pids.tolist()))}")

    backends = []
    if args.pt:
        backends.append(("pt", args.pt))
    if args.tflite:
        backends.append(("tflite", args.tflite))
    if not backends:
        raise SystemExit("give at least one of --pt / --tflite")

    report = {"n_query": len(q_paths), "n_gallery": len(g_paths), "backends": {}}
    for backend, src in backends:
        print(f"\n[{backend}] {Path(src).name}")
        t0 = time.time()
        if backend == "pt":
            qf = embed_pt(src, q_paths, args.batch_size, args.device)
            gf = embed_pt(src, g_paths, args.batch_size, args.device)
        else:
            print("    embedding query ...", flush=True)
            qf = embed_tf(src, q_paths)
            print("    embedding gallery ...", flush=True)
            gf = embed_tf(src, g_paths)
        mAP, cmc = score(qf, gf, q_pids, g_pids, q_camids, g_camids)
        dt = time.time() - t0
        report["backends"][Path(src).name] = {
            "backend": backend, "embed_dim": int(qf.shape[1]),
            "mAP": mAP, **{f"rank{r}": cmc[r] for r in RANKS},
            "duration_s": dt,
        }
        print(f"    embed_dim = {qf.shape[1]}")
        print(f"    mAP     = {mAP:.2f}%")
        for r in RANKS:
            print(f"    rank-{r:<2d} = {cmc[r]:.2f}%")
        print(f"    ({dt:.0f}s)")

    if len(report["backends"]) == 2:
        (_, a), (_, b) = report["backends"].items()
        print(f"\n[delta] int8 - fp32:  mAP {b['mAP']-a['mAP']:+.2f}  "
              f"rank-1 {b['rank1']-a['rank1']:+.2f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"\n[reid-map] wrote {args.out_json}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pt", help="fp32 torch model (.pt, full nn.Module)")
    p.add_argument("--tflite", help="int8 tflite (CPU) or *_edgetpu.tflite (Coral)")
    p.add_argument("--market-dir", required=True,
                   help="Market-1501 root (contains **/query, **/bounding_box_test)")
    p.add_argument("--batch-size", type=int, default=64, help="--pt path only")
    p.add_argument("--device", default=None, help="--pt path only (default: cuda if available)")
    p.add_argument("--out-json", default=None)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
