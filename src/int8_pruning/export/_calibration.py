"""Calibration sets for `export_int8`: (N, H, W, 3) float32, normalized for the model."""
import random
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "int8_pruning.export needs Pillow, from the optional dependencies:\n"
        "    pip install -e '.[convert]'"
    ) from e

__all__ = ["image_folder", "normalize_nhwc"]


def image_folder(root, num, size, mean, std, seed=42, crop=True):
    """Load `num` random images from anywhere under `root` as a calibration set.

    Parameters
    ----------
    root : str or pathlib.Path
        Searched recursively for .jpg, .jpeg and .png files.
    num : int
        Images to draw; all of them when fewer are found.
    size : int or (int, int)
        Square side, or (H, W).
    mean, std : float or sequence of float
        Per-channel, on the [0, 1] scale.
    seed : int, default 42
        Seeds the draw; the set is deterministic given (num, seed).
    crop : bool, default True
        ImageNet eval transform (resize to 256/224, center-crop); False stretches to size.

    Returns
    -------
    numpy.ndarray
        (num, H, W, 3) float32, the `calib` of `export_int8`.

    Raises
    ------
    FileNotFoundError
        If `root` holds no .jpg, .jpeg or .png file.
    """
    h, w = (size, size) if isinstance(size, int) else size
    files = sorted(p for p in Path(root).rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise FileNotFoundError(f"no .jpg / .jpeg / .png under {root}")
    picks = random.Random(seed).sample(files, k=min(num, len(files)))
    target = int(round(max(h, w) * 256 / 224))
    out = np.zeros((len(picks), h, w, 3), dtype=np.uint8)
    for i, p in enumerate(picks):
        with Image.open(p) as img:
            img = img.convert("RGB")
            if crop:
                iw, ih = img.size
                scale = target / min(iw, ih)
                nw, nh = max(w, int(round(iw * scale))), max(h, int(round(ih * scale)))
                img = img.resize((nw, nh), Image.BILINEAR)
                left, top = (nw - w) // 2, (nh - h) // 2
                img = img.crop((left, top, left + w, top + h))
            else:
                img = img.resize((w, h), Image.BILINEAR)
            out[i] = np.asarray(img, dtype=np.uint8)
    return normalize_nhwc(out, mean, std)


def normalize_nhwc(arr_uint8, mean, std):
    """(N, H, W, 3) uint8 -> float32, scaled to [0, 1] and normalized by per-channel `mean`
    and `std`."""
    x = arr_uint8.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return x.astype(np.float32)
