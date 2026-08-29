#!/usr/bin/env python3
"""Fetch the published relu-clip student and write it as this repo's baseline pickle, `outputs/models/relu_clip.pt`.

The other families' baselines come from `scripts/download_and_finetune_models.sh`,
which builds them out of timm / torchvision. This one does not fit there: the
weights are a Hugging Face checkpoint (`jiaheguo521/relu-clip`) whose state dict
only loads into the module defined in `families/classification/relu_clip/student.py`, so the
build step has to import that module -- and, because `torch.save` pickles the
class by import name, has to import it as bare `student` so the converter's
`UNPICKLE_MODULE_DIRS` entry resolves it later.

    python families/classification/relu_clip/download_baseline.py

Downloads ~51 MB. Idempotent: skips if outputs/models/relu_clip.pt exists.

What is published (README of the source repo, all measured on a Coral USB Edge
TPU): efflite4 distilled from CLIP ViT-L/14, 12.71 M params, 68.35 fp32 / 68.25
int8 ImageNet-1k zero-shot top-1 over the full 50 000-image val split, 13.58 MiB
int8, 7.60 MiB on-chip + 7.50 MiB off-chip, 26.49 ms/frame. Those are the numbers
a pruning ladder from this baseline is measured against; none of them are
re-measured here.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from student import (  # noqa: E402  (sibling module, needs the path insert above)
    DEFAULT_BACKBONE, DEFAULT_EMBED_DIM, INPUT_SIZE, build_student,
)

HF_REPO = "jiaheguo521/relu-clip"
HF_WEIGHTS = "efflite4-vitl14/student.safetensors"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=None,
                   help="output .pt (default: outputs/models/relu_clip.pt)")
    p.add_argument("--repo", default=HF_REPO)
    p.add_argument("--weights", default=HF_WEIGHTS)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    out = args.out or (Path(__file__).parents[3] / "outputs" / "models" / "relu_clip.pt")
    if out.exists() and not args.force:
        print(f"[skip] {out} exists (--force to rebuild)")
        return 0

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    print(f"Downloading {args.repo}/{args.weights} ...")
    ckpt = hf_hub_download(repo_id=args.repo, filename=args.weights)
    state = load_file(ckpt)

    model = build_student(DEFAULT_BACKBONE, DEFAULT_EMBED_DIM, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Shape gate: a mis-shaped head survives every later stage and shows up only as collapsed zero-shot.
    with torch.no_grad():
        emb = model(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE))
    if tuple(emb.shape) != (1, DEFAULT_EMBED_DIM):
        raise RuntimeError(f"expected [1, {DEFAULT_EMBED_DIM}] embedding, got "
                           f"{list(emb.shape)}")
    n_params = sum(x.numel() for x in model.parameters())

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, str(out))
    print(f"  -> {out}  (full nn.Module, {n_params:,} params, "
          f"{DEFAULT_EMBED_DIM}-d embedding)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
