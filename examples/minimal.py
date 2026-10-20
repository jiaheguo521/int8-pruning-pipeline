"""The whole toolchain on a small CNN and random data, in seconds on a CPU.

    python examples/minimal.py [out_dir]

Prune half the parameters, recover, export to int8, run the int8 model once. A real
model swaps in its own network, data loaders and calibration images; the calls stay
the same. Compiling for the Edge TPU is one more call, which needs edgetpu_compiler:

    int8_pruning.compile_edgetpu(out_dir / "minimal_int8.tflite")
"""
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import int8_pruning


def main(out_dir):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    images, labels = torch.randn(256, 3, 32, 32), torch.randint(0, 10, (256,))
    train = DataLoader(TensorDataset(images, labels), batch_size=32, shuffle=True)
    dense = int8_pruning.count_params(model)

    # Structured pruning. The classifier keeps its 10 outputs.
    model.eval()
    info = int8_pruning.prune(model, torch.randn(1, 3, 32, 32), 0.5,
                              ignored_layers=[model[-1]])
    print(f"pruned {dense:,} -> {info.params:,} parameters ({info.actual_ratio:.0%})")

    # Recovery: fine-tune on the labels, with an optimizer built after pruning.
    # int8_pruning.distill(model, teacher, loader, epochs, optimizer=...) recovers against
    # the unpruned model instead, when there are no labels.
    int8_pruning.finetune(model, train, 2,
                          optimizer=torch.optim.SGD(model.parameters(), lr=0.01))

    # Export: int8 TFLite, calibrated on images normalized the way the model expects.
    # int8_pruning.export.image_folder(root, n, size, mean, std) builds this from a
    # directory.
    calib = images[:32].permute(0, 2, 3, 1).numpy()
    out = int8_pruning.export_int8(model.eval(), calib,
                                   Path(out_dir) / "minimal_int8.tflite")
    print(f"int8 model: {out.path} ({out.size_bytes:,} bytes, "
          f"input {out.input_dtype} {out.input_shape})")

    # Evaluation: run the int8 model on one image; its outputs come back dequantized.
    it = int8_pruning.make_interpreter(out.path)
    logits = int8_pruning.run_tflite(it, calib[0])[0]
    print(f"int8 output {logits.shape}, class {int(np.argmax(logits))}")
    return out


if __name__ == "__main__":
    # The library reports progress through logging; this shows it.
    logging.basicConfig(format="%(message)s")
    logging.getLogger("int8_pruning").setLevel(logging.INFO)
    main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
