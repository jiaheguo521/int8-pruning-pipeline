"""`export_int8`, `export_onnx`, `compile_edgetpu` and `calibration.image_folder` on a small CNN.

A graph rewrite that outlives its export fails silently: every later forward, ONNX export
or int8 export in the same process runs the rewritten class, and the one that breaks is
whichever comes next. Needs torch, and litert-torch / ai-edge-quantizer or onnx for the
export tests; each skips without what it needs.
"""
import contextlib
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import int8_pruning

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
HAVE_LITERT = all(importlib.util.find_spec(m) for m in ("litert_torch", "ai_edge_quantizer"))
HAVE_ONNX = importlib.util.find_spec("onnx") is not None


class _Heads(nn.Module if torch is not None else object):
    """A detector's shape: (class_out[0..1], box_out[0..1])."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.cls = nn.ModuleList([nn.Conv2d(8, 4, 1), nn.Conv2d(8, 4, 1)])
        self.box = nn.ModuleList([nn.Conv2d(8, 4, 1), nn.Conv2d(8, 4, 1)])

    def forward(self, x):
        f = self.body(x)
        return [h(f) for h in self.cls], [h(f) for h in self.box]


def _cnn():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10)).eval()


@unittest.skipUnless(torch is not None and HAVE_LITERT, "needs torch and litert-torch")
class ExportInt8(unittest.TestCase):
    def test_int8_io_and_the_patch_is_undone(self):
        original = nn.ReLU.forward

        @contextlib.contextmanager
        def patch():
            nn.ReLU.forward = lambda self, x: torch.clamp(x, min=0)
            try:
                yield
            finally:
                nn.ReLU.forward = original

        from ai_edge_quantizer import algorithm_manager
        qsv = algorithm_manager.get_update_qsv_func
        calib = np.random.default_rng(0).standard_normal((8, 16, 16, 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            info = int8_pruning.export_int8(_cnn(), calib, Path(d) / "m_int8.tflite",
                                            graph_patches=[patch])
            self.assertTrue(Path(info.path).is_file())
        self.assertEqual(info.input_dtype, "int8")
        self.assertEqual(info.input_shape, [1, 16, 16, 3])
        self.assertEqual(info.model_patches, ["patch"])
        self.assertIs(nn.ReLU.forward, original)
        # The quantizer's own calibration hook is put back too.
        self.assertIs(algorithm_manager.get_update_qsv_func, qsv)

    def test_run_tflite_returns_every_output_dequantized(self):
        calib = np.random.default_rng(0).standard_normal((8, 16, 16, 3)).astype(np.float32)
        torch.manual_seed(0)
        model = _Heads().eval()
        with tempfile.TemporaryDirectory() as d:
            info = int8_pruning.export_int8(model, calib, Path(d) / "heads_int8.tflite")
            it = int8_pruning.make_interpreter(info.path)
            outs = int8_pruning.run_tflite(it, calib[0])
        self.assertEqual(len(outs), 4)
        self.assertTrue(all(o.dtype == np.float32 and o.shape == (1, 16, 16, 4) for o in outs))
        with torch.no_grad():
            ref = model(torch.from_numpy(calib[:1]).permute(0, 3, 1, 2))
        for got, want in zip(outs, [*ref[0], *ref[1]]):
            # int8 against fp32: close, not equal
            self.assertLess(np.abs(got - want.permute(0, 2, 3, 1).numpy()).max(), 0.1)


@unittest.skipUnless(torch is not None and HAVE_LITERT and shutil.which("edgetpu_compiler"),
                     "needs torch, litert-torch and edgetpu_compiler")
class CompileEdgetpu(unittest.TestCase):
    def test_one_file_and_a_co_compiled_pair(self):
        calib = np.random.default_rng(0).standard_normal((8, 16, 16, 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a_int8.tflite", Path(d) / "b_int8.tflite"
            int8_pruning.export_int8(_cnn(), calib, a)
            int8_pruning.export_int8(_cnn(), calib, b)
            (Path(d) / "pair").mkdir()
            cwd = os.getcwd()
            os.chdir(Path(d) / "pair")  # compiler output must not land in the working directory
            try:
                one = int8_pruning.compile_edgetpu(a, show_ops=True)
            finally:
                os.chdir(cwd)
            self.assertEqual(list((Path(d) / "pair").iterdir()), [])
            self.assertEqual(one, Path(d) / "a_int8_edgetpu.tflite")
            self.assertTrue(one.is_file())
            self.assertTrue((Path(d) / "a_int8.show_ops.txt").stat().st_size > 0)
            pair = int8_pruning.compile_edgetpu([a, b], Path(d) / "pair")
            self.assertEqual([p.name for p in pair],
                             ["a_int8_edgetpu.tflite", "b_int8_edgetpu.tflite"])
            self.assertTrue(all(p.is_file() for p in pair))


@unittest.skipUnless(torch is not None and HAVE_ONNX, "needs torch and onnx")
class ExportOnnx(unittest.TestCase):
    def test_nested_heads_are_flattened_and_named_by_level(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as d:
            info = int8_pruning.export_onnx(_Heads(), Path(d) / "h.onnx", (1, 3, 16, 16))
        self.assertEqual(info.output_names, ["cls_0", "cls_1", "box_0", "box_1"])


@unittest.skipUnless(np is not None, "needs numpy and Pillow")
class ImageFolder(unittest.TestCase):
    def test_shape_and_determinism(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            for i in range(5):
                sub = Path(d) / f"class{i % 2}"
                sub.mkdir(exist_ok=True)
                Image.fromarray(rng.integers(0, 255, (40 + i, 30, 3), dtype=np.uint8)).save(
                    sub / f"{i}.png")
            kw = dict(mean=[0.5] * 3, std=[0.5] * 3, seed=1)
            a = int8_pruning.export.image_folder(d, 3, 24, **kw)
            b = int8_pruning.export.image_folder(d, 3, 24, **kw)
            c = int8_pruning.export.image_folder(d, 3, (32, 16), crop=False, **kw)
        self.assertEqual(a.shape, (3, 24, 24, 3))
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(c.shape, (3, 32, 16, 3))


if __name__ == "__main__":
    unittest.main()
