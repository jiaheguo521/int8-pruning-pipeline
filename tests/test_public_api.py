"""The public API and where code lives, so a new function or file cannot drift past them.

The rule: each pipeline stage is a package (pruning, recovery, export, compilation,
evaluation) whose `__init__.py` only re-exports, from modules whose names start with an
underscore, and lists the names in `__all__`. Everything else in the library is a `_`
module.

The layout checks need only the standard library; the API checks import the stages and
skip without torch.
"""
import ast
import importlib
import sys
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "int8_pruning"
sys.path.insert(0, str(PKG.parent))

STAGES = {
    "pruning": ["prune", "prune_trajectory", "count_params", "PruneResult"],
    "recovery": ["finetune", "distill", "EpochResult"],
    "export": ["export_int8", "export_onnx", "image_folder", "normalize_nhwc",
               "se_mean_pool", "adaptive_max_pool", "ExportInt8Result", "ExportOnnxResult"],
    "compilation": ["compile_edgetpu"],
    "evaluation": ["make_interpreter", "run_tflite", "latency", "LatencyResult"],
}
TOP = ["prune", "prune_trajectory", "count_params", "finetune", "distill", "export_int8",
       "export_onnx", "compile_edgetpu", "make_interpreter", "run_tflite", "latency",
       "seed_everything", *STAGES]

try:
    import torch  # noqa: F401
    import int8_pruning
except ImportError:
    int8_pruning = None


class Layout(unittest.TestCase):
    def test_only_the_stages_are_public(self):
        dirs = {p.name for p in PKG.iterdir() if p.is_dir() and p.name != "__pycache__"}
        self.assertEqual(dirs, set(STAGES))
        for py in PKG.rglob("*.py"):
            if py.name != "__init__.py":
                self.assertTrue(py.name.startswith("_"), py.relative_to(PKG))

    def test_a_stage_init_only_re_exports(self):
        for stage in STAGES:
            tree = ast.parse((PKG / stage / "__init__.py").read_text(encoding="utf-8"))
            defined = [n.name for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
            self.assertEqual(defined, [], stage)


@unittest.skipUnless(int8_pruning is not None, "needs torch")
class PublicApi(unittest.TestCase):
    def test_top_level(self):
        self.assertEqual(sorted(int8_pruning.__all__), sorted(TOP))
        for name in TOP:
            getattr(int8_pruning, name)

    def test_every_stage_reports_its_public_path(self):
        for stage, names in STAGES.items():
            module = importlib.import_module(f"int8_pruning.{stage}")
            self.assertEqual(sorted(module.__all__), sorted(names), stage)
            for name in names:
                self.assertEqual(getattr(module, name).__module__, f"int8_pruning.{stage}",
                                 f"{stage}.{name}")


if __name__ == "__main__":
    unittest.main()
