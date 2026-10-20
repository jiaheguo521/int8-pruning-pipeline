"""The examples: minimal.py end to end, and every other one as far as its --help.

An example is what a new user copies, so a library change that breaks it breaks the
first thing anyone runs. Needs torch, litert-torch and ai-edge-quantizer; skips without.
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# Where a model definition an example imports lives, as when the example runs as a script.
sys.path.insert(0, str(REPO / "examples"))

NEEDS = ("torch", "torch_pruning", "litert_torch", "ai_edge_quantizer")
HAVE = all(importlib.util.find_spec(m) for m in NEEDS)


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAVE, f"needs {', '.join(NEEDS)}")
class EveryExample(unittest.TestCase):
    def test_imports_and_parses_its_help(self):
        for path in sorted((REPO / "examples").glob("*.py")):
            module = _load(path)
            if path.name == "minimal.py" or not hasattr(module, "main"):
                continue                    # a model definition, not an example
            with self.subTest(example=path.name):
                with self.assertRaises(SystemExit) as done:
                    module.main(["--help"])
                self.assertEqual(done.exception.code, 0)


@unittest.skipUnless(HAVE, f"needs {', '.join(NEEDS)}")
class Minimal(unittest.TestCase):
    def test_runs_to_an_int8_model(self):
        minimal = _load(REPO / "examples" / "minimal.py")
        with tempfile.TemporaryDirectory() as d:
            out = minimal.main(d)
            self.assertTrue(Path(out.path).is_file())
        self.assertEqual(out.input_dtype, "int8")
        self.assertEqual(out.output_dtype, "int8")


if __name__ == "__main__":
    unittest.main()
