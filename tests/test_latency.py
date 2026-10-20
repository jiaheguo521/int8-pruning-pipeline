"""evaluation/_evaluation.py runs by path inside the Coral docker image, where int8_pruning
is not installed: it must import nothing from the package. Standard library only."""
import ast
import unittest
from pathlib import Path

LATENCY = Path(__file__).resolve().parents[1] / "src/int8_pruning/evaluation/_evaluation.py"


class LatencyIsStandalone(unittest.TestCase):
    def test_imports_nothing_from_the_package(self):
        tree = ast.parse(LATENCY.read_text(encoding="utf-8"))
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module or "" for n in ast.walk(tree)
                  if isinstance(n, ast.ImportFrom)]
        self.assertEqual([m for m in names if m.split(".")[0] == "int8_pruning"], [])
        self.assertFalse(any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(tree)),
                         "relative import")


if __name__ == "__main__":
    unittest.main()
