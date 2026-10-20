"""`prune` and `prune_trajectory` on a small CNN, on the CPU.

The trajectory's two contracts fail silently when broken: a ratio measured against
the current model instead of the dense one prunes too much and reports the requested
number, and a descending ratio prunes nothing and yields a model under the wrong label.
Needs torch and torch_pruning; skips without them.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
    import torch.nn as nn
    import int8_pruning
    int8_pruning.prune  # imports torch_pruning
except ImportError:
    torch = None


def _cnn():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)).eval()


@unittest.skipUnless(torch is not None, "needs torch and torch_pruning")
class Prune(unittest.TestCase):
    def setUp(self):
        self.example = torch.randn(1, 3, 16, 16)

    def test_reaches_the_ratio_and_keeps_the_output_shape(self):
        model = _cnn()
        dense = int8_pruning.count_params(model)
        info = int8_pruning.prune(model, self.example, 0.5, ignored_layers=[model[-1]])
        self.assertGreaterEqual(info.actual_ratio, 0.5)
        self.assertLess(info.actual_ratio, 0.6)
        self.assertEqual(info.params, int8_pruning.count_params(model))
        self.assertAlmostEqual(1 - info.params / dense, info.actual_ratio)
        self.assertEqual(tuple(model(self.example).shape), (1, 10))

    def test_dense_params_measures_the_ratio_against_the_original_model(self):
        # A model already cut down before prune() (a head shrunk first, say): 0.5 of the
        # original, not 0.5 of what was left.
        model = _cnn()
        original = int8_pruning.count_params(model)
        int8_pruning.prune(model, self.example, 0.3, ignored_layers=[model[-1]])
        info = int8_pruning.prune(model, self.example, 0.5, dense_params=original,
                                  ignored_layers=[model[-1]])
        self.assertLess(info.actual_ratio, 0.6)
        self.assertAlmostEqual(1 - info.params / original, info.actual_ratio)


    def test_an_importance_object_and_pruner_kwargs_reach_torch_pruning(self):
        import torch_pruning as tp
        a, b = _cnn(), _cnn()
        int8_pruning.prune(a, self.example, 0.5, ignored_layers=[a[-1]], round_to=8)
        int8_pruning.prune(b, self.example, 0.5, ignored_layers=[b[-1]], round_to=8,
                           importance=tp.importance.MagnitudeImportance(p=2))
        self.assertEqual(a.state_dict().keys(), b.state_dict().keys())
        for k in a.state_dict():
            self.assertTrue(torch.equal(a.state_dict()[k], b.state_dict()[k]), k)
        self.assertTrue(all(m.out_channels % 8 == 0 for m in a if isinstance(m, nn.Conv2d)))

@unittest.skipUnless(torch is not None, "needs torch and torch_pruning")
class Trajectory(unittest.TestCase):
    def setUp(self):
        self.example = torch.randn(1, 3, 16, 16)

    def test_every_ratio_is_measured_against_the_dense_model(self):
        model = _cnn()
        dense = int8_pruning.count_params(model)
        rungs = list(int8_pruning.prune_trajectory(model, self.example, [0.3, 0.5],
                                                   ignored_layers=[model[-1]]))
        self.assertEqual([r for r, _ in rungs], [0.3, 0.5])
        # 0.5 after 0.3 is half of the dense model, not 0.3 + 0.5 * 0.7 = 0.65.
        last = rungs[-1][1].actual_ratio
        self.assertLess(last, 0.6)
        self.assertAlmostEqual(1 - rungs[-1][1].params / dense, last)

    def test_the_first_rung_is_what_prune_gives(self):
        a, b = _cnn(), _cnn()
        int8_pruning.prune(a, self.example, 0.3, ignored_layers=[a[-1]])
        next(iter(int8_pruning.prune_trajectory(b, self.example, [0.3, 0.5],
                                                ignored_layers=[b[-1]])))
        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(sa.keys(), sb.keys())
        for k in sa:
            self.assertTrue(torch.equal(sa[k], sb[k]), k)

    def test_a_ratio_that_does_not_ascend_is_refused_before_anything_runs(self):
        model = _cnn()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for ratios in ([0.5, 0.3], [0.3, 0.3]):
            with self.assertRaises(ValueError):
                int8_pruning.prune_trajectory(model, self.example, ratios)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)


if __name__ == "__main__":
    unittest.main()
