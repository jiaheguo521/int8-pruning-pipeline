"""`finetune` and `distill` on a small CNN, on the CPU.

Model selection fails silently when broken: the run finishes and keeps the last epoch,
which is a different model from the one its log calls best. Needs torch; skips without it.
"""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import int8_pruning
    int8_pruning.finetune  # imports the recovery module
except ImportError:
    torch = None


def _cnn(out=10):
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, out))


def _sgd(model, lr=1e-2):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


def _loader():
    g = torch.Generator().manual_seed(1)
    data = TensorDataset(torch.randn(32, 3, 8, 8, generator=g),
                         torch.randint(0, 10, (32,), generator=g))
    return DataLoader(data, batch_size=8)


@unittest.skipUnless(torch is not None, "needs torch")
class Finetune(unittest.TestCase):
    def test_select_restores_the_best_epoch(self):
        model, snapshots = _cnn(), []

        def evaluate():
            snapshots.append({k: v.clone() for k, v in model.state_dict().items()})
            return {"score": -len(snapshots)}      # epoch 1 is best, by construction

        history = int8_pruning.finetune(model, _loader(), 3, optimizer=_sgd(model),
                                        evaluate=evaluate, select="score")
        self.assertEqual(len(history), 3)
        self.assertIsInstance(history[0].train_loss, float)
        self.assertIsInstance(history[0].max_loss, float)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, snapshots[0][k]), k)

    def test_a_train_step_replaces_the_classification_loss(self):
        model, calls = _cnn(), []

        def step(m, batch):
            calls.append(1)
            x, _ = batch
            return m(x).pow(2).mean()

        history = int8_pruning.finetune(model, _loader(), 2, optimizer=_sgd(model),
                                        train_step=step)
        self.assertEqual(len(calls), 8)
        self.assertIsInstance(history[0].train_loss, float)
        self.assertIsInstance(history[0].max_loss, float)

    def test_the_schedule_is_stepped_once_per_epoch(self):
        model = _cnn()
        opt = _sgd(model, lr=0.1)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [1, 2], 0.1)
        history = int8_pruning.finetune(model, _loader(), 3, optimizer=opt, scheduler=sched)
        self.assertEqual([round(r.lr, 6) for r in history], [0.01, 0.001, 0.001])

    @unittest.skipUnless(importlib.util.find_spec("torch_pruning"), "needs torch_pruning")
    def test_an_optimizer_built_before_pruning_is_refused(self):
        # Pruning replaces the parameter tensors; an optimizer over the old ones trains
        # nothing, and the run would otherwise finish as if it had.
        model = _cnn()
        opt = _sgd(model)
        int8_pruning.prune(model, torch.randn(1, 3, 8, 8), 0.5, ignored_layers=[model[-1]])
        with self.assertRaises(ValueError):
            int8_pruning.finetune(model, _loader(), 1, optimizer=opt)


@unittest.skipUnless(torch is not None, "needs torch")
class Distill(unittest.TestCase):
    def test_the_teacher_is_frozen_and_untouched(self):
        student = _cnn(16)
        teacher = copy.deepcopy(student)
        before = {k: v.clone() for k, v in teacher.state_dict().items()}
        int8_pruning.distill(student, teacher, _loader(), 2, optimizer=_sgd(student))
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        for k, v in teacher.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)
        self.assertFalse(all(torch.equal(a, b) for a, b in
                             zip(student.state_dict().values(), before.values())))

    def test_a_train_step_takes_a_model_with_several_outputs(self):
        # A detector's per-level heads: two tensors out, compared head by head.
        class TwoHeads(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                self.body = nn.Conv2d(3, 4, 3, padding=1)
                self.a, self.b = nn.Conv2d(4, 2, 1), nn.Conv2d(4, 2, 1)

            def forward(self, x):
                f = self.body(x)
                return self.a(f), self.b(f)

        student = TwoHeads()
        teacher = copy.deepcopy(student)
        calls = []

        def step(s, t, batch):
            calls.append(t.training)
            x, _ = batch
            with torch.no_grad():
                targets = t(x)
            return sum(nn.functional.mse_loss(o, y) for o, y in zip(s(x), targets))

        history = int8_pruning.distill(student, teacher, _loader(), 2,
                                       optimizer=_sgd(student), train_step=step)
        self.assertEqual(len(calls), 8)
        self.assertFalse(any(calls))               # the teacher stays in eval mode
        self.assertEqual(len(history), 2)

    def test_simmat_takes_a_student_narrower_than_its_teacher(self):
        student, teacher = _cnn(8), _cnn(16)
        history = int8_pruning.distill(student, teacher, _loader(), 1,
                                       optimizer=_sgd(student), loss="simmat")
        self.assertEqual(len(history), 1)


    def test_select_restores_the_best_epoch_as_finetune_does(self):
        student, snapshots = _cnn(16), []
        teacher = copy.deepcopy(student)

        def evaluate():
            snapshots.append({k: v.clone() for k, v in student.state_dict().items()})
            return {"score": -len(snapshots)}      # epoch 1 is best, by construction

        int8_pruning.distill(student, teacher, _loader(), 3, optimizer=_sgd(student),
                             evaluate=evaluate, select="score")
        for k, v in student.state_dict().items():
            self.assertTrue(torch.equal(v, snapshots[0][k]), k)

if __name__ == "__main__":
    unittest.main()
