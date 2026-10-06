"""Unit tests for FHistoryLearningEvalNoStop.eval()'s per-checkpoint finite
diagnostics (closeout plan Phase 1 SS4.2) -- purely observational bookkeeping
that must not change the returned (epsilon_dev_history, f_of_z_dev_history)
tuple or any Psi arithmetic."""

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FEDGMM_ROOT = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example"
sys.path.insert(0, str(FEDGMM_ROOT))

from model_selection import learning_eval_nostop  # noqa: E402


class _StubModel:
    """Minimal g/f stand-in: eval()/train() return self (as real nn.Module
    does), __call__ applies a plain function."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)

    def eval(self):
        return self

    def train(self):
        return self


class _NoOpEval(learning_eval_nostop.FHistoryLearningEvalNoStop):
    def do_training_update(self, *args, **kwargs):
        pass


def _identity(x):
    return x


class LearningEvalDiagnosticsTest(unittest.TestCase):
    def test_records_one_entry_per_eval_checkpoint_with_epoch_index(self):
        evaluator = _NoOpEval(num_iter=3, eval_freq=1)
        x_dev = torch.zeros(2, 1)
        epsilon_history, f_history = evaluator.eval(
            x_train=None, z_train=None, y_train=None,
            x_dev=x_dev, z_dev=x_dev, y_dev=x_dev,
            g=_StubModel(_identity), f=_StubModel(_identity),
            g_optimizer=None, f_optimizer=None, game_objective=None,
        )
        # Return signature is untouched by the instrumentation.
        self.assertEqual(len(epsilon_history), 3)
        self.assertEqual(len(f_history), 3)
        self.assertEqual(
            [record["epoch"] for record in evaluator.last_eval_diagnostics],
            [0, 1, 2],
        )
        self.assertTrue(all(
            record["epsilon_dev_finite"] and record["f_of_z_dev_finite"]
            for record in evaluator.last_eval_diagnostics
        ))

    def test_detects_nonfinite_epsilon_dev_without_masking_finite_f_of_z_dev(self):
        def nan_fn(x):
            return x * float("nan")

        evaluator = _NoOpEval(num_iter=1, eval_freq=1)
        x_dev = torch.zeros(2, 1)
        evaluator.eval(
            x_train=None, z_train=None, y_train=None,
            x_dev=x_dev, z_dev=x_dev, y_dev=x_dev,
            g=_StubModel(nan_fn), f=_StubModel(_identity),
            g_optimizer=None, f_optimizer=None, game_objective=None,
        )
        record = evaluator.last_eval_diagnostics[0]
        self.assertFalse(record["epsilon_dev_finite"])
        self.assertTrue(record["f_of_z_dev_finite"])

    def test_eval_freq_skips_non_checkpoint_iterations(self):
        evaluator = _NoOpEval(num_iter=6, eval_freq=2)
        x_dev = torch.zeros(2, 1)
        evaluator.eval(
            x_train=None, z_train=None, y_train=None,
            x_dev=x_dev, z_dev=x_dev, y_dev=x_dev,
            g=_StubModel(_identity), f=_StubModel(_identity),
            g_optimizer=None, f_optimizer=None, game_objective=None,
        )
        self.assertEqual(
            [record["epoch"] for record in evaluator.last_eval_diagnostics],
            [0, 2, 4],
        )

    def test_diagnostics_reset_across_eval_calls(self):
        evaluator = _NoOpEval(num_iter=2, eval_freq=1)
        x_dev = torch.zeros(2, 1)
        for _ in range(2):
            evaluator.eval(
                x_train=None, z_train=None, y_train=None,
                x_dev=x_dev, z_dev=x_dev, y_dev=x_dev,
                g=_StubModel(_identity), f=_StubModel(_identity),
                g_optimizer=None, f_optimizer=None, game_objective=None,
            )
        # A stale second call must not accumulate on top of the first.
        self.assertEqual(len(evaluator.last_eval_diagnostics), 2)


if __name__ == "__main__":
    unittest.main()
