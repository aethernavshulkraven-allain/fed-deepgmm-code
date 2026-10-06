"""Unit tests for do_model_selection's fail-closed assertion (closeout plan
SS3.3) and its ModelSelectionFailure diagnostics (SS4.2). Uses stub models and
a stub learning_eval so the real DeepGMM/Psi arithmetic in
max_approx_psi_eval is never exercised -- that arithmetic is frozen legacy
code this campaign explicitly does not change, and is mocked to a fixed
"no valid candidate" outcome instead."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FEDGMM_ROOT = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example"
sys.path.insert(0, str(FEDGMM_ROOT))

import model_selection_class  # noqa: E402
from experiment_utils import ModelSelectionFailure  # noqa: E402


class _FakeModel:
    def __init__(self):
        self.initialize_calls = 0

    def initialize(self):
        self.initialize_calls += 1


class _FakeLearningEval:
    """Returns one real epsilon_dev/f_of_z_dev tensor per candidate (so
    do_model_selection's own tensor bookkeeping still runs for real) and a
    caller-controlled last_eval_diagnostics list, mirroring
    FHistoryLearningEvalNoStop.eval()'s contract."""

    def __init__(self, diagnostics):
        self.last_eval_diagnostics = diagnostics
        self.eval_calls = 0

    def eval(self, **kwargs):
        self.eval_calls += 1
        return [torch.zeros(2)], [torch.zeros(2)]


def _learning_args():
    return {
        "g_optimizer_factory": lambda g: None,
        "f_optimizer_factory": lambda f: None,
        "game_objective": None,
    }


def _make_selection(g_models, f_models, learning_args_list, learning_eval, failure_context=None):
    return model_selection_class.FHistoryModelSelectionV3(
        g_model_list=g_models,
        f_model_list=f_models,
        learning_args_list=learning_args_list,
        default_g_optimizer_factory=lambda g: None,
        default_f_optimizer_factory=lambda f: None,
        g_simple_model_eval=None,
        f_simple_model_eval=None,
        learning_eval=learning_eval,
        psi_eval_max_no_progress=10,
        psi_eval_burn_in=0,
        failure_context=failure_context or {"dataset": "fixture", "random_seed": 0},
    )


class FailClosedAssertionTest(unittest.TestCase):
    def test_more_than_one_combination_raises_assertion_error(self):
        # A real exception, not `assert` -- assertions are stripped under
        # `python -O`/PYTHONOPTIMIZE, which would silently let a
        # multi-candidate config through instead of failing closed.
        selection = _make_selection(
            [_FakeModel(), _FakeModel()], [_FakeModel()],
            [_learning_args()], _FakeLearningEval([]),
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            selection.do_model_selection(
                x_train=None, z_train=None, y_train=None,
                x_dev=None, z_dev=None, y_dev=None,
            )

    def test_multiple_learning_args_also_raises_assertion_error(self):
        selection = _make_selection(
            [_FakeModel()], [_FakeModel()],
            [_learning_args(), _learning_args()], _FakeLearningEval([]),
        )
        with self.assertRaises(RuntimeError):
            selection.do_model_selection(
                x_train=None, z_train=None, y_train=None,
                x_dev=None, z_dev=None, y_dev=None,
            )

    def test_exactly_one_combination_does_not_raise_the_assertion(self):
        # Still fails downstream (max_approx_psi_eval mocked to -inf), but
        # must get past the fail-closed check itself without raising.
        diagnostics = [{"epoch": 0, "epsilon_dev_finite": True, "f_of_z_dev_finite": True}]
        selection = _make_selection(
            [_FakeModel()], [_FakeModel()], [_learning_args()],
            _FakeLearningEval(diagnostics),
        )
        with patch.object(
            model_selection_class, "max_approx_psi_eval",
            return_value=(float("-inf"), torch.zeros(2)),
        ):
            with self.assertRaises(ModelSelectionFailure):
                selection.do_model_selection(
                    x_train=None, z_train=None, y_train=None,
                    x_dev=None, z_dev=None, y_dev=None,
                )


class ModelSelectionFailureDiagnosticsTest(unittest.TestCase):
    def test_no_valid_candidate_raises_model_selection_failure_with_diagnostics(self):
        diagnostics = [
            {"epoch": 0, "epsilon_dev_finite": True, "f_of_z_dev_finite": True},
            {"epoch": 1, "epsilon_dev_finite": False, "f_of_z_dev_finite": True},
        ]
        learning_eval = _FakeLearningEval(diagnostics)
        selection = _make_selection(
            [_FakeModel()], [_FakeModel()], [_learning_args()], learning_eval,
            failure_context={"dataset": "femnist_z", "random_seed": 1},
        )
        with patch.object(
            model_selection_class, "max_approx_psi_eval",
            return_value=(float("-inf"), torch.zeros(2)),
        ):
            with self.assertRaises(ModelSelectionFailure) as ctx:
                selection.do_model_selection(
                    x_train=None, z_train=None, y_train=None,
                    x_dev=None, z_dev=None, y_dev=None,
                )
        exc = ctx.exception
        self.assertEqual(exc.diagnostics["per_epoch"], diagnostics)
        self.assertEqual(exc.diagnostics["best_score"], float("-inf"))
        self.assertEqual(exc.diagnostics["failure_context"]["dataset"], "femnist_z")
        # format_no_valid_model_selection_error's human-readable message is
        # preserved as the exception's own message.
        self.assertIn("No valid model-selection candidate", str(exc))

    def test_learning_eval_is_still_called_exactly_once_for_the_single_candidate(self):
        learning_eval = _FakeLearningEval([])
        selection = _make_selection(
            [_FakeModel()], [_FakeModel()], [_learning_args()], learning_eval,
        )
        with patch.object(
            model_selection_class, "max_approx_psi_eval",
            return_value=(float("-inf"), torch.zeros(2)),
        ):
            with self.assertRaises(ModelSelectionFailure):
                selection.do_model_selection(
                    x_train=None, z_train=None, y_train=None,
                    x_dev=None, z_dev=None, y_dev=None,
                )
        self.assertEqual(learning_eval.eval_calls, 1)


if __name__ == "__main__":
    unittest.main()
