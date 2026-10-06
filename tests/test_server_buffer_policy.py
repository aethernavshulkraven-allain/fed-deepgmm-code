"""Regression tests for parameter-only federated server optimizer updates."""

import os
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_ROOT = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")
sys.path.insert(0, EXAMPLE_ROOT)

from experiment_utils import (  # noqa: E402
    EFFECTIVE_CONFIG_FIELDS,
    apply_parameter_server_update,
    batchnorm_running_var_min,
    get_effective_config,
    weighted_average_state_dicts,
)


class FrozenParameterExclusionTest(unittest.TestCase):
    def test_frozen_parameter_is_excluded_from_trainable_keys(self):
        model = nn.Linear(2, 2)
        model.bias.requires_grad_(False)
        trainable_keys = frozenset(
            name for name, param in model.named_parameters() if param.requires_grad
        )
        self.assertIn("weight", trainable_keys)
        self.assertNotIn("bias", trainable_keys)

    def test_frozen_parameter_is_aggregated_not_extrapolated(self):
        model = nn.Linear(1, 1)
        model.bias.requires_grad_(False)
        trainable_keys = frozenset(
            name for name, param in model.named_parameters() if param.requires_grad
        )
        base = {
            "weight": torch.tensor([[1.0]], dtype=torch.float64),
            "bias": torch.tensor([1.0], dtype=torch.float64),
        }
        aggregated = {
            "weight": torch.tensor([[3.0]], dtype=torch.float64),
            "bias": torch.tensor([9.0], dtype=torch.float64),
        }

        updated, delta = apply_parameter_server_update(
            base, aggregated, trainable_keys, 1.5,
        )

        # weight is trainable: theta_new = theta_old + lr * (agg - theta_old)
        self.assertEqual(updated["weight"].item(), 1.0 + 1.5 * (3.0 - 1.0))
        self.assertIn("weight", delta)
        # bias is frozen: aggregated directly, exactly like a BatchNorm buffer
        # -- never extrapolated with the server learning rate.
        self.assertEqual(updated["bias"].item(), 9.0)
        self.assertNotIn("bias", delta)


class ParameterOnlyServerUpdateTest(unittest.TestCase):
    def test_optimistic_math_applies_only_to_parameters(self):
        base = {
            "weight": torch.tensor(1.0, dtype=torch.float64),
            "bn.running_var": torch.tensor([0.01], dtype=torch.float64),
            "bn.num_batches_tracked": torch.tensor(7, dtype=torch.long),
        }
        aggregated = {
            "weight": torch.tensor(3.0, dtype=torch.float64),
            "bn.running_var": torch.tensor([0.002], dtype=torch.float64),
            "bn.num_batches_tracked": torch.tensor(12, dtype=torch.long),
        }
        previous_delta = {"weight": torch.tensor(0.5, dtype=torch.float64)}

        updated, delta = apply_parameter_server_update(
            base,
            aggregated,
            {"weight"},
            1.5,
            previous_parameter_delta=previous_delta,
            optimistic=True,
        )

        self.assertEqual(updated["weight"].item(), 6.25)
        self.assertEqual(delta["weight"].item(), 2.0)
        torch.testing.assert_close(
            updated["bn.running_var"], aggregated["bn.running_var"]
        )
        self.assertEqual(updated["bn.num_batches_tracked"].dtype, torch.long)
        self.assertEqual(updated["bn.num_batches_tracked"].item(), 12)

    def test_corrector_delta_can_use_lookahead_without_extrapolating_buffers(self):
        base = {
            "weight": torch.tensor(1.0),
            "running": torch.tensor(0.1),
        }
        lookahead = {
            "weight": torch.tensor(1.5),
            "running": torch.tensor(0.2),
        }
        correction = {
            "weight": torch.tensor(1.8),
            "running": torch.tensor(0.3),
        }

        updated, _ = apply_parameter_server_update(
            base,
            correction,
            {"weight"},
            2.0,
            delta_base_state=lookahead,
        )

        self.assertAlmostEqual(updated["weight"].item(), 1.6, places=6)
        self.assertAlmostEqual(updated["running"].item(), 0.3, places=6)


class BufferAggregationTest(unittest.TestCase):
    def test_integer_counters_use_max_and_preserve_dtype(self):
        states = [
            {"counter": torch.tensor(4, dtype=torch.long)},
            {"counter": torch.tensor(9, dtype=torch.long)},
        ]
        result = weighted_average_state_dicts(states, [0.25, 0.75])
        self.assertEqual(result["counter"].dtype, torch.long)
        self.assertEqual(result["counter"].item(), 9)


class BatchNormInvariantTest(unittest.TestCase):
    def test_negative_running_variance_is_rejected(self):
        model = nn.BatchNorm1d(2).double()
        model.running_var.fill_(-2e-5)
        with self.assertRaisesRegex(FloatingPointError, "running_var is negative"):
            batchnorm_running_var_min(model, "critic")

    def test_training_batch_repairs_the_reproduced_eval_failure(self):
        model = nn.BatchNorm1d(2).double()
        model.running_var.fill_(-2e-5)
        model.eval()
        invalid_output = model(torch.ones(4, 2, dtype=torch.float64))
        self.assertFalse(torch.isfinite(invalid_output).all())

        model.train()
        training_values = torch.tensor(
            [[-2.0, 1.0], [-1.0, 2.0], [1.0, 4.0], [2.0, 5.0]],
            dtype=torch.float64,
        )
        self.assertTrue(torch.isfinite(model(training_values)).all())
        self.assertGreaterEqual(batchnorm_running_var_min(model, "critic"), 0.0)
        model.eval()
        self.assertTrue(torch.isfinite(model(training_values)).all())


class EffectiveConfigBufferPolicyTest(unittest.TestCase):
    def test_policy_is_explicit_in_effective_config(self):
        config = get_effective_config(SimpleNamespace(dataset="abs"))
        self.assertEqual(config["server_buffer_policy"], "direct_client_aggregate")
        self.assertIn("server_buffer_policy", EFFECTIVE_CONFIG_FIELDS)

    def test_unknown_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "server_buffer_policy"):
            get_effective_config(
                SimpleNamespace(dataset="abs", server_buffer_policy="optimistic_all_state")
            )


if __name__ == "__main__":
    unittest.main()
