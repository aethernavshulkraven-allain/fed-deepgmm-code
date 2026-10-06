"""Tests for the P0 #6 fix: data-driven input dims in run_centralized_lowdim.py.

Centralized GDA/OAdam previously hardcoded input_dim_g=1, input_dim_f=2 (correct
only for the 1-D zoo scenarios), which made it impossible to run as a Study A
baseline. Dims are now derived from the loaded scenario's own tensor widths.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example"))

import run_centralized_lowdim as centralized  # noqa: E402


class CanonicalDatasetTest(unittest.TestCase):
    def test_zoo_datasets_still_accepted(self):
        for name in ("abs", "step", "linear", "sin", "absolute", "sine"):
            centralized.canonical_dataset(name)  # must not raise

    def test_eicu_semisynth_is_accepted(self):
        self.assertEqual(centralized.canonical_dataset("eicu_semisynth"), "eicu_semisynth")

    def test_unknown_dataset_still_raises(self):
        with self.assertRaises(ValueError):
            centralized.canonical_dataset("not_a_real_dataset")


class LoadPooledSplitsTest(unittest.TestCase):
    def test_eicu_without_scenario_name_raises(self):
        with self.assertRaises(ValueError):
            centralized.load_pooled_splits(
                "eicu_semisynth", REPO_ROOT, "cpu", scenario_name=None
            )


class BuildModelsTest(unittest.TestCase):
    def test_dims_and_widths_are_whatever_is_passed_in(self):
        import torch

        g, f = centralized.build_models(
            torch.device("cpu"), input_dim_g=43, input_dim_f=43, hidden_widths=[64, 64]
        )
        # First linear layer's input width must match what was requested, not
        # a hardcoded 1/2.
        first_g_linear = next(m for m in g.model if isinstance(m, torch.nn.Linear))
        first_f_linear = next(m for m in f.model if isinstance(m, torch.nn.Linear))
        self.assertEqual(first_g_linear.in_features, 43)
        self.assertEqual(first_f_linear.in_features, 43)
        self.assertEqual(first_g_linear.out_features, 64)

    def test_zoo_shaped_dims_still_work(self):
        import torch

        g, f = centralized.build_models(
            torch.device("cpu"), input_dim_g=1, input_dim_f=2, hidden_widths=[20, 20]
        )
        first_g_linear = next(m for m in g.model if isinstance(m, torch.nn.Linear))
        first_f_linear = next(m for m in f.model if isinstance(m, torch.nn.Linear))
        self.assertEqual(first_g_linear.in_features, 1)
        self.assertEqual(first_f_linear.in_features, 2)


class HiddenWidthsSelectionTest(unittest.TestCase):
    def test_eicu_uses_64_width_zoo_uses_20(self):
        self.assertEqual(centralized.EICU_HIDDEN_WIDTHS, [64, 64])
        self.assertEqual(centralized.ZOO_HIDDEN_WIDTHS, [20, 20])


if __name__ == "__main__":
    unittest.main()
