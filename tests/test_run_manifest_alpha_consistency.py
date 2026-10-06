"""Regression test for the alpha-label vs. partition_alpha wiring bug found
in the highdim deterministic stability/retune/finals preparers: a manifest
row's "alpha" column is a human-readable label, but build_config() only ever
reads "partition_alpha" to configure the actual data partition. A generator
that updates one column without the other silently launches a run at a
different alpha than its own run_id/label claims. _require_launch_eligible()
now fails closed on that mismatch before any job is built."""

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import run_manifest  # noqa: E402


def _minimal_row(**overrides):
    row = {
        "run_id": "r1",
        "dataset": "femnist_z",
        "method": "fedgda_d",
        "seed": "0",
        "client_num_in_total": "10",
        "client_num_per_round": "10",
        "comm_round": "10",
        "epochs": "1",
        "batch_size": "32",
        "client_optimizer": "sgd",
        "learning_rate": "0.003",
        "weight_decay": "0.01",
        "partition_alpha": "0.5",
    }
    row.update(overrides)
    return row


class BuildConfigAlphaConsistencyTest(unittest.TestCase):
    def _build(self, row):
        return run_manifest.build_config(
            row,
            output_root=Path("/tmp/out"),
            gpu_id=0,
            default_learning_rate=None,
            default_weight_decay=None,
            override_comm_round=None,
            override_epochs=None,
            override_simple_model_selection_epochs=None,
            override_f_history_model_selection_epochs=None,
            override_model_selection_batch_size=None,
            override_model_selection_max_samples=None,
            override_skip_model_selection=None,
            override_skip_gmm_eval=None,
            override_auxiliary_regression=None,
            override_auxiliary_regression_epochs=None,
            override_append_round_csv=None,
            override_periodic_checkpoint_interval=None,
            override_dataloader_num_workers=None,
            override_dataloader_pin_memory=None,
        )

    def test_matching_alpha_and_partition_alpha_builds_cleanly(self):
        row = _minimal_row(alpha="0.5", partition_alpha="0.5")
        config = self._build(row)
        self.assertEqual(config["partition_alpha"], 0.5)

    def test_mismatched_alpha_label_is_rejected_before_launch(self):
        # This is exactly the bug found in
        # prepare_highdim_deterministic_stability_alpha0p1_20260826.py et al.:
        # alpha was relabeled to 0.1 but partition_alpha was left at the
        # inherited screen-template value of 0.5.
        row = _minimal_row(alpha="0.1", partition_alpha="0.5")
        with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "alpha label"):
            self._build(row)

    def test_row_without_alpha_label_column_is_unaffected(self):
        row = _minimal_row(partition_alpha="0.5")
        self.assertNotIn("alpha", row)
        config = self._build(row)
        self.assertEqual(config["partition_alpha"], 0.5)

    def test_non_numeric_alpha_label_is_rejected(self):
        row = _minimal_row(alpha="not-a-number", partition_alpha="0.5")
        with self.assertRaises(run_manifest.ManifestLaunchError):
            self._build(row)


if __name__ == "__main__":
    unittest.main()
