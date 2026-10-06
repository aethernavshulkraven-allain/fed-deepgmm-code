"""Tests for run_manifest.py's scenario_seed/optimizer_seed/seed_pair_id wiring.

protocol_v1.md S7.1: scenario_seed (which DGP/scenario artifact) and
optimizer_seed (this run's random_seed) must be recorded and propagated
separately, even though the manifest CSV's "seed" column continues to mean
"optimizer seed" for backward-compatible path/random_seed semantics. These
tests protect: (1) legacy rows without the new columns behave exactly as
before, (2) new rows with explicit scenario_seed get it threaded through
build_config and into the written YAML without disturbing random_seed.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import run_manifest  # noqa: E402


def _minimal_row(**overrides):
    row = {
        "run_id": "r1",
        "dataset": "eicu_semisynth",
        "method": "fedgda_s",
        "seed": "1101",
        "client_num_in_total": "3",
        "client_num_per_round": "3",
        "comm_round": "10",
        "epochs": "1",
        "batch_size": "3",
        "client_optimizer": "sgd",
        "learning_rate": "0.001",
        "weight_decay": "0.01",
        "partition_alpha": "0.0",
    }
    row.update(overrides)
    return row


class BuildConfigSeedFieldsTest(unittest.TestCase):
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

    def test_legacy_row_without_new_columns_defaults_scenario_seed_to_seed(self):
        row = _minimal_row()
        config = self._build(row)
        self.assertEqual(config["random_seed"], 1101)
        self.assertEqual(config["optimizer_seed"], 1101)
        self.assertEqual(config["scenario_seed"], 1101)
        self.assertEqual(config["seed_pair_id"], "")
        self.assertEqual(config["campaign_role"], "")

    def test_explicit_scenario_seed_is_kept_separate_from_optimizer_seed(self):
        row = _minimal_row(seed="1101", scenario_seed="101", seed_pair_id="confirmatory_01")
        config = self._build(row)
        self.assertEqual(config["random_seed"], 1101)
        self.assertEqual(config["optimizer_seed"], 1101)
        self.assertEqual(config["scenario_seed"], 101)
        self.assertEqual(config["seed_pair_id"], "confirmatory_01")

    def test_campaign_role_passthrough_for_ablation_rows(self):
        row = _minimal_row(campaign_role="aggregation_ablation", aggregation_weighting="sample_size")
        config = self._build(row)
        self.assertEqual(config["campaign_role"], "aggregation_ablation")
        self.assertEqual(config["aggregation_weighting"], "sample_size")

    def test_auxiliary_regression_is_opt_in(self):
        config = self._build(_minimal_row())
        self.assertFalse(config["auxiliary_regression"])
        self.assertEqual(config["auxiliary_regression_epochs"], 0)

        enabled = self._build(
            _minimal_row(auxiliary_regression="true", auxiliary_regression_epochs="2")
        )
        self.assertTrue(enabled["auxiliary_regression"])
        self.assertEqual(enabled["auxiliary_regression_epochs"], 2)

    def test_write_config_emits_seed_fields_into_train_args(self):
        row = _minimal_row(seed="1101", scenario_seed="101", seed_pair_id="confirmatory_01")
        config = self._build(row)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            run_manifest.write_config(path, config)
            text = path.read_text()
        self.assertIn("scenario_seed: 101", text)
        self.assertIn("optimizer_seed: 1101", text)
        self.assertIn("seed_pair_id: \"confirmatory_01\"", text)

    def test_write_config_omits_seed_pair_id_when_blank(self):
        row = _minimal_row()
        config = self._build(row)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            run_manifest.write_config(path, config)
            text = path.read_text()
        self.assertNotIn("seed_pair_id", text)
        self.assertNotIn("campaign_role", text)


if __name__ == "__main__":
    unittest.main()
