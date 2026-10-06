"""Gate 6 regression test: the shipped campaign contract
(experiments/eicu_study_a_validation/default_contract.json) must accept a
real 105-row manifest produced by prepare_eicu_study_a_manifest.py's --stage
all, using the actual field names/values that implementation now produces
(scenario_seed/optimizer_seed/seed_pair_id, g0="mlp", centralized methods
gda_d/sgda_s/oadam_s, aggregation_weighting="none" for centralized, etc.).

This is the "one canonical manifest/config/metadata/results schema accepted
by the campaign validator" item from the six implementation gates -- a
plain unit test import of both scripts, run against hand-built scenario
metadata fixtures (no real training or eICU data required).
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import prepare_eicu_study_a_manifest as manifest_gen  # noqa: E402
import validate_eicu_study_a_campaign as validator_mod  # noqa: E402

CONTRACT_PATH = Path(REPO_ROOT) / "experiments" / "eicu_study_a_validation" / "default_contract.json"


def _write_scenario_metadata(scenario_dir):
    for g0 in manifest_gen.G0_VARIANTS:
        for _, scenario_seed, _ in manifest_gen.CONFIRMATORY_SEED_PAIRS:
            meta = {
                "n_features_x": 43,
                "n_features_z": 43,
                "n_clients": 8,
                "input_dim": 43,
                "instrument_dim": 43,
                "outcome_dim": 1,
                "g0": {"kind": g0},
                "g0_display_label": manifest_gen.G0_DISPLAY_LABEL[g0],
                "scenario_seed": scenario_seed,
                "scenario_scope": "full_eicu",
                "is_demo": False,
                "scenario_checksum_sha256": "0" * 64,
                "scenario_checksum": "0" * 64,
                "eligible_client_ids": list(range(8)),
                "eligible_client_provenance": {
                    "method": "structural_instrument_variation_then_simulated_first_stage",
                    "real_z_filter": {"eligibility_seed": 20260101, "n_hospitals_after": 8},
                    "simulated_first_stage": {"n_clients_certified": 8},
                },
            }
            path = os.path.join(scenario_dir, f"{g0}_scenario_seed{scenario_seed}_metadata.json")
            with open(path, "w") as handle:
                json.dump(meta, handle)


def _selected_hyperparameters():
    return {
        manifest_gen.selection_key(g0, method): {"learning_rate": 0.001, "server_learning_rate": 1.5}
        for g0 in manifest_gen.G0_VARIANTS
        for method in manifest_gen.METHODS
    }


class Gate6CampaignValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="gate6_campaign_")
        self.scenario_dir = os.path.join(self.tmpdir, "scenarios")
        os.makedirs(self.scenario_dir)
        _write_scenario_metadata(self.scenario_dir)
        self.selected_path = os.path.join(self.tmpdir, "selected.json")
        with open(self.selected_path, "w") as handle:
            json.dump(_selected_hyperparameters(), handle)
        self.manifest_path = os.path.join(self.tmpdir, "manifest.csv")
        exit_code = manifest_gen.main([
            "--stage", "all",
            "--scenario-dir", self.scenario_dir,
            "--output-root", os.path.join(self.tmpdir, "results"),
            "--out", self.manifest_path,
            "--selected-hyperparameters", self.selected_path,
        ])
        self.assertEqual(exit_code, 0)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_shipped_contract_accepts_the_generated_105_row_manifest(self):
        report = validator_mod.validate_campaign(
            manifest=Path(self.manifest_path),
            contract=CONTRACT_PATH,
            phase="prelaunch",
        )
        if report["blocking_errors"]:
            self.fail(
                "prelaunch validation failed:\n"
                + json.dumps(report["blocking_errors"][:10], indent=2)
            )
        self.assertTrue(report["launchable"])

    def test_coverage_reports_all_three_required_roles_complete(self):
        report = validator_mod.validate_campaign(
            manifest=Path(self.manifest_path),
            contract=CONTRACT_PATH,
            phase="prelaunch",
        )
        coverage = report["coverage"]
        for role, expected_rows in (
            ("confirmatory", 30), ("centralized_baseline", 45), ("aggregation_ablation", 30),
        ):
            self.assertTrue(coverage[role]["complete"], f"{role} coverage incomplete: {coverage[role]}")
            self.assertEqual(coverage[role]["observed_rows"], expected_rows)

    def test_missing_row_breaks_launchability(self):
        """Sanity check that the contract is not vacuously permissive: drop
        one required row and prelaunch validation must reject it.
        """
        import csv

        with open(self.manifest_path, newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows = [r for r in rows if r["role"] != "confirmatory" or r["g0"] != "linear"]
        broken_path = os.path.join(self.tmpdir, "broken_manifest.csv")
        with open(broken_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        report = validator_mod.validate_campaign(
            manifest=Path(broken_path),
            contract=CONTRACT_PATH,
            phase="prelaunch",
        )
        self.assertFalse(report["launchable"])
        self.assertTrue(report["blocking_errors"])

    def test_wrong_ablation_aggregation_weighting_is_rejected(self):
        """Sanity check: an ablation row using uniform_clients (instead of
        sample_size) must be rejected -- confirms role-value validation is
        actually wired up, not just row-count checking.
        """
        import csv

        with open(self.manifest_path, newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        for row in rows:
            if row["role"] == "aggregation_ablation":
                row["aggregation_weighting"] = "uniform_clients"
        broken_path = os.path.join(self.tmpdir, "broken_ablation.csv")
        with open(broken_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        report = validator_mod.validate_campaign(
            manifest=Path(broken_path),
            contract=CONTRACT_PATH,
            phase="prelaunch",
        )
        self.assertFalse(report["launchable"])


if __name__ == "__main__":
    unittest.main()
