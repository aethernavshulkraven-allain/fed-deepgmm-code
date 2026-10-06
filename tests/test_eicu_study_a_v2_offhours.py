"""Focused tests for the isolated eICU Study A v2 setup."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prepare_eicu_study_a_v2_cohort as cohort_v2  # noqa: E402
import prepare_eicu_study_a_v2_manifest as manifest_v2  # noqa: E402
import prepare_eicu_study_a_v2_scenarios as scenario_v2  # noqa: E402
import select_eicu_study_a_v2_tuning as select_v2  # noqa: E402


def synthetic_cohort(n_hospitals: int = 4, rows_per_hospital: int = 20) -> pd.DataFrame:
    records = []
    for hospital in range(n_hospitals):
        for row in range(rows_per_hospital):
            split = "train" if row < 14 else ("dev" if row < 17 else "test")
            records.append(
                {
                    "patientunitstayid": hospital * 1000 + row,
                    "patienthealthsystemstayid": hospital * 1000 + row,
                    "uniquepid": f"{hospital}-{row}",
                    "hospitalid": hospital + 10,
                    "wardid": hospital * 10 + row % 2,
                    "z_off_hours": row % 2,
                    "split": split,
                    "age": 30.0 + row,
                    "admissionweight": np.nan if row % 7 == 0 else 60.0 + row,
                    "gender": "Male" if row % 2 else "Female",
                    "hospitaladmitsource": (
                        "Emergency Department" if row % 3 else "Operating Room"
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


class TestCohort(unittest.TestCase):
    def test_off_hours_boundaries(self):
        minutes = cohort_v2.parse_time_minutes(
            pd.Series(["06:59:00", "07:00:00", "18:59:00", "19:00:00"])
        )
        self.assertEqual(
            cohort_v2.off_hours_from_minutes(minutes).tolist(), [1, 0, 0, 1]
        )

    def test_patient_group_split_has_no_leakage(self):
        frame = synthetic_cohort(n_hospitals=2, rows_per_hospital=14)
        # Give one patient two admissions in the same hospital.
        frame.loc[1, "uniquepid"] = frame.loc[0, "uniquepid"]
        assigned = cohort_v2.assign_patient_group_splits(frame, seed=123)
        frame["split"] = assigned
        self.assertTrue((frame.groupby("uniquepid")["split"].nunique() == 1).all())
        self.assertTrue((frame.groupby("hospitalid")["split"].nunique() == 3).all())

    def test_client_gate_uses_hospital_not_ward(self):
        frame = synthetic_cohort(n_hospitals=2, rows_per_hospital=20)
        frame.loc[frame["hospitalid"] == 11, "z_off_hours"] = 0
        kept, audit = cohort_v2.client_eligibility(frame, min_client_rows=7)
        self.assertEqual(set(kept["hospitalid"]), {10})
        rejected = audit.loc[audit["hospitalid"] == 11].iloc[0]
        self.assertIn(
            "no_within_hospital_off_hours_variation",
            rejected["exclusion_reasons"],
        )


class TestScenarios(unittest.TestCase):
    def test_train_only_preprocessing_and_no_identifier_inputs(self):
        frame = synthetic_cohort()
        train = frame["split"].to_numpy() == "train"
        w, names, preprocessing = scenario_v2.build_covariates(frame, train)
        self.assertEqual(len(w), len(frame))
        self.assertEqual(preprocessing["fit_rows"], "train_only")
        self.assertTrue(
            scenario_v2.FORBIDDEN_MODEL_COLUMNS.isdisjoint(set(names))
        )
        age_column = names.index("age")
        self.assertAlmostEqual(float(w[train, age_column].mean()), 0.0, places=12)

    def test_continuous_treatment_is_reproducible(self):
        frame = synthetic_cohort()
        train = frame["split"].to_numpy() == "train"
        w, _, _ = scenario_v2.build_covariates(frame, train)
        z = frame["z_off_hours"].to_numpy(dtype=float)
        codes = pd.Categorical(frame["hospitalid"]).codes
        kwargs = dict(
            scenario_seed=7,
            generation_attempt=0,
            instrument_strength=2.0,
            rho_x=1.0,
            treatment_noise=0.5,
            outcome_noise=0.5,
            client_heterogeneity=0.5,
        )
        first = scenario_v2.generate_common_dgp(w, z, codes, **kwargs)
        second = scenario_v2.generate_common_dgp(w, z, codes, **kwargs)
        np.testing.assert_array_equal(first["treatment"], second["treatment"])
        self.assertGreater(len(np.unique(first["treatment"])), 2)

    def test_frozen_mlp_changes_by_scenario_seed(self):
        _, first = scenario_v2.make_g0(
            "mlp", 3, scenario_seed=1, generation_attempt=0
        )
        _, second = scenario_v2.make_g0(
            "mlp", 3, scenario_seed=2, generation_attempt=0
        )
        self.assertNotEqual(first["w1"], second["w1"])
        self.assertEqual(first["hidden_width"], 32)


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        for seed in (11, 101, 102, 103, 104, 105):
            for g0 in manifest_v2.G0_VARIANTS:
                metadata = {
                    "protocol_version": manifest_v2.PROTOCOL_VERSION,
                    "certification_passed": True,
                    "n_clients": 4,
                    "input_dim": 8,
                    "instrument_dim": 8,
                    "g0_display_label": (
                        "frozen_random_mlp" if g0 == "mlp" else g0
                    ),
                    "scenario_checksum_sha256": f"{seed:064x}",
                    "scenario_scope": "demo",
                }
                path = self.root / f"{g0}_scenario_seed{seed}_metadata.json"
                path.write_text(json.dumps(metadata))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_tuning_matrix_and_architecture(self):
        rows = manifest_v2.generate_tuning(self.root, self.root / "results")
        # 3 g0 x 2 federated methods x the protocol's LR x server-LR grid.
        self.assertEqual(
            len(rows),
            len(manifest_v2.G0_VARIANTS)
            * len(manifest_v2.FEDERATED_METHODS)
            * len(manifest_v2.LOCAL_LR_MULTIPLIERS)
            * len(manifest_v2.SERVER_LEARNING_RATES),
        )
        self.assertTrue(all(row["hidden_widths"] == "32,32" for row in rows))
        self.assertTrue(all(row["model_activation"] == "relu" for row in rows))
        self.assertTrue(all(row["batch_size"] == 4 for row in rows))
        self.assertTrue(
            all(row["require_multibatch_stochastic"] == "true" for row in rows)
        )
        self.assertTrue(
            all(row["aggregation_weighting"] == "uniform_clients" for row in rows)
        )
        self.assertTrue(all(row["scenario_seed"] == 11 for row in rows))

    def test_final_matrix_is_frozen_105(self):
        selected = {
            f"{g0}:{method}": {
                "learning_rate": 0.001,
                "server_learning_rate": 1.0,
                "run_id": "validation-selected",
            }
            for g0 in manifest_v2.G0_VARIANTS
            for method in manifest_v2.FEDERATED_METHODS
        }
        rows = manifest_v2.generate_final(
            self.root, self.root / "results", selected
        )
        counts = pd.Series([row["role"] for row in rows]).value_counts().to_dict()
        self.assertEqual(
            counts,
            {
                "centralized_baseline": 45,
                "confirmatory": 30,
                "aggregation_ablation": 30,
            },
        )
        self.assertEqual(len({row["run_id"] for row in rows}), 105)
        self.assertTrue(
            all(
                row["test_mse_used_for_selection"] == "false"
                for row in rows
            )
        )


class TestTuningSelection(unittest.TestCase):
    def test_exact_validation_rule_and_no_test_field_dependency(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            rows = []
            for g0 in manifest_v2.G0_VARIANTS:
                for method in manifest_v2.FEDERATED_METHODS:
                    for candidate in range(select_v2.EXPECTED_CANDIDATES_PER_GROUP):
                        row = {
                            "g0": g0,
                            "method": method,
                            "output_root": str(root),
                            "dataset": "eicu_test",
                            "optimizer_seed": 1011,
                            "run_id": f"{g0}-{method}-{candidate}",
                            "learning_rate": 0.001 * (candidate + 1),
                            "server_learning_rate": 1.0,
                        }
                        path = select_v2.run_dir(row)
                        path.mkdir(parents=True)
                        val_mse = 1.0 if candidate in (0, 1) else 2.0 + candidate
                        val_moment = 1.0 if candidate == 1 else 2.0
                        metrics = {
                            "diverged": False,
                            "best_validation_mse": val_mse,
                            "equal_client_validation_moment_violation_at_best_validation": val_moment,
                            "test_mse_at_best_validation": -999999.0
                            if candidate == 5
                            else 999999.0,
                            "final_vs_best_validation_gap": -999999.0
                            if candidate == 0
                            else 999999.0,
                        }
                        (path / "metrics.json").write_text(json.dumps(metrics))
                        rows.append(row)
            selected, _ = select_v2.select(rows)
            self.assertEqual(len(selected), 6)
            self.assertTrue(
                all(
                    choice["run_id"].endswith("-1")
                    for choice in selected.values()
                )
            )
            self.assertTrue(
                all(choice["test_fields_read"] == [] for choice in selected.values())
            )


if __name__ == "__main__":
    unittest.main()
