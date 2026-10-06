"""End-to-end tests for scripts/analyze_eicu_study_a_v2_confirmatory.py.

Built and run BEFORE any Study A v2 confirmatory result exists (see
experiments/eicu_study_a_v2_offhours_demo_20260727/status.md: the 30-row
primary federated matrix is not materialized yet). Every run directory here
is a synthetic fixture written to a temp dir -- nothing under this repo's
results/ or experiments/ trees is touched or required, matching the "no real
v2 confirmatory data exists yet" constraint this analyzer was written under.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import analyze_eicu_study_a_v2_confirmatory as confirmatory  # noqa: E402

G0_VARIANTS = confirmatory.G0_VARIANTS
SEED_PAIRS = confirmatory.CONFIRMATORY_SEED_PAIRS


def manifest_row(root, g0, method, scenario_seed, optimizer_seed, role="confirmatory"):
    run_id = f"confirmatory_{g0}_{method}_seed{optimizer_seed}"
    return {
        "role": role,
        "g0": g0,
        "method": method,
        "dataset": "eicu_test",
        "output_root": str(root),
        "scenario_seed": scenario_seed,
        "optimizer_seed": optimizer_seed,
        "seed": optimizer_seed,
        "run_id": run_id,
    }


def write_run_metrics(root, row, *, primary_mse, diverged=False, extra=None):
    run_dir = confirmatory.run_dir_for(row)
    os.makedirs(run_dir, exist_ok=True)
    metrics = {
        "diverged": diverged,
        "equal_client_test_mse_at_best_validation": primary_mse,
        "test_mse_at_best_validation": primary_mse,
        "sample_weighted_test_mse_at_best_validation": primary_mse * 1.1,
        "equal_client_test_moment_violation_at_best_validation": 0.01,
        "sample_weighted_test_moment_violation_at_best_validation": 0.012,
        "equal_client_final_test_structural_mse": primary_mse + 0.05,
        "sample_weighted_final_test_structural_mse": primary_mse * 1.1 + 0.05,
        "final_test_mse": primary_mse + 0.05,
        "final_vs_best_test_gap": 0.05,
        "final_vs_best_validation_gap": 0.03,
        "runtime_seconds": 120.5,
        "equal_client_absolute_ate_error_at_best_validation": None,
        "sample_weighted_absolute_ate_error_at_best_validation": None,
        "equal_client_individual_effect_mae_at_best_validation": None,
        "sample_weighted_individual_effect_mae_at_best_validation": None,
        "per_client_metrics_artifact": None,
    }
    if extra:
        metrics.update(extra)
    with open(os.path.join(run_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle)
    return run_dir


def write_mse_by_round(run_dir, n_rounds=60, base=1.0, diverged_round=None):
    path = os.path.join(run_dir, "mse_by_round.csv")
    fieldnames = ["round", "val_mse", "primary_val_mse", "finite", "diverged"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for round_idx in range(n_rounds):
            is_diverged = diverged_round is not None and round_idx == diverged_round
            writer.writerow({
                "round": round_idx,
                "val_mse": base,
                "primary_val_mse": base,
                "finite": "False" if is_diverged else "True",
                "diverged": "True" if is_diverged else "False",
            })


def write_per_client_metrics_csv(run_dir, values):
    path = os.path.join(run_dir, "per_client_metrics.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client_id", "n", "structural_mse", "moment_violation"])
        writer.writeheader()
        for i, value in enumerate(values):
            writer.writerow({"client_id": i, "n": 10, "structural_mse": value, "moment_violation": 0.01})
    return path


def write_checkpoint_eval_summary(run_dir, *, equal_client_ate, sample_weighted_ate, equal_client_ite, sample_weighted_ite):
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    payload = {
        "aggregates": {
            "absolute_ate_error": {"equal_client": equal_client_ate, "sample_weighted": sample_weighted_ate},
            "individual_effect_mae": {"equal_client": equal_client_ite, "sample_weighted": sample_weighted_ite},
        }
    }
    with open(os.path.join(checkpoints_dir, "per_client_eval_best_validation_test_summary.json"), "w") as handle:
        json.dump(payload, handle)


def build_full_matrix(root, *, ogda_offset_by_g0):
    """Build all 3 g0 x 5 pairs x 2 methods = 30 rows. FedGDA MSE is fixed at
    1.0 per pair; FedOGDA MSE = 1.0 + ogda_offset_by_g0[g0] (negative =
    FedOGDA better on the primary endpoint)."""
    rows = []
    for g0 in G0_VARIANTS:
        offset = ogda_offset_by_g0[g0]
        for scenario_seed, optimizer_seed in SEED_PAIRS:
            gda_row = manifest_row(root, g0, "fedgda_s", scenario_seed, optimizer_seed)
            ogda_row = manifest_row(root, g0, "fedogda_s", scenario_seed, optimizer_seed)
            gda_dir = write_run_metrics(root, gda_row, primary_mse=1.0)
            ogda_dir = write_run_metrics(root, ogda_row, primary_mse=1.0 + offset)
            write_mse_by_round(gda_dir)
            write_mse_by_round(ogda_dir)
            rows.append(gda_row)
            rows.append(ogda_row)
    return rows


class ClassifyVerdictTest(unittest.TestCase):
    def test_unanimous_ogda_better_is_favored(self):
        result = confirmatory.classify_verdict([-1, -1, -1, -1, -1], min_sign_consistency=4)
        self.assertEqual(result["verdict"], "fedogda_favored")

    def test_four_of_five_clears_the_bar(self):
        result = confirmatory.classify_verdict([-1, -1, -1, -1, 0.5], min_sign_consistency=4)
        self.assertEqual(result["verdict"], "fedogda_favored")

    def test_bare_majority_three_of_five_is_inconclusive(self):
        result = confirmatory.classify_verdict([-1, -1, -1, 0.5, 0.5], min_sign_consistency=4)
        self.assertEqual(result["verdict"], "inconclusive")

    def test_unanimous_gda_better_is_favored(self):
        result = confirmatory.classify_verdict([1, 1, 1, 1, 1], min_sign_consistency=4)
        self.assertEqual(result["verdict"], "fedgda_favored")

    def test_empty_differences_is_inconclusive(self):
        result = confirmatory.classify_verdict([], min_sign_consistency=4)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertIsNone(result["mean_difference"])

    def test_pooled_threshold_twelve_of_fifteen(self):
        favored = confirmatory.classify_verdict([-1] * 12 + [1] * 3, min_sign_consistency=12)
        self.assertEqual(favored["verdict"], "fedogda_favored")
        not_favored = confirmatory.classify_verdict([-1] * 11 + [1] * 4, min_sign_consistency=12)
        self.assertEqual(not_favored["verdict"], "inconclusive")


class ConfirmatoryAnalyzerEndToEndTest(unittest.TestCase):
    def test_fedogda_clearly_better_gives_favored_verdicts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = build_full_matrix(
                tmpdir,
                ogda_offset_by_g0={"linear": -0.5, "interaction": -0.5, "mlp": -0.5},
            )
            result = confirmatory.run_analysis(rows)
        for g0 in G0_VARIANTS:
            self.assertEqual(
                result["pairwise_primary_endpoint"]["per_g0"][g0]["verdict"],
                "fedogda_favored",
            )
        self.assertEqual(result["pairwise_primary_endpoint"]["pooled"]["verdict"], "fedogda_favored")
        self.assertEqual(result["pairwise_primary_endpoint"]["pooled"]["n_pairs"], 15)

    def test_fedogda_worse_reports_negative_result_cleanly(self):
        # Explicit requirement: the analyzer must be able to report FedOGDA
        # losing, not just winning.
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = build_full_matrix(
                tmpdir,
                ogda_offset_by_g0={"linear": 0.5, "interaction": 0.5, "mlp": 0.5},
            )
            result = confirmatory.run_analysis(rows)
        for g0 in G0_VARIANTS:
            self.assertEqual(
                result["pairwise_primary_endpoint"]["per_g0"][g0]["verdict"],
                "fedgda_favored",
            )
        self.assertEqual(result["pairwise_primary_endpoint"]["pooled"]["verdict"], "fedgda_favored")
        report = confirmatory.render_report(result)
        self.assertIn("fedgda_favored", report)
        # Must not spin this into a FedOGDA win anywhere in the pooled verdict line.
        self.assertNotIn("**pooled verdict: fedogda_favored**", report)

    def test_mixed_signs_across_g0_is_inconclusive_pooled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = build_full_matrix(
                tmpdir,
                ogda_offset_by_g0={"linear": -0.5, "interaction": 0.5, "mlp": 0.0001},
            )
            result = confirmatory.run_analysis(rows)
        self.assertEqual(result["pairwise_primary_endpoint"]["per_g0"]["linear"]["verdict"], "fedogda_favored")
        self.assertEqual(result["pairwise_primary_endpoint"]["per_g0"]["interaction"]["verdict"], "fedgda_favored")
        report = confirmatory.render_report(result)
        self.assertIn("inconclusive", report)

    def test_diverged_seed_excluded_from_pairing_and_counted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = []
            for scenario_seed, optimizer_seed in SEED_PAIRS:
                gda_row = manifest_row(tmpdir, "linear", "fedgda_s", scenario_seed, optimizer_seed)
                ogda_row = manifest_row(tmpdir, "linear", "fedogda_s", scenario_seed, optimizer_seed)
                # optimizer_seed 1103's FedOGDA run diverged.
                ogda_diverged = optimizer_seed == 1103
                write_run_metrics(tmpdir, gda_row, primary_mse=1.0)
                write_run_metrics(tmpdir, ogda_row, primary_mse=0.5, diverged=ogda_diverged)
                rows.append(gda_row)
                rows.append(ogda_row)
            result = confirmatory.run_analysis(rows)
        pair = result["pairwise_primary_endpoint"]["per_g0"]["linear"]
        self.assertEqual(pair["n_pairs"], 4)
        self.assertNotIn(1103, pair["seeds"])
        self.assertEqual(result["summary"]["linear"]["fedogda_s"]["n_diverged"], 1)

    def test_incomplete_manifest_missing_runs_do_not_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = []
            for scenario_seed, optimizer_seed in SEED_PAIRS:
                gda_row = manifest_row(tmpdir, "linear", "fedgda_s", scenario_seed, optimizer_seed)
                ogda_row = manifest_row(tmpdir, "linear", "fedogda_s", scenario_seed, optimizer_seed)
                write_run_metrics(tmpdir, gda_row, primary_mse=1.0)
                # FedOGDA runs simply have not been launched yet -- no metrics.json.
                rows.append(gda_row)
                rows.append(ogda_row)
            result = confirmatory.run_analysis(rows)
        self.assertEqual(result["summary"]["linear"]["fedogda_s"]["n_seeds_complete"], 0)
        self.assertIsNone(result["summary"]["linear"]["fedogda_s"]["primary"])
        self.assertEqual(result["pairwise_primary_endpoint"]["per_g0"]["linear"]["n_pairs"], 0)
        self.assertEqual(result["pairwise_primary_endpoint"]["per_g0"]["linear"]["verdict"], "inconclusive")
        # Must not raise, and must still render a report.
        confirmatory.render_report(result)

    def test_effect_metrics_fall_back_to_checkpoint_eval_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            row = manifest_row(tmpdir, "linear", "fedgda_s", 101, 1101)
            run_dir = write_run_metrics(tmpdir, row, primary_mse=1.0)
            write_checkpoint_eval_summary(
                run_dir,
                equal_client_ate=0.2, sample_weighted_ate=0.25,
                equal_client_ite=0.4, sample_weighted_ite=0.45,
            )
            summary = confirmatory.load_per_client_eval_summary(row)
            metrics = confirmatory.load_metrics(row)
            effect = confirmatory.effect_metrics(metrics, summary)
        self.assertEqual(effect["source"], "per_client_checkpoint_eval")
        self.assertAlmostEqual(effect["equal_client_absolute_ate_error_at_best_validation"], 0.2)
        self.assertAlmostEqual(effect["equal_client_individual_effect_mae_at_best_validation"], 0.4)

    def test_effect_metrics_unavailable_when_no_fallback_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            row = manifest_row(tmpdir, "linear", "fedgda_s", 101, 1101)
            write_run_metrics(tmpdir, row, primary_mse=1.0)
            metrics = confirmatory.load_metrics(row)
            summary = confirmatory.load_per_client_eval_summary(row)
            effect = confirmatory.effect_metrics(metrics, summary)
        self.assertEqual(effect["source"], "unavailable")

    def test_group_summary_reports_effect_availability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = []
            for scenario_seed, optimizer_seed in SEED_PAIRS:
                row = manifest_row(tmpdir, "linear", "fedgda_s", scenario_seed, optimizer_seed)
                run_dir = write_run_metrics(tmpdir, row, primary_mse=1.0)
                if optimizer_seed == 1101:
                    write_checkpoint_eval_summary(
                        run_dir, equal_client_ate=0.1, sample_weighted_ate=0.1,
                        equal_client_ite=0.2, sample_weighted_ite=0.2,
                    )
                rows.append(row)
            result = confirmatory.run_analysis(rows)
        effect = result["summary"]["linear"]["fedgda_s"]["effect"]
        self.assertEqual(effect["n_available"], 1)
        self.assertEqual(effect["n_expected"], 5)

    def test_per_hospital_distribution_falls_back_to_csv_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            row = manifest_row(tmpdir, "linear", "fedgda_s", 101, 1101)
            run_dir = write_run_metrics(tmpdir, row, primary_mse=1.0)
            csv_path = write_per_client_metrics_csv(run_dir, [0.1, 0.2, 0.3])
            metrics = confirmatory.load_metrics(row)
            metrics["per_client_metrics_artifact"] = csv_path
            distribution = confirmatory.per_hospital_distribution(row, metrics)
        self.assertEqual(distribution["source"], "per_client_metrics_csv")
        self.assertEqual(distribution["n_clients"], 3)
        self.assertAlmostEqual(distribution["structural_mse"]["median"], 0.2)

    def test_stability_uses_primary_val_mse_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            row = manifest_row(tmpdir, "linear", "fedgda_s", 101, 1101)
            run_dir = write_run_metrics(tmpdir, row, primary_mse=1.0)
            write_mse_by_round(run_dir, n_rounds=60, base=2.0)
            stability = confirmatory.stability_for_run(row)
        self.assertEqual(stability["metric_column"], "primary_val_mse")
        self.assertAlmostEqual(stability["tail_mean"], 2.0)
        self.assertFalse(stability["diverged"])

    def test_non_confirmatory_rows_are_filtered_out(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = build_full_matrix(tmpdir, ogda_offset_by_g0={"linear": -0.5, "interaction": -0.5, "mlp": -0.5})
            # Add centralized / ablation rows that should be ignored.
            extra_row = manifest_row(tmpdir, "linear", "gda_d", 101, 1101, role="centralized_baseline")
            write_run_metrics(tmpdir, extra_row, primary_mse=99.0)
            rows.append(extra_row)
            result = confirmatory.run_analysis(rows)
        # If the centralized row leaked in, "gda_d" would appear as a method
        # under g0=linear alongside fedgda_s/fedogda_s.
        self.assertEqual(set(result["summary"]["linear"].keys()), {"fedgda_s", "fedogda_s"})

    def test_no_test_field_is_ever_used_for_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = build_full_matrix(tmpdir, ogda_offset_by_g0={"linear": -0.5, "interaction": -0.5, "mlp": -0.5})
            result = confirmatory.run_analysis(rows)
        self.assertEqual(result["test_fields_used_for_selection"], [])
        self.assertGreater(len(result["test_fields_read_for_final_reporting_only"]), 0)

    def test_cli_main_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = build_full_matrix(tmpdir, ogda_offset_by_g0={"linear": -0.5, "interaction": -0.5, "mlp": -0.5})
            manifest_path = os.path.join(tmpdir, "final_manifest.json")
            with open(manifest_path, "w") as handle:
                json.dump(rows, handle)
            out_dir = os.path.join(tmpdir, "confirmatory_report")
            exit_code = confirmatory.main(["--manifest", manifest_path, "--out", out_dir])
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "summary.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "README.md")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "per_group_primary_endpoint.csv")))
            with open(os.path.join(out_dir, "summary.json")) as handle:
                written = json.load(handle)
            self.assertEqual(written["pairwise_primary_endpoint"]["pooled"]["verdict"], "fedogda_favored")


if __name__ == "__main__":
    unittest.main()
