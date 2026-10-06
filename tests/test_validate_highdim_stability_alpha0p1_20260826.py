"""Tests for scripts/validate_highdim_stability_alpha0p1_20260826.py -- the
alpha=0.1 stability escape hatch (closeout plan SS4.7 / SS9.1), covering both
frozen failure conditions: divergence and the constant-predictor test."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_highdim_stability_alpha0p1_20260826 as validator  # noqa: E402

COMM_ROUND = validator.STABILITY_COMM_ROUND
DATASETS = [f"fixture{i}" for i in range(6)]
METHODS = ("fedgda_d", "fedogda_d")
FIELDNAMES = [
    "run_id", "dataset", "method", "seed", "alpha", "learning_rate",
    "critic_multiplier", "client_optimizer", "comm_round", "final_result_dir",
    "server_buffer_policy",
]


def _write_curve(run_dir: Path, val_mse_series, diverged=False, nonfinite_round=None) -> None:
    with (run_dir / "mse_by_round.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "round", "train_mse", "val_mse", "primary_val_mse", "equal_client_val_mse",
            "train_moment_violation", "val_moment_violation", "gmm_train_objective",
            "gmm_val_objective", "gmm_eval", "g_bn_min_running_var", "f_bn_min_running_var",
            "finite", "diverged",
        ])
        for i in range(COMM_ROUND):
            mse = val_mse_series[i]
            row_finite = "True"
            row_diverged = "False"
            if nonfinite_round is not None and i == nonfinite_round:
                mse = float("nan")
                row_finite = "False"
            if diverged and i == COMM_ROUND - 1:
                row_diverged = "True"
            writer.writerow([
                i, mse, mse, mse, "", 0.1, 0.1, 0.1, 0.1, 1.0, "", "", row_finite, row_diverged,
            ])


def write_run(run_dir: Path, *, run_id, val_mse_series, val_target_variance,
              dataset="fixture", method="fedgda_d", diverged=False, nonfinite_round=None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "best_validation.pt").write_bytes(b"fake")
    (run_dir / "checkpoints" / "final.pt").write_bytes(b"fake")
    (run_dir / "predictions.npz").write_bytes(b"fake")
    (run_dir / "effective_config.json").write_text(json.dumps({
        "dataset": dataset, "variant": method, "run_id": run_id,
        "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
        "random_seed": 0, "comm_round": COMM_ROUND,
        "learning_rate": 0.01, "critic_multiplier": 5.0,
        "server_buffer_policy": "direct_client_aggregate",
        "test_mse_used_for_selection": False, "selection_metric_source": "validation",
        "log_test_mse_by_round": False,
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "diverged": diverged, "best_gmm_eval": 1.0, "best_validation_mse": min(val_mse_series),
        "run_status": "completed", "rounds_completed": COMM_ROUND,
        "server_buffer_policy": "direct_client_aggregate",
        "nonfinite_first_round": nonfinite_round,
        "nonfinite_diagnostics": ([{"round": nonfinite_round}] if nonfinite_round is not None else []),
        "val_target_variance": val_target_variance,
    }))
    _write_curve(run_dir, val_mse_series, diverged=diverged, nonfinite_round=nonfinite_round)


class ClassifyStabilityRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _row(self, run_id, run_dir):
        return {
            "run_id": run_id, "dataset": "fixture", "method": "fedgda_d",
            "client_optimizer": "sgd",
            "final_result_dir": str(run_dir), "server_buffer_policy": "direct_client_aggregate",
        }

    def test_beats_constant_predictor_passes(self):
        run_dir = self.tmp / "pass_run"
        # last-50 mean = 0.1, well under the constant-predictor baseline of 1.0.
        series = [1.0] * (COMM_ROUND - 50) + [0.1] * 50
        write_run(run_dir, run_id="pass_run", val_mse_series=series, val_target_variance=1.0)
        result = validator.classify_stability_run(run_dir, self._row("pass_run", run_dir))
        self.assertEqual(result["outcome"], "pass")
        self.assertAlmostEqual(result["last50_val_mse"], 0.1)

    def test_worse_than_constant_predictor_requires_retune(self):
        run_dir = self.tmp / "bad_run"
        # last-50 mean = 2.0, worse than the constant-predictor baseline of 1.0.
        series = [0.1] * (COMM_ROUND - 50) + [2.0] * 50
        write_run(run_dir, run_id="bad_run", val_mse_series=series, val_target_variance=1.0)
        result = validator.classify_stability_run(run_dir, self._row("bad_run", run_dir))
        self.assertEqual(result["outcome"], "retune_required")
        self.assertIn("constant-predictor", result["reason"])

    def test_exactly_equal_to_constant_predictor_requires_retune(self):
        # Not strictly better than the constant predictor -> not a pass.
        run_dir = self.tmp / "tie_run"
        series = [1.0] * COMM_ROUND
        write_run(run_dir, run_id="tie_run", val_mse_series=series, val_target_variance=1.0)
        result = validator.classify_stability_run(run_dir, self._row("tie_run", run_dir))
        self.assertEqual(result["outcome"], "retune_required")

    def test_divergence_requires_retune_regardless_of_constant_predictor(self):
        run_dir = self.tmp / "diverged_run"
        series = [0.01] * COMM_ROUND  # would otherwise easily pass
        write_run(
            run_dir, run_id="diverged_run", val_mse_series=series,
            val_target_variance=1.0, nonfinite_round=200,
        )
        result = validator.classify_stability_run(run_dir, self._row("diverged_run", run_dir))
        self.assertEqual(result["outcome"], "retune_required")
        self.assertIn("diverged", result["reason"])
        # Divergent runs are classified without needing a valid last50/constant
        # predictor computation.
        self.assertIsNone(result["last50_val_mse"])


class ValidateStabilityManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.rows = []

    def _add_cell(self, dataset, method, *, passes: bool):
        run_id = f"stability_{dataset}_{method}"
        run_dir = self.tmp / "results" / dataset / method / run_id
        series = (
            [1.0] * (COMM_ROUND - 50) + [0.1] * 50 if passes
            else [1.0] * (COMM_ROUND - 50) + [5.0] * 50
        )
        write_run(
            run_dir, run_id=run_id, val_mse_series=series, val_target_variance=1.0,
            dataset=dataset, method=method,
        )
        self.rows.append({
            "run_id": run_id, "dataset": dataset, "method": method, "seed": "0",
            "alpha": "0.1", "learning_rate": "0.01", "critic_multiplier": "5",
            "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
            "comm_round": str(COMM_ROUND), "final_result_dir": str(run_dir),
            "server_buffer_policy": "direct_client_aggregate",
        })

    def _write_manifest(self) -> Path:
        manifest_path = self.tmp / "stability_manifest.csv"
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)
        return manifest_path

    def test_all_12_cells_pass(self):
        for dataset in DATASETS:
            for method in METHODS:
                self._add_cell(dataset, method, passes=True)
        manifest_path = self._write_manifest()
        result = validator.validate_stability(manifest_path)
        self.assertTrue(result["all_cells_pass"])
        self.assertEqual(result["retune_required_cells"], [])
        self.assertEqual(len(result["cells"]), 12)

    def test_one_failing_cell_is_flagged_without_affecting_others(self):
        cells = [(d, m) for d in DATASETS for m in METHODS]
        for dataset, method in cells[:-1]:
            self._add_cell(dataset, method, passes=True)
        failing_dataset, failing_method = cells[-1]
        self._add_cell(failing_dataset, failing_method, passes=False)
        manifest_path = self._write_manifest()
        result = validator.validate_stability(manifest_path)
        self.assertFalse(result["all_cells_pass"])
        self.assertEqual(
            result["retune_required_cells"], [f"{failing_dataset}|{failing_method}"],
        )
        # Every other cell still individually resolved as passing.
        passing = [c for c in result["cells"].values() if c["outcome"] == "pass"]
        self.assertEqual(len(passing), 11)

    def test_wrong_row_count_rejected(self):
        for dataset in DATASETS[:5]:
            for method in METHODS:
                self._add_cell(dataset, method, passes=True)
        manifest_path = self._write_manifest()
        with self.assertRaisesRegex(ValueError, "exactly 12 rows"):
            validator.validate_stability(manifest_path)


if __name__ == "__main__":
    unittest.main()
