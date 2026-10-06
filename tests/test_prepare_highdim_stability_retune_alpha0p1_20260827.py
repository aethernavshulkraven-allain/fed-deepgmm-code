"""Tests for scripts/prepare_highdim_stability_retune_alpha0p1_20260827.py --
the SS9.1 fallback branch that must exist before the stability stage is
launched, even though (real current state) no cell has ever needed it yet.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prepare_highdim_stability_retune_alpha0p1_20260827 as prepare_retune  # noqa: E402

DATASETS = ["femnist_x", "femnist_z", "femnist_xz", "cifar10_x", "cifar10_z", "cifar10_xz"]
METHODS = ("fedgda_d", "fedogda_d")
METHOD_OPTIMIZERS = {"fedgda_d": "sgd", "fedogda_d": "ogda"}

SCREEN_FIELDNAMES = [
    "run_id", "protocol_version", "run_group", "training_scope", "method", "method_label",
    "dataset", "seed", "alpha", "output_root", "final_result_dir", "implementation_status",
    "run_status", "preflight_required", "preflight_status", "model", "federated_optimizer",
    "client_optimizer", "client_num_in_total", "client_num_per_round", "comm_round", "epochs",
    "batch_size", "partition_method", "partition_alpha", "data_cache_dir", "learning_rate",
    "learning_rate_status", "weight_decay", "critic_multiplier", "server_learning_rate",
    "gradient_clip_norm", "simple_model_selection_epochs", "f_history_model_selection_epochs",
    "model_selection_batch_size", "using_gpu", "gpu_id", "notes", "auxiliary_regression",
    "auxiliary_regression_epochs", "objective_lambda_1", "append_round_csv",
    "periodic_checkpoint_interval", "log_test_mse_by_round", "test_mse_used_for_selection",
    "selection_metric_source", "objective_mode", "aggregation_weighting", "server_buffer_policy",
    "source_manifest", "source_run_id",
]


def _screen_row(dataset, method, lr, cm, index):
    run_id = f"det_screen_postbn_{dataset}_{method}_seed0_alpha0p5_lr{index}_cm{index}"
    return {
        "run_id": run_id, "protocol_version": "highdim_deterministic_screen_post_bn_v1",
        "run_group": "highdim_deterministic_screen_post_bn_20260822", "training_scope": "federated",
        "method": method, "method_label": "FedGDA-D" if method == "fedgda_d" else "FedOGDA-D",
        "dataset": dataset, "seed": "0", "alpha": "0.5",
        "output_root": "results/highdim_deterministic_screen_post_bn_20260822",
        "final_result_dir": f"results/highdim_deterministic_screen_post_bn_20260822/{dataset}/{method}/seed_0/{run_id}",
        "implementation_status": "ready", "run_status": "not_started", "preflight_required": "True",
        "preflight_status": "bn_buffer_diagnostic_certified", "model": "lr", "federated_optimizer": "FedAvg",
        "client_optimizer": METHOD_OPTIMIZERS[method], "client_num_in_total": "10",
        "client_num_per_round": "10", "comm_round": "150", "epochs": "3", "batch_size": "0",
        "partition_method": "hetero", "partition_alpha": "0.5", "data_cache_dir": "data",
        "learning_rate": str(lr), "learning_rate_status": "screen_candidate", "weight_decay": "0.001",
        "critic_multiplier": str(cm), "server_learning_rate": "1.5", "gradient_clip_norm": "1.0",
        "simple_model_selection_epochs": "100", "f_history_model_selection_epochs": "60",
        "model_selection_batch_size": "200", "using_gpu": "True", "gpu_id": "",
        "notes": "fixture", "auxiliary_regression": "False", "auxiliary_regression_epochs": "0",
        "objective_lambda_1": "0.1", "append_round_csv": "True", "periodic_checkpoint_interval": "0",
        "log_test_mse_by_round": "False", "test_mse_used_for_selection": "False",
        "selection_metric_source": "validation", "objective_mode": "legacy",
        "aggregation_weighting": "sample_size", "server_buffer_policy": "direct_client_aggregate",
        "source_manifest": "fixture", "source_run_id": "fixture",
    }


def _write_screen_manifest(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCREEN_FIELDNAMES)
        writer.writeheader()
        for dataset in DATASETS:
            for method in METHODS:
                # 3 candidate rows per cell, matching the real screen's
                # varying per-cell row counts (7-11 in reality; 3 here for
                # a fast test fixture).
                for i in range(3):
                    writer.writerow(_screen_row(dataset, method, 0.01 * (i + 1), 5.0 * (i + 1), i))


def _fake_stability_results(*, retune_cells):
    cells = {}
    for dataset in DATASETS:
        for method in METHODS:
            cell_name = f"{dataset}|{method}"
            cells[cell_name] = {
                "dataset": dataset, "method": method,
                "outcome": "retune_required" if cell_name in retune_cells else "pass",
            }
    return {"status": "complete", "cells": cells}


class PrepareRetuneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.screen_manifest_path = self.tmp / "screen_manifest.csv"
        _write_screen_manifest(self.screen_manifest_path)

    def _write_stability(self, retune_cells) -> Path:
        path = self.tmp / "stability_results.json"
        path.write_text(json.dumps(_fake_stability_results(retune_cells=retune_cells)))
        return path

    def test_no_retune_required_is_rejected(self):
        stability_path = self._write_stability(retune_cells=set())
        with self.assertRaisesRegex(ValueError, "no cell requires retuning"):
            prepare_retune.prepare(stability_path, self.screen_manifest_path, self.tmp / "out")

    def test_generates_the_flagged_cells_exact_grid(self):
        flagged = {"femnist_z|fedgda_d", "cifar10_x|fedogda_d"}
        stability_path = self._write_stability(retune_cells=flagged)
        output_dir = self.tmp / "out"
        result = prepare_retune.prepare(stability_path, self.screen_manifest_path, output_dir)
        self.assertEqual(set(result["retune_cells"]), flagged)
        self.assertEqual(result["runs"], 6)  # 2 cells x 3 candidates each
        with (output_dir / "retune_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["alpha"] == "0.1" for row in rows))
        # This campaign's "alpha" IS the Dirichlet partition concentration,
        # so partition_alpha must move to 0.1 with it rather than staying
        # pinned at the inherited screen-template value of 0.5.
        self.assertTrue(all(row["partition_alpha"] == "0.1" for row in rows))
        self.assertTrue(all(row["seed"] == "0" for row in rows))
        self.assertTrue(all(row["comm_round"] == "150" for row in rows))
        cells_seen = {(row["dataset"], row["method"]) for row in rows}
        self.assertEqual(cells_seen, {("femnist_z", "fedgda_d"), ("cifar10_x", "fedogda_d")})
        # Unflagged cells are never touched.
        self.assertNotIn("femnist_x", [row["dataset"] for row in rows if row["method"] == "fedogda_d"] or [])

    def test_grid_reuses_exact_lr_cm_pairs_from_the_screen_not_a_new_grid(self):
        flagged = {"femnist_z|fedgda_d"}
        stability_path = self._write_stability(retune_cells=flagged)
        output_dir = self.tmp / "out"
        prepare_retune.prepare(stability_path, self.screen_manifest_path, output_dir)
        with (output_dir / "retune_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        actual_pairs = {(row["learning_rate"], row["critic_multiplier"]) for row in rows}
        expected_pairs = {(f"{0.01 * (i + 1):g}", f"{5.0 * (i + 1):g}") for i in range(3)}
        self.assertEqual(actual_pairs, expected_pairs)

    def test_generated_manifest_is_launchable_by_real_run_manifest_dry_run(self):
        flagged = {"femnist_z|fedgda_d", "cifar10_xz|fedogda_d"}
        stability_path = self._write_stability(retune_cells=flagged)
        output_dir = self.tmp / "out"
        prepare_retune.prepare(stability_path, self.screen_manifest_path, output_dir)
        manifest_path = output_dir / "retune_manifest.csv"
        result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "run_manifest.py"),
                "--manifest", str(manifest_path),
                "--config-dir", str(self.tmp / "generated_configs"),
                "--output-root", str(self.tmp / "results"),
                "--dry-run",
            ],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
        json_start = result.stdout.rindex("{\n")
        summary = json.loads(result.stdout[json_start:])
        self.assertEqual(summary["launchable"], 6)
        self.assertEqual(summary["skipped_unlaunchable"], 0)

    def test_incomplete_stability_results_blocked(self):
        path = self.tmp / "stability_results.json"
        path.write_text(json.dumps({"status": "in_progress", "cells": {}}))
        with self.assertRaisesRegex(ValueError, "absent or incomplete"):
            prepare_retune.prepare(path, self.screen_manifest_path, self.tmp / "out")

    def test_real_stability_results_do_not_exist_yet(self):
        # Ground truth for the current real repo state: stability has not run.
        self.assertFalse((
            REPO_ROOT / "experiments/highdim_coauthor_protocol_v1"
            / "deterministic_stability_alpha0p1_20260826/stability_results.json"
        ).exists())


if __name__ == "__main__":
    unittest.main()
