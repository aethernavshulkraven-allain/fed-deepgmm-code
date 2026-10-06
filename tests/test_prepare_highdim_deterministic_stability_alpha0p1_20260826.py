"""Tests for scripts/prepare_highdim_deterministic_stability_alpha0p1_20260826.py.

No real V4 winners exist yet (V4 has not been launched -- closeout plan
Phase 5 is out of scope for this pass), so these tests build a synthetic
v4_winners.json fixture matching the preparer's documented input contract,
plus a synthetic screen-manifest template (full real column set) each row
is built from.

Critically, this also feeds the generated manifest through the real
run_manifest.py --dry-run, not just prepare()'s own return value -- a
manifest missing a required column (epochs, client_num_in_total,
client_num_per_round, batch_size, partition_alpha, ...) would still report
a correct-looking row/hash count from prepare() alone while being unusable
by run_manifest.py.
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

import prepare_highdim_deterministic_stability_alpha0p1_20260826 as prepare_stability  # noqa: E402

DATASETS = ["femnist_x", "femnist_z", "femnist_xz", "cifar10_x", "cifar10_z", "cifar10_xz"]
METHODS = ("fedgda_d", "fedogda_d")
METHOD_OPTIMIZERS = {"fedgda_d": "sgd", "fedogda_d": "ogda"}

# Every column the real screen_manifest.csv carries, with realistic frozen-
# protocol values (mirrors a real row from
# deterministic_screen_post_bn_20260822/screen_manifest.csv).
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


def _screen_row(dataset: str, method: str) -> dict:
    run_id = f"det_screen_postbn_{dataset}_{method}_seed0_alpha0p5_lr0p003_cm1"
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
        "learning_rate": "0.003", "learning_rate_status": "screen_candidate", "weight_decay": "0.001",
        "critic_multiplier": "1", "server_learning_rate": "1.5", "gradient_clip_norm": "1.0",
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
                writer.writerow(_screen_row(dataset, method))


def _fake_winners() -> dict:
    cells = {}
    for i, dataset in enumerate(DATASETS):
        for j, method in enumerate(METHODS):
            cell_name = f"{dataset}|{method}"
            cells[cell_name] = {
                "dataset": dataset, "method": method,
                "winner": {
                    "lr": 0.001 * (i + 1), "cm": 5.0 + j,
                    "run_ids": {
                        "0": f"det_adjudicate_v4_{dataset}_{method}_seed0",
                        "1": f"det_adjudicate_v4_{dataset}_{method}_seed1",
                        "2": f"det_adjudicate_v4_{dataset}_{method}_seed2",
                    },
                },
            }
    return {"status": "complete", "alpha": 0.5, "seeds": [0, 1, 2], "cells": cells}


class PrepareStabilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.screen_manifest_path = self.tmp / "screen_manifest.csv"
        _write_screen_manifest(self.screen_manifest_path)

    def _write_winners(self, winners: dict) -> Path:
        path = self.tmp / "v4_winners.json"
        path.write_text(json.dumps(winners))
        return path

    def test_generates_exactly_12_rows_one_per_cell(self):
        winners_path = self._write_winners(_fake_winners())
        output_dir = self.tmp / "stability"
        result = prepare_stability.prepare(winners_path, self.screen_manifest_path, output_dir)
        self.assertEqual(result["runs"], 12)
        with (output_dir / "stability_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["alpha"] == "0.1" for row in rows))
        self.assertTrue(all(row["seed"] == "0" for row in rows))
        self.assertTrue(all(row["comm_round"] == "500" for row in rows))
        self.assertTrue(all(
            row["server_buffer_policy"] == "direct_client_aggregate" for row in rows
        ))
        self.assertEqual(len({row["run_id"] for row in rows}), 12)
        self.assertEqual(len({row["final_result_dir"] for row in rows}), 12)
        self.assertTrue((output_dir / "generated_artifact_hashes.json").exists())

    def test_rows_carry_every_column_run_manifest_needs(self):
        # The exact bug this test exists to catch: a hand-picked field subset
        # silently dropping columns build_config() requires without a default.
        winners_path = self._write_winners(_fake_winners())
        output_dir = self.tmp / "stability"
        prepare_stability.prepare(winners_path, self.screen_manifest_path, output_dir)
        with (output_dir / "stability_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = (
            "epochs", "client_num_in_total", "client_num_per_round", "batch_size",
            "partition_alpha", "partition_method", "comm_round",
        )
        for row in rows:
            for field in required:
                self.assertIn(field, row, f"missing column {field!r}")
                self.assertNotEqual(row[field], "", f"blank {field!r} in {row['run_id']}")
        # This campaign's "alpha" IS the Dirichlet partition concentration
        # (that's the whole point of an alpha=0.1 stability check), so the
        # executable partition_alpha must move together with the alpha
        # label, not stay pinned at the inherited screen-template value.
        self.assertTrue(all(row["partition_alpha"] == "0.1" for row in rows))
        self.assertTrue(all(row["alpha"] == "0.1" for row in rows))
        self.assertTrue(all(row["client_num_in_total"] == "10" for row in rows))
        self.assertTrue(all(row["client_num_per_round"] == "10" for row in rows))
        self.assertTrue(all(row["batch_size"] == "0" for row in rows))

    def test_generated_manifest_is_launchable_by_real_run_manifest_dry_run(self):
        # The actual regression test: feed the real generated manifest through
        # the real run_manifest.py --dry-run, not just prepare()'s return value.
        winners_path = self._write_winners(_fake_winners())
        output_dir = self.tmp / "stability"
        prepare_stability.prepare(winners_path, self.screen_manifest_path, output_dir)
        manifest_path = output_dir / "stability_manifest.csv"
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
        self.assertEqual(summary["launchable"], 12)
        self.assertEqual(summary["skipped_unlaunchable"], 0)

    def test_incomplete_v4_winners_blocked(self):
        winners = _fake_winners()
        winners["status"] = "in_progress"
        winners_path = self._write_winners(winners)
        with self.assertRaisesRegex(ValueError, "absent or incomplete"):
            prepare_stability.prepare(winners_path, self.screen_manifest_path, self.tmp / "stability")

    def test_wrong_alpha_blocked(self):
        winners = _fake_winners()
        winners["alpha"] = 0.1
        winners_path = self._write_winners(winners)
        with self.assertRaisesRegex(ValueError, "alpha=0.5"):
            prepare_stability.prepare(winners_path, self.screen_manifest_path, self.tmp / "stability")

    def test_missing_cell_blocked(self):
        winners = _fake_winners()
        del winners["cells"][next(iter(winners["cells"]))]
        winners_path = self._write_winners(winners)
        with self.assertRaisesRegex(ValueError, "exactly 12 cells"):
            prepare_stability.prepare(winners_path, self.screen_manifest_path, self.tmp / "stability")

    def test_missing_seed_run_id_blocked(self):
        winners = _fake_winners()
        first_cell = next(iter(winners["cells"].values()))
        del first_cell["winner"]["run_ids"]["1"]
        winners_path = self._write_winners(winners)
        with self.assertRaisesRegex(ValueError, "seeds 0, 1, 2"):
            prepare_stability.prepare(winners_path, self.screen_manifest_path, self.tmp / "stability")

    def test_no_template_for_cell_blocked(self):
        winners = _fake_winners()
        # A cell the screen manifest fixture has no template row for.
        winners["cells"]["nonexistent_dataset|fedgda_d"] = winners["cells"].pop(
            next(iter(winners["cells"]))
        )
        winners["cells"]["nonexistent_dataset|fedgda_d"]["dataset"] = "nonexistent_dataset"
        winners["cells"]["nonexistent_dataset|fedgda_d"]["method"] = "fedgda_d"
        winners_path = self._write_winners(winners)
        with self.assertRaisesRegex(ValueError, "no screen-manifest template"):
            prepare_stability.prepare(winners_path, self.screen_manifest_path, self.tmp / "stability")

    def test_real_v4_winners_do_not_exist_yet(self):
        # Ground truth for the current, real repo state: V4 has not run.
        self.assertFalse((
            REPO_ROOT / "experiments/highdim_coauthor_protocol_v1"
            / "psi_adjudication_post_bn_v4/v4_winners.json"
        ).exists())


if __name__ == "__main__":
    unittest.main()
