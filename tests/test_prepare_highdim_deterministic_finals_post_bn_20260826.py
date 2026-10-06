"""Tests for scripts/prepare_highdim_deterministic_finals_post_bn_20260826.py.

Synthetic V4-winners + stability-results + screen-manifest-template
fixtures -- V4 has not run yet (Phase 5 is out of scope), so this exercises
the exact-reuse accounting frozen in PROTOCOL_DECISION_ADDENDUM_20260826.md
SS6 without depending on real data.

Also feeds the generated finals_launch_manifest.csv through the real
run_manifest.py --dry-run: the launch manifest must contain only the 132
new trajectories, every column build_config() needs, and none of the 48
reused ones (which run_manifest.py has no concept of reusing).
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prepare_highdim_deterministic_finals_post_bn_20260826 as prepare_finals  # noqa: E402

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
            cells[f"{dataset}|{method}"] = {
                "dataset": dataset, "method": method,
                "winner": {
                    "lr": 0.001 * (i + 1), "cm": 5.0 + j,
                    "run_ids": {
                        "0": f"v4_{dataset}_{method}_seed0",
                        "1": f"v4_{dataset}_{method}_seed1",
                        "2": f"v4_{dataset}_{method}_seed2",
                    },
                },
            }
    return {"status": "complete", "alpha": 0.5, "seeds": [0, 1, 2], "cells": cells}


def _fake_stability(tmp: Path, *, all_pass: bool):
    manifest_path = tmp / "stability_manifest.csv"
    fieldnames = ["run_id", "final_result_dir"]
    rows = []
    cells = {}
    for dataset in DATASETS:
        for method in METHODS:
            cell_name = f"{dataset}|{method}"
            run_id = f"stability_{dataset}_{method}"
            rows.append({"run_id": run_id, "final_result_dir": f"results/stability/{run_id}"})
            cells[cell_name] = {
                "dataset": dataset, "method": method, "run_id": run_id,
                "outcome": "pass" if all_pass else (
                    "retune_required" if cell_name == f"{DATASETS[0]}|{METHODS[0]}" else "pass"
                ),
            }
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    results_path = tmp / "stability_results.json"
    results_path.write_text(json.dumps({"status": "complete", "cells": cells}))
    return results_path, manifest_path


class PrepareFinalsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.winners_path = self.tmp / "v4_winners.json"
        self.winners_path.write_text(json.dumps(_fake_winners()))
        self.screen_manifest_path = self.tmp / "screen_manifest.csv"
        _write_screen_manifest(self.screen_manifest_path)

    def test_exact_reuse_accounting_matches_the_frozen_addendum(self):
        results_path, stability_manifest_path = _fake_stability(self.tmp, all_pass=True)
        output_dir = self.tmp / "finals"
        result = prepare_finals.prepare(
            self.winners_path, results_path, stability_manifest_path,
            self.screen_manifest_path, output_dir,
        )
        self.assertEqual(result["total"], 180)
        self.assertEqual(result["reused"], 48)  # 36 V4 + 12 stability
        self.assertEqual(result["new"], 132)  # 48 + 24 + 60

        with (output_dir / "finals_launch_manifest.csv").open(newline="") as handle:
            launch_rows = list(csv.DictReader(handle))
        self.assertEqual(len(launch_rows), 132)  # only new trajectories are ever launched
        self.assertEqual(len({row["run_id"] for row in launch_rows}), 132)
        self.assertEqual(len({row["final_result_dir"] for row in launch_rows}), 132)

        ledger = json.loads((output_dir / "finals_evidence_ledger.json").read_text())
        self.assertEqual(ledger["total_trajectories"], 180)
        self.assertEqual(len(ledger["trajectories"]), 180)
        self.assertEqual(len({t["run_id"] for t in ledger["trajectories"]}), 180)

        by_cell = {}
        for entry in ledger["trajectories"]:
            by_cell.setdefault((entry["dataset"], entry["method"]), []).append(entry)
        self.assertEqual(len(by_cell), 12)
        for cell_entries in by_cell.values():
            self.assertEqual(len(cell_entries), 15)  # 3 alphas x 5 seeds
            alpha_seed_pairs = Counter((e["alpha"], e["seed"]) for e in cell_entries)
            self.assertEqual(len(alpha_seed_pairs), 15)
            reused = [e for e in cell_entries if e["reused"]]
            self.assertEqual(len(reused), 4)  # 3 alpha0.5 seeds 0-2 + 1 alpha0.1 seed 0

    def test_launch_manifest_rows_carry_every_column_run_manifest_needs(self):
        results_path, stability_manifest_path = _fake_stability(self.tmp, all_pass=True)
        output_dir = self.tmp / "finals"
        prepare_finals.prepare(
            self.winners_path, results_path, stability_manifest_path,
            self.screen_manifest_path, output_dir,
        )
        with (output_dir / "finals_launch_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = (
            "epochs", "client_num_in_total", "client_num_per_round", "batch_size",
            "partition_alpha", "partition_method", "comm_round",
        )
        for row in rows:
            for field in required:
                self.assertIn(field, row)
                self.assertNotEqual(row[field], "")
        # This campaign's "alpha" IS the Dirichlet partition concentration,
        # and finals rows span alpha 0.1/0.5/1.0 -- partition_alpha must
        # track each row's own alpha, not stay pinned at a single value.
        self.assertTrue(all(row["partition_alpha"] == row["alpha"] for row in rows))
        self.assertEqual(
            {row["alpha"] for row in rows}, {"0.1", "0.5", "1"},
        )
        self.assertTrue(all(row["comm_round"] == "500" for row in rows))

    def test_launch_manifest_is_launchable_by_real_run_manifest_dry_run(self):
        results_path, stability_manifest_path = _fake_stability(self.tmp, all_pass=True)
        output_dir = self.tmp / "finals"
        prepare_finals.prepare(
            self.winners_path, results_path, stability_manifest_path,
            self.screen_manifest_path, output_dir,
        )
        manifest_path = output_dir / "finals_launch_manifest.csv"
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
        self.assertEqual(summary["launchable"], 132)
        self.assertEqual(summary["skipped_unlaunchable"], 0)

    def test_retune_required_cell_blocks_the_whole_matrix_without_retune_results(self):
        results_path, stability_manifest_path = _fake_stability(self.tmp, all_pass=False)
        with self.assertRaisesRegex(ValueError, "retune"):
            prepare_finals.prepare(
                self.winners_path, results_path, stability_manifest_path,
                self.screen_manifest_path, self.tmp / "finals",
            )

    def test_retuned_cell_uses_the_retuned_lr_cm_for_all_five_alpha0p1_seeds(self):
        # SS9.3: a retuned cell's original failed stability run does not
        # count as a final winner trajectory -- ALL 5 alpha=0.1 seeds become
        # new rows at the RETUNED (lr, cm), none reused; alpha=0.5/alpha=1.0
        # are untouched (still the V4-winner's lr/cm, still reused where
        # applicable).
        results_path, stability_manifest_path = _fake_stability(self.tmp, all_pass=False)
        retuned_cell = f"{DATASETS[0]}|{METHODS[0]}"  # matches _fake_stability's flagged cell
        retune_results_path = self.tmp / "retune_results.json"
        retune_results_path.write_text(json.dumps({
            "status": "complete",
            "stage": "promote",
            "cells": {retuned_cell: {"winner": {"lr": 0.5, "cm": 99.0, "run_id": "retune_winner"}}},
        }))
        output_dir = self.tmp / "finals"
        result = prepare_finals.prepare(
            self.winners_path, results_path, stability_manifest_path,
            self.screen_manifest_path, output_dir, retune_results_path,
        )
        # One fewer reused (the retuned cell's alpha=0.1 seed-0 is no longer
        # reused), one more new (it becomes a 5th new alpha=0.1 row instead
        # of 4 new + 1 reused).
        self.assertEqual(result["total"], 180)
        self.assertEqual(result["reused"], 47)
        self.assertEqual(result["new"], 133)

        ledger = json.loads((output_dir / "finals_evidence_ledger.json").read_text())
        dataset, method = retuned_cell.split("|", 1)
        alpha0p1_entries = [
            e for e in ledger["trajectories"]
            if e["dataset"] == dataset and e["method"] == method and e["alpha"] == 0.1
        ]
        self.assertEqual(len(alpha0p1_entries), 5)
        self.assertTrue(all(not e["reused"] for e in alpha0p1_entries))
        self.assertEqual({e["seed"] for e in alpha0p1_entries}, {0, 1, 2, 3, 4})

        with (output_dir / "finals_launch_manifest.csv").open(newline="") as handle:
            launch_rows = list(csv.DictReader(handle))
        retuned_launch_rows = [
            r for r in launch_rows if r["dataset"] == dataset and r["method"] == method and r["alpha"] == "0.1"
        ]
        self.assertEqual(len(retuned_launch_rows), 5)
        self.assertTrue(all(r["learning_rate"] == "0.5" for r in retuned_launch_rows))
        self.assertTrue(all(r["critic_multiplier"] == "99" for r in retuned_launch_rows))

        # alpha=0.5 (untouched by retuning: 3 reused V4 seeds + 2 new seeds,
        # all still at the V4 winner's lr/cm).
        alpha0p5_entries = [
            e for e in ledger["trajectories"]
            if e["dataset"] == dataset and e["method"] == method and e["alpha"] == 0.5
        ]
        self.assertEqual(len(alpha0p5_entries), 5)
        self.assertEqual(sum(1 for e in alpha0p5_entries if e["reused"]), 3)

    def test_missing_retune_result_for_a_flagged_cell_blocked(self):
        results_path, stability_manifest_path = _fake_stability(self.tmp, all_pass=False)
        retune_results_path = self.tmp / "retune_results.json"
        retune_results_path.write_text(json.dumps({"status": "complete", "stage": "promote", "cells": {}}))
        with self.assertRaisesRegex(ValueError, "missing these cells"):
            prepare_finals.prepare(
                self.winners_path, results_path, stability_manifest_path,
                self.screen_manifest_path, self.tmp / "finals", retune_results_path,
            )

    def test_reused_alpha0p5_entries_use_the_real_v4_run_id_not_an_alias(self):
        results_path, stability_manifest_path = _fake_stability(self.tmp, all_pass=True)
        output_dir = self.tmp / "finals"
        prepare_finals.prepare(
            self.winners_path, results_path, stability_manifest_path,
            self.screen_manifest_path, output_dir,
        )
        ledger = json.loads((output_dir / "finals_evidence_ledger.json").read_text())
        target = [
            e for e in ledger["trajectories"]
            if e["alpha"] == 0.5 and e["seed"] == 0
            and e["dataset"] == DATASETS[0] and e["method"] == METHODS[0]
        ]
        self.assertEqual(len(target), 1)
        # Must be the REAL V4 run_id verbatim -- not a "det_finals_postbn_..."
        # alias -- so validate_artifacts()'s exact run_id match succeeds
        # against the real, unmodified V4 directory.
        expected_run_id = f"v4_{DATASETS[0]}_{METHODS[0]}_seed0"
        self.assertEqual(target[0]["run_id"], expected_run_id)
        self.assertEqual(target[0]["source_stage"], "psi_adjudication_post_bn_v4")
        self.assertTrue(target[0]["reused"])
        # And it must NOT appear in the launch manifest -- reused rows are
        # never handed to run_manifest.py.
        with (output_dir / "finals_launch_manifest.csv").open(newline="") as handle:
            launch_run_ids = {row["run_id"] for row in csv.DictReader(handle)}
        self.assertNotIn(expected_run_id, launch_run_ids)

    def test_reused_alpha0p1_seed0_uses_the_real_stability_run_id(self):
        results_path, stability_manifest_path = _fake_stability(self.tmp, all_pass=True)
        output_dir = self.tmp / "finals"
        prepare_finals.prepare(
            self.winners_path, results_path, stability_manifest_path,
            self.screen_manifest_path, output_dir,
        )
        ledger = json.loads((output_dir / "finals_evidence_ledger.json").read_text())
        target = [
            e for e in ledger["trajectories"]
            if e["alpha"] == 0.1 and e["seed"] == 0
            and e["dataset"] == DATASETS[0] and e["method"] == METHODS[0]
        ]
        self.assertEqual(len(target), 1)
        self.assertEqual(target[0]["run_id"], f"stability_{DATASETS[0]}_{METHODS[0]}")
        self.assertEqual(target[0]["source_stage"], "deterministic_stability_alpha0p1_20260826")
        self.assertTrue(target[0]["reused"])


if __name__ == "__main__":
    unittest.main()
