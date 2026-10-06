"""End-to-end tests for the alpha=0.1 retune fallback's Rank, Confirm, and
Promote stages (closeout plan SS9.1 escape hatch; doe_review_and_revised_
grid.md Part VI/VII's Screen->Rank->Confirm->Promote fallback). Covers:

- prepare_highdim_stability_retune_rank_alpha0p1_20260827.py: 2 rows/cell
  (Screen's top-2), seed 0, 500 rounds, alpha AND partition_alpha both 0.1.
- prepare_highdim_stability_retune_confirm_alpha0p1_20260827.py: 4 rows/cell
  (top-2 x seeds {1,2}), 500 rounds -- seed 0 is never re-launched.
- score_highdim_stability_retune_promote_alpha0p1_20260827.py: combines
  Rank's seed-0 run with Confirm's seed-{1,2} runs into a 3-seed Candidate
  per top-2 arm and applies the frozen median rule to freeze one winner,
  in the exact shape prepare_highdim_deterministic_finals_post_bn_20260826.py's
  --retune-results already expects.

Both preparers' generated manifests are also fed through the real
run_manifest.py --dry-run, not just their own return values.
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

import prepare_highdim_stability_retune_confirm_alpha0p1_20260827 as prepare_confirm  # noqa: E402
import prepare_highdim_stability_retune_rank_alpha0p1_20260827 as prepare_rank  # noqa: E402
import score_highdim_stability_retune_promote_alpha0p1_20260827 as promote_scorer  # noqa: E402

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
METHOD_OPTIMIZERS = {"fedgda_d": "sgd", "fedogda_d": "ogda"}


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


def _write_screen_manifest(path: Path, cells) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCREEN_FIELDNAMES)
        writer.writeheader()
        for dataset, method in cells:
            writer.writerow(_screen_row(dataset, method))


def _screen_results(cells_top2: dict) -> dict:
    cells = {}
    for (dataset, method), top2 in cells_top2.items():
        cells[f"{dataset}|{method}"] = {
            "dataset": dataset, "method": method,
            "top2": top2,
            "eligible_candidates": len(top2),
            "terminal_candidates": 0,
            "boundary_detail": [],
        }
    return {"status": "complete", "stage": "screen", "cells": cells}


def write_run(run_dir: Path, *, dataset, method, run_id, seed, gmm_eval, val_mse,
              learning_rate, critic_multiplier, comm_round=500) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "best_validation.pt").write_bytes(b"fake")
    (run_dir / "checkpoints" / "final.pt").write_bytes(b"fake")
    (run_dir / "predictions.npz").write_bytes(b"fake")
    (run_dir / "effective_config.json").write_text(json.dumps({
        "dataset": dataset, "variant": method, "run_id": run_id,
        "client_optimizer": METHOD_OPTIMIZERS[method],
        "random_seed": seed, "comm_round": comm_round,
        "learning_rate": learning_rate, "critic_multiplier": critic_multiplier,
        "server_buffer_policy": "direct_client_aggregate",
        "test_mse_used_for_selection": False, "selection_metric_source": "validation",
        "log_test_mse_by_round": False,
        # Must match the screen-manifest fixture template's values -- these
        # are cross-checked against the real manifest row by
        # _validate_effective_config()/validate_artifacts().
        "client_num_in_total": 10, "client_num_per_round": 10, "batch_size": 0,
        "epochs": 3, "weight_decay": 0.001, "server_learning_rate": 1.5,
        "partition_alpha": 0.1,
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "diverged": False, "best_gmm_eval": gmm_eval, "best_validation_mse": val_mse,
        "run_status": "completed", "rounds_completed": comm_round,
        "server_buffer_policy": "direct_client_aggregate",
        "nonfinite_first_round": None, "nonfinite_diagnostics": [],
        "g_bn_min_running_var": 0.01, "f_bn_min_running_var": 0.01,
    }))
    with (run_dir / "mse_by_round.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "round", "train_mse", "val_mse", "primary_val_mse", "equal_client_val_mse",
            "train_moment_violation", "val_moment_violation", "gmm_train_objective",
            "gmm_val_objective", "gmm_eval", "g_bn_min_running_var", "f_bn_min_running_var",
            "finite", "diverged",
        ])
        for i in range(comm_round):
            writer.writerow([
                i, val_mse, val_mse, val_mse, "", 0.2, 0.2, 0.1, 0.1,
                gmm_eval, 0.01, 0.01, "True", "False",
            ])


class PrepareRankTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.screen_manifest_path = self.tmp / "screen_manifest.csv"
        _write_screen_manifest(self.screen_manifest_path, [("femnist_z", "fedgda_d")])
        self.screen_results_path = self.tmp / "screen_results.json"
        self.screen_results_path.write_text(json.dumps(_screen_results({
            ("femnist_z", "fedgda_d"): [
                {"lr": 0.01, "cm": 5.0, "gmm_eval": 2.0, "val_mse": 0.4, "run_id": "screen_a"},
                {"lr": 0.02, "cm": 10.0, "gmm_eval": 1.0, "val_mse": 0.5, "run_id": "screen_b"},
            ],
        })))

    def test_generates_two_rows_seed0_500_rounds_alpha0p1(self):
        output_dir = self.tmp / "rank"
        result = prepare_rank.prepare(self.screen_results_path, self.screen_manifest_path, output_dir)
        self.assertEqual(result["runs"], 2)
        with (output_dir / "rank_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["seed"] == "0" for row in rows))
        self.assertTrue(all(row["comm_round"] == "500" for row in rows))
        self.assertTrue(all(row["alpha"] == "0.1" for row in rows))
        # The exact bug class this campaign keeps re-introducing: alpha
        # relabeled without moving the executable partition_alpha with it.
        self.assertTrue(all(row["partition_alpha"] == "0.1" for row in rows))
        self.assertEqual({row["learning_rate"] for row in rows}, {"0.01", "0.02"})

    def test_generated_manifest_is_launchable_by_real_run_manifest_dry_run(self):
        output_dir = self.tmp / "rank"
        prepare_rank.prepare(self.screen_results_path, self.screen_manifest_path, output_dir)
        result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "run_manifest.py"),
                "--manifest", str(output_dir / "rank_manifest.csv"),
                "--config-dir", str(self.tmp / "generated_configs"),
                "--output-root", str(self.tmp / "results"),
                "--dry-run",
            ],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
        json_start = result.stdout.rindex("{\n")
        summary = json.loads(result.stdout[json_start:])
        self.assertEqual(summary["launchable"], 2)
        self.assertEqual(summary["skipped_unlaunchable"], 0)

    def test_wrong_stage_input_blocked(self):
        bad_path = self.tmp / "bad.json"
        bad_path.write_text(json.dumps({"status": "complete", "stage": "promote", "cells": {}}))
        with self.assertRaisesRegex(ValueError, "Screen stage"):
            prepare_rank.prepare(bad_path, self.screen_manifest_path, self.tmp / "rank2")


class PrepareConfirmTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.screen_manifest_path = self.tmp / "screen_manifest.csv"
        _write_screen_manifest(self.screen_manifest_path, [("femnist_z", "fedgda_d")])
        self.screen_results_path = self.tmp / "screen_results.json"
        self.screen_results_path.write_text(json.dumps(_screen_results({
            ("femnist_z", "fedgda_d"): [
                {"lr": 0.01, "cm": 5.0, "gmm_eval": 2.0, "val_mse": 0.4, "run_id": "screen_a"},
                {"lr": 0.02, "cm": 10.0, "gmm_eval": 1.0, "val_mse": 0.5, "run_id": "screen_b"},
            ],
        })))

    def test_generates_four_rows_seeds_1_and_2_only(self):
        output_dir = self.tmp / "confirm"
        result = prepare_confirm.prepare(self.screen_results_path, self.screen_manifest_path, output_dir)
        self.assertEqual(result["runs"], 4)
        with (output_dir / "confirm_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["seed"] for row in rows}, {"1", "2"})
        self.assertTrue(all(row["comm_round"] == "500" for row in rows))
        self.assertTrue(all(row["partition_alpha"] == "0.1" for row in rows))

    def test_generated_manifest_is_launchable_by_real_run_manifest_dry_run(self):
        output_dir = self.tmp / "confirm"
        prepare_confirm.prepare(self.screen_results_path, self.screen_manifest_path, output_dir)
        result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "run_manifest.py"),
                "--manifest", str(output_dir / "confirm_manifest.csv"),
                "--config-dir", str(self.tmp / "generated_configs"),
                "--output-root", str(self.tmp / "results"),
                "--dry-run",
            ],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
        json_start = result.stdout.rindex("{\n")
        summary = json.loads(result.stdout[json_start:])
        self.assertEqual(summary["launchable"], 4)
        self.assertEqual(summary["skipped_unlaunchable"], 0)


def _redirect_final_result_dirs(manifest_path: Path, artifacts_root: Path) -> None:
    """Rewrite final_result_dir to an absolute path under a per-test tmp dir.

    The real preparers write repo-relative final_result_dir values (matching
    production convention, where the promote/screen/etc. scorers resolve a
    relative path against REPO_ROOT). Left unredirected, a test's fake
    artifacts would land in the real repo's results/ tree at a path keyed
    only by run_id -- and every test in this class reuses the same
    (dataset, method, lr, cm) fixture values, so their run_ids collide and
    tests contaminate each other via leftover files across runs."""
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        row["final_result_dir"] = str(artifacts_root / row["run_id"])
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PromoteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.screen_manifest_path = self.tmp / "screen_manifest.csv"
        _write_screen_manifest(self.screen_manifest_path, [("femnist_z", "fedgda_d")])

    def _prepare_and_populate(self, top2, seed_scores):
        """seed_scores: {(lr, cm): {seed: (gmm_eval, val_mse)}}"""
        screen_results_path = self.tmp / "screen_results.json"
        screen_results_path.write_text(json.dumps(_screen_results({
            ("femnist_z", "fedgda_d"): top2,
        })))
        rank_dir = self.tmp / "rank"
        confirm_dir = self.tmp / "confirm"
        prepare_rank.prepare(screen_results_path, self.screen_manifest_path, rank_dir)
        prepare_confirm.prepare(screen_results_path, self.screen_manifest_path, confirm_dir)
        _redirect_final_result_dirs(rank_dir / "rank_manifest.csv", self.tmp / "artifacts")
        _redirect_final_result_dirs(confirm_dir / "confirm_manifest.csv", self.tmp / "artifacts")

        with (rank_dir / "rank_manifest.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                lr, cm = float(row["learning_rate"]), float(row["critic_multiplier"])
                gmm_eval, val_mse = seed_scores[(lr, cm)][0]
                write_run(
                    Path(row["final_result_dir"]), dataset=row["dataset"], method=row["method"],
                    run_id=row["run_id"], seed=0, gmm_eval=gmm_eval, val_mse=val_mse,
                    learning_rate=lr, critic_multiplier=cm,
                )
        with (confirm_dir / "confirm_manifest.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                lr, cm = float(row["learning_rate"]), float(row["critic_multiplier"])
                seed = int(row["seed"])
                gmm_eval, val_mse = seed_scores[(lr, cm)][seed]
                write_run(
                    Path(row["final_result_dir"]), dataset=row["dataset"], method=row["method"],
                    run_id=row["run_id"], seed=seed, gmm_eval=gmm_eval, val_mse=val_mse,
                    learning_rate=lr, critic_multiplier=cm,
                )
        return rank_dir, confirm_dir

    def test_promotes_highest_median_psi_candidate(self):
        top2 = [
            {"lr": 0.01, "cm": 5.0, "gmm_eval": 2.0, "val_mse": 0.4, "run_id": "screen_a"},
            {"lr": 0.02, "cm": 10.0, "gmm_eval": 1.0, "val_mse": 0.5, "run_id": "screen_b"},
        ]
        seed_scores = {
            (0.01, 5.0): {0: (5.0, 0.3), 1: (5.5, 0.3), 2: (4.5, 0.3)},  # median psi 5.0
            (0.02, 10.0): {0: (1.0, 0.5), 1: (1.2, 0.5), 2: (0.8, 0.5)},  # median psi 1.0
        }
        rank_dir, confirm_dir = self._prepare_and_populate(top2, seed_scores)
        result = promote_scorer.promote(
            rank_dir / "rank_manifest.csv", rank_dir / "rank_summary.json",
            confirm_dir / "confirm_manifest.csv", confirm_dir / "confirm_summary.json",
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["stage"], "promote")
        cell = result["cells"]["femnist_z|fedgda_d"]
        self.assertEqual(cell["winner"]["lr"], 0.01)
        self.assertEqual(cell["winner"]["cm"], 5.0)
        self.assertEqual(cell["outcome"], "promoted")

    def test_practical_tie_resolved_by_median_mse(self):
        top2 = [
            {"lr": 0.01, "cm": 5.0, "gmm_eval": 2.0, "val_mse": 0.4, "run_id": "screen_a"},
            {"lr": 0.02, "cm": 10.0, "gmm_eval": 1.9, "val_mse": 0.5, "run_id": "screen_b"},
        ]
        # Near-identical median Psi (practical tie) but candidate B has
        # lower median MSE, so B should win the tiebreak.
        seed_scores = {
            (0.01, 5.0): {0: (5.0, 0.5), 1: (5.0, 0.5), 2: (5.0, 0.5)},
            (0.02, 10.0): {0: (5.0, 0.1), 1: (5.0, 0.1), 2: (5.0, 0.1)},
        }
        rank_dir, confirm_dir = self._prepare_and_populate(top2, seed_scores)
        result = promote_scorer.promote(
            rank_dir / "rank_manifest.csv", rank_dir / "rank_summary.json",
            confirm_dir / "confirm_manifest.csv", confirm_dir / "confirm_summary.json",
        )
        cell = result["cells"]["femnist_z|fedgda_d"]
        self.assertEqual(cell["winner"]["lr"], 0.02)
        self.assertEqual(cell["outcome"], "tie_resolved_by_mse")

    def test_both_candidates_ineligible_raises(self):
        top2 = [
            {"lr": 0.01, "cm": 5.0, "gmm_eval": 2.0, "val_mse": 0.4, "run_id": "screen_a"},
            {"lr": 0.02, "cm": 10.0, "gmm_eval": 1.0, "val_mse": 0.5, "run_id": "screen_b"},
        ]
        screen_results_path = self.tmp / "screen_results.json"
        screen_results_path.write_text(json.dumps(_screen_results({("femnist_z", "fedgda_d"): top2})))
        rank_dir, confirm_dir = self.tmp / "rank", self.tmp / "confirm"
        prepare_rank.prepare(screen_results_path, self.screen_manifest_path, rank_dir)
        prepare_confirm.prepare(screen_results_path, self.screen_manifest_path, confirm_dir)
        _redirect_final_result_dirs(rank_dir / "rank_manifest.csv", self.tmp / "artifacts")
        _redirect_final_result_dirs(confirm_dir / "confirm_manifest.csv", self.tmp / "artifacts")
        # Never write any run artifacts -- both candidates are "incomplete".
        with self.assertRaisesRegex(ValueError, "did not resolve to a promotable winner"):
            promote_scorer.promote(
                rank_dir / "rank_manifest.csv", rank_dir / "rank_summary.json",
                confirm_dir / "confirm_manifest.csv", confirm_dir / "confirm_summary.json",
            )

    def _standard_top2_and_scores(self):
        top2 = [
            {"lr": 0.01, "cm": 5.0, "gmm_eval": 2.0, "val_mse": 0.4, "run_id": "screen_a"},
            {"lr": 0.02, "cm": 10.0, "gmm_eval": 1.0, "val_mse": 0.5, "run_id": "screen_b"},
        ]
        seed_scores = {
            (0.01, 5.0): {0: (5.0, 0.3), 1: (5.5, 0.3), 2: (4.5, 0.3)},
            (0.02, 10.0): {0: (1.0, 0.5), 1: (1.2, 0.5), 2: (0.8, 0.5)},
        }
        return top2, seed_scores

    def test_wrong_rank_campaign_rejected(self):
        top2, seed_scores = self._standard_top2_and_scores()
        rank_dir, confirm_dir = self._prepare_and_populate(top2, seed_scores)
        summary_path = rank_dir / "rank_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["campaign"] = "some_other_campaign"
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "rank summary campaign"):
            promote_scorer.promote(
                rank_dir / "rank_manifest.csv", summary_path,
                confirm_dir / "confirm_manifest.csv", confirm_dir / "confirm_summary.json",
            )

    def test_wrong_confirm_alpha_rejected(self):
        top2, seed_scores = self._standard_top2_and_scores()
        rank_dir, confirm_dir = self._prepare_and_populate(top2, seed_scores)
        summary_path = confirm_dir / "confirm_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["alpha"] = 0.5
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "confirm summary alpha"):
            promote_scorer.promote(
                rank_dir / "rank_manifest.csv", rank_dir / "rank_summary.json",
                confirm_dir / "confirm_manifest.csv", summary_path,
            )

    def test_confirm_seeds_not_matching_1_and_2_rejected(self):
        top2, seed_scores = self._standard_top2_and_scores()
        rank_dir, confirm_dir = self._prepare_and_populate(top2, seed_scores)
        summary_path = confirm_dir / "confirm_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["seeds"] = [1, 3]
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "confirm summary seeds"):
            promote_scorer.promote(
                rank_dir / "rank_manifest.csv", rank_dir / "rank_summary.json",
                confirm_dir / "confirm_manifest.csv", summary_path,
            )

    def test_duplicate_cell_in_rank_plan_rejected(self):
        top2, seed_scores = self._standard_top2_and_scores()
        rank_dir, confirm_dir = self._prepare_and_populate(top2, seed_scores)
        summary_path = rank_dir / "rank_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["plan"] = summary["plan"] + [summary["plan"][0]]
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "duplicate rank plan cell"):
            promote_scorer.promote(
                rank_dir / "rank_manifest.csv", summary_path,
                confirm_dir / "confirm_manifest.csv", confirm_dir / "confirm_summary.json",
            )

    def test_extra_confirm_candidate_rejected(self):
        # Confirm claims 3 candidates for a cell rank only sent 2 to --
        # rank and confirm disagreeing on the top-2 must fail closed, not
        # silently use only the 2 that happen to match.
        top2, seed_scores = self._standard_top2_and_scores()
        rank_dir, confirm_dir = self._prepare_and_populate(top2, seed_scores)
        summary_path = confirm_dir / "confirm_summary.json"
        summary = json.loads(summary_path.read_text())
        cell_plan = summary["plan"][0]
        cell_plan["candidates"] = cell_plan["candidates"] + [
            {**cell_plan["candidates"][0], "lr": 0.09, "cm": 99.0}
        ]
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "confirm stage must carry exactly 2 candidates"):
            promote_scorer.promote(
                rank_dir / "rank_manifest.csv", rank_dir / "rank_summary.json",
                confirm_dir / "confirm_manifest.csv", summary_path,
            )

    def test_duplicate_rank_candidate_arm_rejected(self):
        # Rank names the same (lr, cm) twice instead of two distinct
        # candidates -- without this check, one confirm arm would silently
        # go unused while every count-based check still passes.
        top2, seed_scores = self._standard_top2_and_scores()
        rank_dir, confirm_dir = self._prepare_and_populate(top2, seed_scores)
        summary_path = rank_dir / "rank_summary.json"
        summary = json.loads(summary_path.read_text())
        cell_plan = summary["plan"][0]
        cell_plan["candidates"] = [cell_plan["candidates"][0], cell_plan["candidates"][0]]
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, r"rank \(lr, cm\) candidate"):
            promote_scorer.promote(
                summary_path.parent / "rank_manifest.csv", summary_path,
                confirm_dir / "confirm_manifest.csv", confirm_dir / "confirm_summary.json",
            )

    def test_rank_confirm_dataset_method_disagreement_rejected(self):
        top2, seed_scores = self._standard_top2_and_scores()
        rank_dir, confirm_dir = self._prepare_and_populate(top2, seed_scores)
        summary_path = confirm_dir / "confirm_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["plan"][0]["method"] = "fedogda_d"
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "disagrees with rank's"):
            promote_scorer.promote(
                rank_dir / "rank_manifest.csv", rank_dir / "rank_summary.json",
                confirm_dir / "confirm_manifest.csv", summary_path,
            )

    def test_confirm_candidate_missing_a_required_seed_rejected(self):
        top2, seed_scores = self._standard_top2_and_scores()
        rank_dir, confirm_dir = self._prepare_and_populate(top2, seed_scores)
        summary_path = confirm_dir / "confirm_summary.json"
        summary = json.loads(summary_path.read_text())
        cell_plan = summary["plan"][0]
        del cell_plan["candidates"][0]["run_ids"]["2"]
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "must carry run_ids for exactly seeds"):
            promote_scorer.promote(
                rank_dir / "rank_manifest.csv", rank_dir / "rank_summary.json",
                confirm_dir / "confirm_manifest.csv", summary_path,
            )

    def test_confirm_candidate_extra_seed_key_rejected(self):
        top2, seed_scores = self._standard_top2_and_scores()
        rank_dir, confirm_dir = self._prepare_and_populate(top2, seed_scores)
        summary_path = confirm_dir / "confirm_summary.json"
        summary = json.loads(summary_path.read_text())
        cell_plan = summary["plan"][0]
        cell_plan["candidates"][0]["run_ids"]["3"] = "some_extra_run_id"
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "must carry run_ids for exactly seeds"):
            promote_scorer.promote(
                rank_dir / "rank_manifest.csv", rank_dir / "rank_summary.json",
                confirm_dir / "confirm_manifest.csv", summary_path,
            )


if __name__ == "__main__":
    unittest.main()
