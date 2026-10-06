"""Regression tests for immutable terminal runs and partial-run recovery."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_manifest  # noqa: E402


def _row(run_id: str = "resume_safety") -> dict[str, str]:
    return {
        "run_id": run_id,
        "dataset": "femnist_x",
        "method": "fedogda_d",
        "seed": "0",
        "client_optimizer": "ogda",
        "client_num_in_total": "10",
        "client_num_per_round": "5",
        "comm_round": "2",
        "epochs": "1",
        "batch_size": "0",
        "learning_rate": "0.001",
        "critic_multiplier": "3",
        "weight_decay": "0",
        "server_learning_rate": "1.5",
        "partition_alpha": "0.5",
        "server_buffer_policy": "direct_client_aggregate",
    }


def _write_complete_run(run_dir: Path, row: dict[str, str], *, terminal: bool) -> None:
    run_dir.mkdir(parents=True)
    config = {
        "dataset": row["dataset"],
        "variant": row["method"],
        "run_id": row["run_id"],
        "client_optimizer": row["client_optimizer"],
        "random_seed": int(row["seed"]),
        "client_num_in_total": int(row["client_num_in_total"]),
        "client_num_per_round": int(row["client_num_per_round"]),
        "comm_round": int(row["comm_round"]),
        "epochs": int(row["epochs"]),
        "batch_size": int(row["batch_size"]),
        "learning_rate": float(row["learning_rate"]),
        "critic_multiplier": float(row["critic_multiplier"]),
        "weight_decay": float(row["weight_decay"]),
        "server_learning_rate": float(row["server_learning_rate"]),
        "partition_alpha": float(row["partition_alpha"]),
        "server_buffer_policy": row["server_buffer_policy"],
        "test_mse_used_for_selection": False,
        "selection_metric_source": "validation",
        "log_test_mse_by_round": False,
    }
    metrics = {
        "run_status": "completed",
        "rounds_completed": int(row["comm_round"]),
        "diverged": terminal,
        "failure_reason": "nonfinite critic output" if terminal else None,
        "test_mse_at_best_validation": 0.4,
        "final_test_mse": 0.5,
        "best_validation_round": 0,
        "server_buffer_policy": "direct_client_aggregate",
        "nonfinite_first_round": 0 if terminal else None,
        "nonfinite_diagnostics": ([{"round": 0}] if terminal else []),
        "g_bn_min_running_var": 0.01,
        "f_bn_min_running_var": None,
    }
    (run_dir / "effective_config.json").write_text(json.dumps(config))
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    fieldnames = [
        "round",
        "train_mse",
        "val_mse",
        "primary_val_mse",
        "equal_client_val_mse",
        "train_moment_violation",
        "val_moment_violation",
        "gmm_train_objective",
        "gmm_val_objective",
        "gmm_eval",
        "g_bn_min_running_var",
        "f_bn_min_running_var",
        "finite",
        "diverged",
    ]
    with (run_dir / "mse_by_round.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for round_index in range(int(row["comm_round"])):
            bad_round = terminal and round_index == 0
            writer.writerow({
                "round": round_index,
                "train_mse": 0.5,
                "val_mse": 0.5,
                "primary_val_mse": 0.5,
                "equal_client_val_mse": 0.5,
                "train_moment_violation": "nan" if bad_round else 0.2,
                "val_moment_violation": "nan" if bad_round else 0.2,
                "gmm_train_objective": "nan" if bad_round else 0.1,
                "gmm_val_objective": "nan" if bad_round else 0.1,
                "gmm_eval": 0.1,
                "g_bn_min_running_var": 0.01,
                "f_bn_min_running_var": "",
                "finite": not bad_round,
                "diverged": bad_round,
            })
    (run_dir / "predictions.npz").touch()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "best_validation.pt").touch()
    (run_dir / "checkpoints" / "final.pt").touch()


def _job(root: Path, row: dict[str, str]) -> run_manifest.Job:
    config = {
        "dataset": row["dataset"],
        "variant": row["method"],
        "run_id": row["run_id"],
        "client_optimizer": row["client_optimizer"],
        "random_seed": int(row["seed"]),
        "client_num_in_total": int(row["client_num_in_total"]),
        "client_num_per_round": int(row["client_num_per_round"]),
        "comm_round": int(row["comm_round"]),
        "epochs": int(row["epochs"]),
        "batch_size": int(row["batch_size"]),
        "learning_rate": float(row["learning_rate"]),
        "critic_multiplier": float(row["critic_multiplier"]),
        "weight_decay": float(row["weight_decay"]),
        "server_learning_rate": float(row["server_learning_rate"]),
        "partition_alpha": float(row["partition_alpha"]),
        "server_buffer_policy": row["server_buffer_policy"],
        "gpu_id": 0,
        "overwrite": False,
    }
    return run_manifest.Job(
        row=row,
        config=config,
        config_path=root / "generated.yaml",
        run_dir=root / "run",
        command=[sys.executable, "-c", "raise RuntimeError('must not launch')"],
        env={},
        gpu_id=0,
    )


class ResumeSafetyTests(unittest.TestCase):
    def test_scientifically_superseded_row_cannot_launch(self) -> None:
        row = _row()
        row["scientific_status"] = "superseded_pre_fix_selected_shortlist"
        with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "not launch-eligible"):
            run_manifest._require_launch_eligible(row)

    def test_completed_terminal_run_is_skipped_and_never_launched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _row()
            job = _job(root, row)
            _write_complete_run(job.run_dir, row, terminal=True)
            with mock.patch.object(run_manifest.subprocess, "Popen") as popen:
                results = run_manifest.run_jobs(
                    [job],
                    gpu_ids=[0],
                    max_parallel=1,
                    resume_skip_completed=True,
                    overwrite_incomplete=True,
                    stop_on_failure=True,
                    results_json=root / "results.json",
                )
            popen.assert_not_called()
            self.assertEqual(results[0]["status"], "skipped_terminal_ineligible")
            self.assertTrue((job.run_dir / "metrics.json").exists())
            ledger = root / "results_attempts.jsonl"
            self.assertGreaterEqual(len(ledger.read_text().splitlines()), 3)

    def test_malformed_complete_run_is_preserved_and_blocks_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _row()
            job = _job(root, row)
            _write_complete_run(job.run_dir, row, terminal=False)
            config_path = job.run_dir / "effective_config.json"
            config = json.loads(config_path.read_text())
            config["learning_rate"] = 0.9
            config_path.write_text(json.dumps(config))
            with mock.patch.object(run_manifest.subprocess, "Popen") as popen:
                results = run_manifest.run_jobs(
                    [job],
                    gpu_ids=[0],
                    max_parallel=1,
                    resume_skip_completed=True,
                    overwrite_incomplete=True,
                    stop_on_failure=True,
                )
            popen.assert_not_called()
            self.assertEqual(results[0]["status"], "failed_existing_artifacts")
            self.assertEqual(json.loads(config_path.read_text())["learning_rate"], 0.9)

    def test_round_count_and_order_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            row = _row()
            _write_complete_run(run_dir, row, terminal=False)
            curve_path = run_dir / "mse_by_round.csv"
            rows = list(csv.DictReader(curve_path.open()))
            rows[1]["round"] = "0"
            with curve_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "expected 1"):
                run_manifest.validate_artifacts(run_dir, row)

    def test_relevant_batchnorm_telemetry_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            row = _row()
            _write_complete_run(run_dir, row, terminal=False)
            curve_path = run_dir / "mse_by_round.csv"
            rows = list(csv.DictReader(curve_path.open()))
            rows[0]["g_bn_min_running_var"] = ""
            with curve_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(
                run_manifest.ManifestLaunchError,
                "g_bn_min_running_var is blank",
            ):
                run_manifest.validate_artifacts(run_dir, row)

    def test_metrics_policy_and_nonfinite_evidence_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            row = _row()
            _write_complete_run(run_dir, row, terminal=False)
            metrics_path = run_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["nonfinite_first_round"] = 1
            metrics["nonfinite_diagnostics"] = [{"round": 1}]
            metrics_path.write_text(json.dumps(metrics))
            validation = run_manifest.validate_artifacts(run_dir, row)
            self.assertTrue(validation["terminal_ineligible"])

            metrics["server_buffer_policy"] = "legacy_state_dict_arithmetic"
            metrics_path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(
                run_manifest.ManifestLaunchError,
                "server_buffer_policy",
            ):
                run_manifest.validate_artifacts(run_dir, row)

    def test_nonfinite_critic_output_alone_is_terminal_ineligible(self) -> None:
        # PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md's "critic collapse is
        # diagnostic-only" rule is about near-constant but FINITE critic
        # output -- a distinct, lesser phenomenon with no frozen threshold.
        # A critic output going nonfinite (NaN/Inf) is a numerical failure,
        # not a diagnostic pattern, and must gate promotion regardless of
        # every other value being finite. Mirrors the exact diagnostic dict
        # shape fedavg_api.py's train() loop appends to nonfinite_diagnostics
        # (see its "critic_outputs_finite" key) -- see
        # experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813/CRITIC_COLLAPSE_VS_NONFINITE_ADDENDUM_20260822.md.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            row = _row()
            _write_complete_run(run_dir, row, terminal=False)
            metrics_path = run_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["nonfinite_first_round"] = 1
            metrics["nonfinite_diagnostics"] = [{
                "round": 1,
                "state_finite": True,
                "metrics_finite": True,
                "critic_outputs_finite": False,
                "g_bn_min_running_var": 1e-6,
                "f_bn_min_running_var": 1e-6,
            }]
            metrics_path.write_text(json.dumps(metrics))
            validation = run_manifest.validate_artifacts(run_dir, row)
            self.assertTrue(validation["terminal_ineligible"])

    def test_partial_expected_config_does_not_disable_manifest_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            row = _row()
            _write_complete_run(run_dir, row, terminal=False)
            config_path = run_dir / "effective_config.json"
            config = json.loads(config_path.read_text())
            config["client_num_per_round"] = 9
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(
                run_manifest.ManifestLaunchError,
                "client_num_per_round mismatch",
            ):
                run_manifest.validate_artifacts(
                    run_dir,
                    row,
                    expected_config={"dataset": row["dataset"]},
                )

    def test_partial_archive_preserves_original_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "method" / "partial"
            run_dir.mkdir(parents=True)
            (run_dir / "mse_by_round.csv").write_text("round,val_mse\n0,0.5\n")
            archived = run_manifest._archive_partial_run(run_dir, "attempt-1")
            self.assertFalse(run_dir.exists())
            self.assertEqual(
                (archived / "mse_by_round.csv").read_text(),
                "round,val_mse\n0,0.5\n",
            )


if __name__ == "__main__":
    unittest.main()
