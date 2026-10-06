"""Tests for scripts/aggregate_highdim_deterministic_finals_post_bn_20260826.py.

Covers the row-count/unresolved-blocking guards with cheap fixtures, plus one
full 180-trajectory happy path proving test metrics unlock correctly once
every planned trajectory is resolved. Uses the finals_evidence_ledger.json
contract (not a flat CSV manifest) -- reused entries carry their real
run_id, matching what validate_artifacts() requires.
"""

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

import aggregate_highdim_deterministic_finals_post_bn_20260826 as aggregate_finals  # noqa: E402
import run_manifest  # noqa: E402

COMM_ROUND = 500
DATASETS = ["femnist_x", "femnist_z", "femnist_xz", "cifar10_x", "cifar10_z", "cifar10_xz"]
METHODS = ("fedgda_d", "fedogda_d")
ALPHAS = (0.1, 0.5, 1.0)
SEEDS = (0, 1, 2, 3, 4)


def _write_run(run_dir: Path, run_id: str, *, dataset="fixture", method="fedgda_d",
                seed=0, terminal: bool = False, alpha=0.5, lr=0.01, cm=5.0) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "best_validation.pt").write_bytes(b"fake")
    (run_dir / "checkpoints" / "final.pt").write_bytes(b"fake")
    (run_dir / "predictions.npz").write_bytes(b"fake")
    (run_dir / "effective_config.json").write_text(json.dumps({
        "dataset": dataset, "variant": method, "run_id": run_id,
        "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
        "random_seed": seed, "comm_round": COMM_ROUND,
        "server_buffer_policy": "direct_client_aggregate",
        "test_mse_used_for_selection": False, "selection_metric_source": "validation",
        "log_test_mse_by_round": False,
        "learning_rate": lr, "critic_multiplier": cm, "partition_alpha": alpha,
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "diverged": terminal, "best_gmm_eval": 1.0, "best_validation_mse": 0.1,
        "run_status": "completed", "rounds_completed": COMM_ROUND,
        "server_buffer_policy": "direct_client_aggregate",
        "nonfinite_first_round": 400 if terminal else None,
        "nonfinite_diagnostics": ([{"round": 400}] if terminal else []),
        "test_mse_at_best_validation": 0.2, "final_test_mse": 0.25, "best_validation_round": 10,
        "g_bn_min_running_var": 0.01, "f_bn_min_running_var": 0.02,
    }))
    with (run_dir / "mse_by_round.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "round", "train_mse", "val_mse", "primary_val_mse", "equal_client_val_mse",
            "train_moment_violation", "val_moment_violation", "gmm_train_objective",
            "gmm_val_objective", "gmm_eval", "g_bn_min_running_var", "f_bn_min_running_var",
            "finite", "diverged",
        ])
        for i in range(COMM_ROUND):
            row_finite = "True"
            row_diverged = "False"
            if terminal and i == 400:
                row_finite = "False"
            writer.writerow([
                i, 0.1, 0.1, 0.1, "", 0.1, 0.1, 0.1, 0.1, 1.0, 0.01, 0.02,
                row_finite, row_diverged,
            ])


def _entry(run_id, dataset, method, seed, alpha, final_result_dir, reused=False, source_stage="",
           lr=0.01, cm=5.0):
    return {
        "run_id": run_id, "dataset": dataset, "method": method, "seed": seed, "alpha": alpha,
        "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
        "final_result_dir": str(final_result_dir), "reused": reused, "source_stage": source_stage,
        "learning_rate": lr, "critic_multiplier": cm, "partition_alpha": alpha,
        "comm_round": COMM_ROUND,
    }


class AggregateGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_ledger(self, trajectories, *, status="complete") -> Path:
        ledger_path = self.tmp / "finals_evidence_ledger.json"
        ledger_path.write_text(json.dumps({
            "status": status,
            "total_trajectories": len(trajectories),
            "reused_trajectories": sum(1 for t in trajectories if t.get("reused")),
            "new_trajectories": sum(1 for t in trajectories if not t.get("reused")),
            "trajectories": trajectories,
        }))
        return ledger_path

    def test_wrong_trajectory_count_rejected(self):
        ledger_path = self._write_ledger([_entry("only_one", "d", "fedgda_d", 0, 0.5, "x")])
        with self.assertRaisesRegex(ValueError, "exactly 180 trajectories"):
            aggregate_finals.aggregate(ledger_path)

    def test_incomplete_ledger_status_rejected(self):
        ledger_path = self._write_ledger([], status="in_progress")
        with self.assertRaisesRegex(ValueError, "absent or incomplete"):
            aggregate_finals.aggregate(ledger_path)

    def test_unresolved_trajectory_locks_the_whole_report(self):
        trajectories = []
        for i in range(180):
            run_dir = self.tmp / f"run_{i}"
            run_id = f"run_{i}"
            if i != 0:
                _write_run(run_dir, run_id, dataset="d", method="fedgda_d")
            # i == 0 deliberately never written -- unresolved
            trajectories.append(_entry(run_id, "d", "fedgda_d", 0, 0.5, run_dir))
        ledger_path = self._write_ledger(trajectories)
        with self.assertRaisesRegex(ValueError, "not yet auditably resolved"):
            aggregate_finals.aggregate(ledger_path)

    def test_duplicate_run_id_rejected(self):
        trajectories = []
        for i in range(180):
            run_dir = self.tmp / f"run_{i}"
            run_id = "same_run_id" if i < 2 else f"run_{i}"
            trajectories.append(_entry(run_id, "d", "fedgda_d", 0, 0.5, run_dir))
        ledger_path = self._write_ledger(trajectories)
        with self.assertRaisesRegex(ValueError, "duplicate run_ids"):
            aggregate_finals.aggregate(ledger_path)


class AggregateSeedBindingTest(unittest.TestCase):
    """The exact bug a review caught: _row_report's row dict omitted "seed",
    so validate_artifacts()'s random_seed cross-check silently no-op'd --
    a ledger entry claiming seed=4 could point at a run whose real
    effective_config.json says random_seed=0, undetected."""

    def test_ledger_seed_mismatched_against_real_effective_config_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            run_dir = tmp / "run"
            run_id = "det_finals_seed_mismatch"
            # The run really trained at seed 0 (what's on disk)...
            _write_run(run_dir, run_id, dataset="femnist_x", method="fedgda_d", seed=0)
            # ...but the ledger entry claims it's the seed-4 trajectory.
            entry = _entry(run_id, "femnist_x", "fedgda_d", 4, 0.5, run_dir)
            with self.assertRaisesRegex(aggregate_finals.ManifestLaunchError, "random_seed mismatch"):
                aggregate_finals._row_report(run_dir, entry)


class AggregateReusedEntryTest(unittest.TestCase):
    """The exact bug this covers: a reused entry's run_id must match what's
    really on disk, or validate_artifacts() rejects it."""

    def test_reused_entry_with_real_run_id_validates_successfully(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            run_dir = tmp / "v4_result"
            real_run_id = "det_adjudicate_v4_femnist_x_fedgda_d_seed0_lr0p001_cm5"
            _write_run(run_dir, real_run_id, dataset="femnist_x", method="fedgda_d", seed=0)
            entry = _entry(
                real_run_id, "femnist_x", "fedgda_d", 0, 0.5, run_dir,
                reused=True, source_stage="psi_adjudication_post_bn_v4",
            )
            report = aggregate_finals._row_report(run_dir, entry)
            self.assertFalse(report["terminal_ineligible"])
            self.assertEqual(report["final_test_mse"], 0.25)
            self.assertTrue(report["reused"])
            # Freshly recomputed provenance, not trusted from the ledger.
            self.assertEqual(
                report["effective_config_checksum"],
                aggregate_finals.config_checksum(
                    json.loads((run_dir / "effective_config.json").read_text())
                ),
            )


class AggregateLabelCrossCheckTest(unittest.TestCase):
    """The exact bug class this covers (closeout review finding): the ledger
    labels a trajectory's alpha/lr/cm, but nothing used to check those
    labels against what the run actually trained at -- a preparer bug (like
    the alpha=0.1 stability/retune/finals partition_alpha bug) could mislabel
    evidence and this aggregator would report it as if the label were true."""

    def test_partition_alpha_mismatch_against_real_config_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            run_dir = tmp / "run"
            run_id = "det_finals_postbn_femnist_x_fedgda_d_seed0_alpha0p1_lr0p01_cm5"
            # The run actually trained at partition_alpha=0.5 ...
            _write_run(run_dir, run_id, dataset="femnist_x", method="fedgda_d", seed=0, alpha=0.5)
            # ... but the ledger labels it alpha=0.1.
            entry = _entry(run_id, "femnist_x", "fedgda_d", 0, 0.1, run_dir)
            with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "partition_alpha mismatch"):
                aggregate_finals._row_report(run_dir, entry)

    def test_learning_rate_mismatch_against_real_config_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            run_dir = tmp / "run"
            run_id = "det_finals_postbn_femnist_x_fedgda_d_seed0_alpha0p5_lr0p01_cm5"
            _write_run(run_dir, run_id, dataset="femnist_x", method="fedgda_d", seed=0, lr=0.01)
            entry = _entry(run_id, "femnist_x", "fedgda_d", 0, 0.5, run_dir, lr=0.02)
            with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "learning_rate mismatch"):
                aggregate_finals._row_report(run_dir, entry)

    def test_critic_multiplier_mismatch_against_real_config_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            run_dir = tmp / "run"
            run_id = "det_finals_postbn_femnist_x_fedgda_d_seed0_alpha0p5_lr0p01_cm5"
            _write_run(run_dir, run_id, dataset="femnist_x", method="fedgda_d", seed=0, cm=5.0)
            entry = _entry(run_id, "femnist_x", "fedgda_d", 0, 0.5, run_dir, cm=10.0)
            with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "critic_multiplier mismatch"):
                aggregate_finals._row_report(run_dir, entry)


def _build_full_matrix(tmp: Path) -> list[dict]:
    trajectories = []
    for dataset in DATASETS:
        for method in METHODS:
            for alpha in ALPHAS:
                for seed in SEEDS:
                    run_id = f"final_{dataset}_{method}_a{alpha}_s{seed}"
                    run_dir = tmp / "results" / run_id
                    _write_run(run_dir, run_id, dataset=dataset, method=method, seed=seed, alpha=alpha)
                    trajectories.append(_entry(run_id, dataset, method, seed, alpha, run_dir))
    return trajectories


def _write_ledger(tmp: Path, trajectories: list[dict]) -> Path:
    ledger_path = tmp / "finals_evidence_ledger.json"
    ledger_path.write_text(json.dumps({
        "status": "complete", "total_trajectories": len(trajectories),
        "reused_trajectories": 0, "new_trajectories": len(trajectories),
        "trajectories": trajectories,
    }))
    return ledger_path


class AggregateExactAxesTest(unittest.TestCase):
    """A trajectory carrying a dataset/method/alpha/seed outside this
    campaign's frozen six-dataset/two-method/three-alpha/five-seed matrix
    must not silently pass just because its own group still totals 5 --
    these are set-equality checks, not just counts."""

    def test_unexpected_dataset_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            trajectories = _build_full_matrix(tmp)
            bad = trajectories[0]
            run_id = f"bogus_{bad['run_id']}"
            run_dir = tmp / "results" / run_id
            _write_run(
                run_dir, run_id, dataset="femnist_bogus", method=bad["method"],
                seed=bad["seed"], alpha=bad["alpha"],
            )
            trajectories[0] = _entry(run_id, "femnist_bogus", bad["method"], bad["seed"], bad["alpha"], run_dir)
            ledger_path = _write_ledger(tmp, trajectories)
            with self.assertRaisesRegex(ValueError, "unexpected datasets"):
                aggregate_finals.aggregate(ledger_path)

    def test_unexpected_method_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            trajectories = _build_full_matrix(tmp)
            bad = trajectories[0]
            run_id = f"bogus_{bad['run_id']}"
            run_dir = tmp / "results" / run_id
            _write_run(
                run_dir, run_id, dataset=bad["dataset"], method="fed_eg_d",
                seed=bad["seed"], alpha=bad["alpha"],
            )
            trajectories[0] = _entry(run_id, bad["dataset"], "fed_eg_d", bad["seed"], bad["alpha"], run_dir)
            ledger_path = _write_ledger(tmp, trajectories)
            with self.assertRaisesRegex(ValueError, "unexpected methods"):
                aggregate_finals.aggregate(ledger_path)

    def test_unexpected_alpha_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            trajectories = _build_full_matrix(tmp)
            bad = trajectories[0]
            run_id = f"bogus_{bad['run_id']}"
            run_dir = tmp / "results" / run_id
            _write_run(
                run_dir, run_id, dataset=bad["dataset"], method=bad["method"],
                seed=bad["seed"], alpha=0.7,
            )
            trajectories[0] = _entry(run_id, bad["dataset"], bad["method"], bad["seed"], 0.7, run_dir)
            ledger_path = _write_ledger(tmp, trajectories)
            with self.assertRaisesRegex(ValueError, "unexpected alphas"):
                aggregate_finals.aggregate(ledger_path)

    def test_duplicate_seed_masking_a_missing_seed_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            trajectories = _build_full_matrix(tmp)
            target = next(
                i for i, t in enumerate(trajectories)
                if t["dataset"] == DATASETS[0] and t["method"] == METHODS[0]
                and t["alpha"] == ALPHAS[0] and t["seed"] == 4
            )
            bad = trajectories[target]
            # Same count (5 entries in this group), but seed 4 is silently
            # replaced by a second seed-0 entry -- seed identity, not count,
            # is what must catch this.
            run_id = f"dupe_{bad['run_id']}"
            run_dir = tmp / "results" / run_id
            _write_run(
                run_dir, run_id, dataset=bad["dataset"], method=bad["method"],
                seed=0, alpha=bad["alpha"],
            )
            trajectories[target] = _entry(run_id, bad["dataset"], bad["method"], 0, bad["alpha"], run_dir)
            ledger_path = _write_ledger(tmp, trajectories)
            with self.assertRaisesRegex(ValueError, "expected seeds"):
                aggregate_finals.aggregate(ledger_path)


class AggregateFullMatrixTest(unittest.TestCase):
    def test_full_180_matrix_unlocks_test_metrics_and_summarizes_by_cell_alpha(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            trajectories = []
            terminal_run_id = None
            for dataset in DATASETS:
                for method in METHODS:
                    for alpha in ALPHAS:
                        for seed in SEEDS:
                            run_id = f"final_{dataset}_{method}_a{alpha}_s{seed}"
                            run_dir = tmp / "results" / run_id
                            terminal = (
                                terminal_run_id is None and dataset == DATASETS[0]
                                and method == METHODS[0] and alpha == 0.5 and seed == 0
                            )
                            if terminal:
                                terminal_run_id = run_id
                            _write_run(
                                run_dir, run_id, dataset=dataset, method=method,
                                seed=seed, terminal=terminal, alpha=alpha,
                            )
                            trajectories.append(_entry(run_id, dataset, method, seed, alpha, run_dir))
            ledger_path = tmp / "finals_evidence_ledger.json"
            ledger_path.write_text(json.dumps({
                "status": "complete", "total_trajectories": len(trajectories),
                "reused_trajectories": 0, "new_trajectories": len(trajectories),
                "trajectories": trajectories,
            }))

            result = aggregate_finals.aggregate(ledger_path)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["trajectories"]), 180)
            self.assertEqual(len(result["cross_seed_summary"]), 36)  # 12 cells x 3 alphas

            terminal_entries = [
                t for t in result["trajectories"] if t["run_id"] == terminal_run_id
            ]
            self.assertEqual(len(terminal_entries), 1)
            self.assertTrue(terminal_entries[0]["terminal_ineligible"])
            self.assertIsNone(terminal_entries[0]["final_test_mse"])

            stable_entries = [
                t for t in result["trajectories"] if t["run_id"] != terminal_run_id
            ]
            self.assertTrue(all(t["final_test_mse"] == 0.25 for t in stable_entries))

            key = f"{DATASETS[0]}|{METHODS[0]}|0.5"
            self.assertEqual(result["cross_seed_summary"][key]["seeds_terminal"], 1)
            self.assertEqual(result["cross_seed_summary"][key]["seeds_stable"], 4)


if __name__ == "__main__":
    unittest.main()
