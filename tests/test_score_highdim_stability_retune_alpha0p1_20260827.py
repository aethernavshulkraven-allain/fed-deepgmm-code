"""Tests for scripts/score_highdim_stability_retune_alpha0p1_20260827.py --
the alpha=0.1 retune fallback's Screen stage. Selects each retuned cell's
top-2 candidates by the same frozen last-50-mean Psi rule as the corrected
screen, over however many cells were actually retuned (unlike the screen's
fixed 12). It never itself promotes a winner -- that is
score_highdim_stability_retune_promote_alpha0p1_20260827.py's job, after
the Rank and Confirm stages this Screen stage's top-2 feeds into."""

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

import score_highdim_stability_retune_alpha0p1_20260827 as scorer  # noqa: E402

COMM_ROUND = scorer.RETUNE_COMM_ROUND


def write_run(run_dir: Path, *, dataset, method, run_id, gmm_eval, val_mse,
              learning_rate=0.01, critic_multiplier=5.0, val_target_variance=1.0) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "best_validation.pt").write_bytes(b"fake")
    (run_dir / "checkpoints" / "final.pt").write_bytes(b"fake")
    (run_dir / "predictions.npz").write_bytes(b"fake")
    (run_dir / "effective_config.json").write_text(json.dumps({
        "dataset": dataset, "variant": method, "run_id": run_id,
        "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
        "random_seed": 0, "comm_round": COMM_ROUND,
        "learning_rate": learning_rate, "critic_multiplier": critic_multiplier,
        "server_buffer_policy": "direct_client_aggregate",
        "test_mse_used_for_selection": False, "selection_metric_source": "validation",
        "log_test_mse_by_round": False,
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "diverged": False, "best_gmm_eval": gmm_eval, "best_validation_mse": val_mse,
        "run_status": "completed", "rounds_completed": COMM_ROUND,
        "server_buffer_policy": "direct_client_aggregate",
        "nonfinite_first_round": None, "nonfinite_diagnostics": [],
        "g_bn_min_running_var": 0.01, "f_bn_min_running_var": 0.01,
        # Default comfortably above every fixture's val_mse (max used
        # elsewhere in this file is 0.5), so existing tests keep clearing
        # the constant-predictor baseline unless a test overrides it.
        "val_target_variance": val_target_variance,
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
            writer.writerow([
                i, val_mse, val_mse, val_mse, "", 0.2, 0.2, 0.1, 0.1,
                gmm_eval, 0.01, 0.01, "True", "False",
            ])


class ScoreRetuneScreenTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.results_root = self.tmp / "results"
        self.rows = []

    def _add_row(self, *, dataset, method, lr, cm, gmm_eval, val_mse, val_target_variance=1.0):
        run_id = f"det_stability_retune_alpha0p1_{dataset}_{method}_seed0_lr{lr}_cm{cm}"
        run_dir = self.results_root / dataset / method / "seed_0" / run_id
        write_run(
            run_dir, dataset=dataset, method=method, run_id=run_id, gmm_eval=gmm_eval, val_mse=val_mse,
            learning_rate=lr, critic_multiplier=cm, val_target_variance=val_target_variance,
        )
        self.rows.append({
            "run_id": run_id, "dataset": dataset, "method": method, "seed": "0",
            "alpha": "0.1", "learning_rate": str(lr), "critic_multiplier": str(cm),
            "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
            "comm_round": str(COMM_ROUND), "final_result_dir": str(run_dir),
            "server_buffer_policy": "direct_client_aggregate",
        })

    def _write_manifest(self) -> Path:
        manifest_path = self.tmp / "retune_manifest.csv"
        fieldnames = list(self.rows[0].keys())
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return manifest_path

    def test_selects_top2_by_last50_psi_per_cell(self):
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5, gmm_eval=1.0, val_mse=0.5)
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.02, cm=10, gmm_eval=2.0, val_mse=0.4)
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.03, cm=20, gmm_eval=0.5, val_mse=0.3)
        manifest_path = self._write_manifest()
        result = scorer.score_retune(manifest_path)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["stage"], "screen")
        cell = result["cells"]["femnist_z|fedgda_d"]
        top2_lrs = [c["lr"] for c in cell["top2"]]
        self.assertEqual(top2_lrs, [0.02, 0.01])  # ranked by descending gmm_eval
        self.assertEqual(cell["eligible_candidates"], 3)

    def test_two_independent_cells_scored_separately(self):
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5, gmm_eval=1.0, val_mse=0.5)
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.02, cm=10, gmm_eval=2.0, val_mse=0.4)
        self._add_row(dataset="cifar10_x", method="fedogda_d", lr=0.001, cm=5, gmm_eval=5.0, val_mse=0.1)
        self._add_row(dataset="cifar10_x", method="fedogda_d", lr=0.003, cm=10, gmm_eval=1.0, val_mse=0.2)
        manifest_path = self._write_manifest()
        result = scorer.score_retune(manifest_path)
        self.assertEqual(len(result["cells"]), 2)
        self.assertEqual(result["cells"]["femnist_z|fedgda_d"]["top2"][0]["lr"], 0.02)
        self.assertEqual(result["cells"]["cifar10_x|fedogda_d"]["top2"][0]["lr"], 0.001)

    def test_exact_tie_at_top2_excluded_boundary_raises(self):
        # Three candidates: rank-1 is clear, but rank-2 and rank-3 tie
        # exactly -- which one belongs in the top-2 sent to Rank is
        # genuinely ambiguous, so this must fail closed rather than pick
        # one by manifest order.
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5, gmm_eval=5.0, val_mse=0.5)
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.02, cm=10, gmm_eval=2.0, val_mse=0.5)
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.03, cm=20, gmm_eval=2.0, val_mse=0.5)
        manifest_path = self._write_manifest()
        with self.assertRaisesRegex(ValueError, "unresolved exact Psi tie"):
            scorer.score_retune(manifest_path)

    def test_fewer_than_two_eligible_candidates_raises(self):
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5, gmm_eval=1.0, val_mse=0.5)
        manifest_path = self._write_manifest()
        with self.assertRaisesRegex(ValueError, "only 1 eligible screen candidate"):
            scorer.score_retune(manifest_path)

    def test_terminal_candidate_excluded_from_ranking(self):
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5, gmm_eval=1.0, val_mse=0.5)
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.02, cm=10, gmm_eval=1.5, val_mse=0.4)
        # A diverged candidate: overwrite its curve with a nonfinite entry.
        run_id = "det_stability_retune_alpha0p1_femnist_z_fedgda_d_seed0_lr0.03_cm20"
        run_dir = self.results_root / "femnist_z" / "fedgda_d" / "seed_0" / run_id
        write_run(
            run_dir, dataset="femnist_z", method="fedgda_d", run_id=run_id, gmm_eval=99.0, val_mse=0.01,
            learning_rate=0.03, critic_multiplier=20.0,
        )
        with (run_dir / "metrics.json").open() as handle:
            metrics = json.load(handle)
        metrics["diverged"] = True
        metrics["nonfinite_first_round"] = 100
        metrics["nonfinite_diagnostics"] = [{"round": 100}]
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        self.rows.append({
            "run_id": run_id, "dataset": "femnist_z", "method": "fedgda_d", "seed": "0",
            "alpha": "0.1", "learning_rate": "0.03", "critic_multiplier": "20",
            "client_optimizer": "sgd", "comm_round": str(COMM_ROUND), "final_result_dir": str(run_dir),
            "server_buffer_policy": "direct_client_aggregate",
        })
        manifest_path = self._write_manifest()
        result = scorer.score_retune(manifest_path)
        cell = result["cells"]["femnist_z|fedgda_d"]
        # The diverged candidate (gmm_eval=99.0) must not appear in top2
        # despite its huge diagnostic value -- it's excluded as terminal.
        top2_lrs = {c["lr"] for c in cell["top2"]}
        self.assertEqual(top2_lrs, {0.01, 0.02})
        self.assertEqual(cell["terminal_candidates"], 1)


class ConstantPredictorEligibilityTest(unittest.TestCase):
    """doe_review_and_revised_grid.md's escape hatch is only meaningful if a
    retune candidate actually beats the baseline whose failure triggered
    retuning -- a finite, nondivergent candidate that's still no better than
    a constant predictor must never be ranked or advanced, no matter how
    good its Psi score looks (closeout review finding)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.results_root = self.tmp / "results"
        self.rows = []

    def _add_row(self, *, dataset, method, lr, cm, gmm_eval, val_mse, val_target_variance=1.0):
        run_id = f"det_stability_retune_alpha0p1_{dataset}_{method}_seed0_lr{lr}_cm{cm}"
        run_dir = self.results_root / dataset / method / "seed_0" / run_id
        write_run(
            run_dir, dataset=dataset, method=method, run_id=run_id, gmm_eval=gmm_eval, val_mse=val_mse,
            learning_rate=lr, critic_multiplier=cm, val_target_variance=val_target_variance,
        )
        self.rows.append({
            "run_id": run_id, "dataset": dataset, "method": method, "seed": "0",
            "alpha": "0.1", "learning_rate": str(lr), "critic_multiplier": str(cm),
            "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
            "comm_round": str(COMM_ROUND), "final_result_dir": str(run_dir),
            "server_buffer_policy": "direct_client_aggregate",
        })

    def _write_manifest(self) -> Path:
        manifest_path = self.tmp / "retune_manifest.csv"
        fieldnames = list(self.rows[0].keys())
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return manifest_path

    def test_candidate_worse_than_constant_predictor_excluded(self):
        # This candidate has the best (highest) Psi of the three, but its
        # last50 val_mse (0.9) does not beat the constant-predictor baseline
        # (0.5) -- it must be excluded from ranking entirely, not merely
        # ranked lower.
        self._add_row(
            dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5,
            gmm_eval=99.0, val_mse=0.9, val_target_variance=0.5,
        )
        self._add_row(
            dataset="femnist_z", method="fedgda_d", lr=0.02, cm=10,
            gmm_eval=2.0, val_mse=0.4, val_target_variance=0.5,
        )
        self._add_row(
            dataset="femnist_z", method="fedgda_d", lr=0.03, cm=20,
            gmm_eval=1.0, val_mse=0.3, val_target_variance=0.5,
        )
        manifest_path = self._write_manifest()
        result = scorer.score_retune(manifest_path)
        cell = result["cells"]["femnist_z|fedgda_d"]
        top2_lrs = {c["lr"] for c in cell["top2"]}
        self.assertEqual(top2_lrs, {0.02, 0.03})
        self.assertEqual(cell["eligible_candidates"], 2)
        self.assertEqual(cell["baseline_failed_candidates"], 1)
        self.assertEqual(len(result["baseline_failed_runs"]), 1)

    def test_exactly_one_baseline_passing_candidate_cannot_satisfy_top2(self):
        self._add_row(
            dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5,
            gmm_eval=1.0, val_mse=0.9, val_target_variance=0.5,
        )
        self._add_row(
            dataset="femnist_z", method="fedgda_d", lr=0.02, cm=10,
            gmm_eval=2.0, val_mse=0.4, val_target_variance=0.5,
        )
        manifest_path = self._write_manifest()
        with self.assertRaisesRegex(ValueError, "only 1 eligible screen candidate"):
            scorer.score_retune(manifest_path)

    def test_two_baseline_passing_candidates_advance(self):
        self._add_row(
            dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5,
            gmm_eval=1.0, val_mse=0.4, val_target_variance=0.5,
        )
        self._add_row(
            dataset="femnist_z", method="fedgda_d", lr=0.02, cm=10,
            gmm_eval=2.0, val_mse=0.3, val_target_variance=0.5,
        )
        manifest_path = self._write_manifest()
        result = scorer.score_retune(manifest_path)
        cell = result["cells"]["femnist_z|fedgda_d"]
        self.assertEqual(cell["eligible_candidates"], 2)
        self.assertEqual(cell["baseline_failed_candidates"], 0)

    def test_missing_val_target_variance_fails_closed(self):
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5, gmm_eval=1.0, val_mse=0.4)
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.02, cm=10, gmm_eval=2.0, val_mse=0.3)
        manifest_path = self._write_manifest()
        run_dir = self.results_root / "femnist_z" / "fedgda_d" / "seed_0" / self.rows[0]["run_id"]
        with (run_dir / "metrics.json").open() as handle:
            metrics = json.load(handle)
        del metrics["val_target_variance"]
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "val_target_variance is missing or invalid"):
            scorer.score_retune(manifest_path)

    def test_nonfinite_val_target_variance_fails_closed(self):
        self._add_row(
            dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5,
            gmm_eval=1.0, val_mse=0.4, val_target_variance=-1.0,
        )
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.02, cm=10, gmm_eval=2.0, val_mse=0.3)
        manifest_path = self._write_manifest()
        with self.assertRaisesRegex(ValueError, "not a valid variance"):
            scorer.score_retune(manifest_path)


if __name__ == "__main__":
    unittest.main()
