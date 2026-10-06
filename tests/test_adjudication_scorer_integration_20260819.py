"""End-to-end integration test for scripts/score_highdim_adjudication_20260819.py:
exercises the real disk-reading path (load_seed_result, resolve_new_run_dir,
resolve_reused_run_dir, load_finals_index, build_cell_candidates,
score_manifest) against a small fixture tree that mimics the real
directory layout -- not just the pure scoring logic already covered by
tests/test_adjudication_scorer_20260819.py's synthetic Candidate objects.

This is the "wiring" the review feedback asked to freeze before real
adjudication results exist: does the CLI actually find, parse, and score
real completed/reused runs correctly.
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import score_highdim_adjudication_20260819 as scorer  # noqa: E402


def write_run(
    run_dir: Path,
    *,
    dataset: str,
    method: str,
    seed: int,
    learning_rate: float,
    critic_multiplier: float,
    psi_values: list[float],
    mse_values: list[float],
    diverged: bool,
) -> None:
    comm_round = scorer.REQUIRED_COMM_ROUNDS
    if len(psi_values) != comm_round or len(mse_values) != comm_round:
        raise ValueError("fixture curves must contain exactly 500 rounds")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "best_validation.pt").write_bytes(b"fake")
    (run_dir / "checkpoints" / "final.pt").write_bytes(b"fake")
    (run_dir / "predictions.npz").write_bytes(b"fake")
    (run_dir / "effective_config.json").write_text(json.dumps({
        "dataset": dataset, "variant": method, "comm_round": comm_round,
        "random_seed": seed, "run_id": run_dir.name,
        "client_optimizer": scorer.METHOD_TO_OPTIMIZER[method],
        "learning_rate": learning_rate, "critic_multiplier": critic_multiplier,
        "server_buffer_policy": "direct_client_aggregate",
        "test_mse_used_for_selection": False,
        "selection_metric_source": "validation",
        "log_test_mse_by_round": False,
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "diverged": diverged, "best_gmm_eval": max(psi_values),
        "best_validation_mse": min(mse_values), "run_status": "completed",
        "rounds_completed": comm_round,
        "server_buffer_policy": "direct_client_aggregate",
        "nonfinite_first_round": 0 if diverged else None,
        "nonfinite_diagnostics": ([{"round": 0}] if diverged else []),
    }))
    with (run_dir / "mse_by_round.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "round", "train_mse", "val_mse", "primary_val_mse",
            "equal_client_val_mse", "train_moment_violation",
            "val_moment_violation", "gmm_train_objective",
            "gmm_val_objective", "gmm_eval", "g_bn_min_running_var",
            "f_bn_min_running_var", "finite", "diverged",
        ])
        for i, (psi, mse) in enumerate(zip(psi_values, mse_values)):
            writer.writerow([
                i, mse, mse, mse, "", 0.2, 0.2, 0.1, 0.1, psi,
                "", "", str(not diverged), str(diverged),
            ])


class EndToEndScoringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.campaign_dir = self.tmp / "psi_adjudication_20260819_v2"
        self.finals_dir = self.tmp / "deterministic_finals_20260813"
        self.new_run_root = self.tmp / "results_new"
        self.campaign_dir.mkdir()
        self.finals_dir.mkdir()
        self.new_run_root.mkdir()

        self._orig = (scorer.CAMPAIGN_V2_DIR, scorer.FINALS_DIR, scorer.NEW_RUN_RESULTS_ROOT)
        scorer.CAMPAIGN_V2_DIR = self.campaign_dir
        scorer.FINALS_DIR = self.finals_dir
        scorer.NEW_RUN_RESULTS_ROOT = self.new_run_root

    def tearDown(self):
        scorer.CAMPAIGN_V2_DIR, scorer.FINALS_DIR, scorer.NEW_RUN_RESULTS_ROOT = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_finals_manifest(self, rows):
        fieldnames = ["dataset", "method", "learning_rate", "critic_multiplier", "seed",
                      "alpha", "run_id", "final_result_dir"]
        with (self.finals_dir / "finals_manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_adjudication_manifest(self, cells, summary):
        rows = []
        for cell in summary["plan"]:
            for candidate in cell["candidates"]:
                if candidate["reused_from_finals"]:
                    continue
                for seed in (0, 1, 2):
                    run_dir = scorer.resolve_new_run_dir(
                        cell["dataset"], cell["method"], seed,
                        candidate["lr"], candidate["cm"],
                    )
                    rows.append({
                        "run_id": run_dir.name,
                        "dataset": cell["dataset"],
                        "method": cell["method"],
                        "seed": seed,
                        "client_optimizer": scorer.METHOD_TO_OPTIMIZER[cell["method"]],
                        "comm_round": scorer.REQUIRED_COMM_ROUNDS,
                        "learning_rate": candidate["lr"],
                        "critic_multiplier": candidate["cm"],
                        "server_buffer_policy": "direct_client_aggregate",
                    })
        path = self.campaign_dir / f"adjudication_{cells}_manifest.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_valid_seed_fixture(self, run_dir: Path, seed: int = 0) -> dict:
        expected = {
            "dataset": "fixture_ds",
            "variant": "fedogda_d",
            "client_optimizer": "ogda",
            "learning_rate": 0.01,
            "critic_multiplier": 5.0,
        }
        write_run(
            run_dir, dataset=expected["dataset"], method=expected["variant"], seed=seed,
            learning_rate=expected["learning_rate"],
            critic_multiplier=expected["critic_multiplier"],
            psi_values=[2.0] * 500, mse_values=[0.4] * 500, diverged=False,
        )
        return expected

    def test_seed_loader_rejects_499_or_501_rounds(self):
        for row_count in (499, 501):
            run_dir = self.tmp / f"curve_{row_count}"
            expected = self._write_valid_seed_fixture(run_dir)
            curve_path = run_dir / "mse_by_round.csv"
            rows = list(csv.reader(curve_path.open()))
            if row_count == 499:
                rows = rows[:-1]
            else:
                rows.append([500, 0.4, 2.0, "True", "False"])
            with curve_path.open("w", newline="") as handle:
                csv.writer(handle).writerows(rows)
            result = scorer.load_seed_result(run_dir, 0, expected)
            self.assertEqual(result.status, "invalid")
            self.assertTrue(
                "must have 500 rows" in result.note or "row count" in result.note
            )

    def test_seed_loader_rejects_duplicate_round_and_config_mismatch(self):
        run_dir = self.tmp / "duplicate_round"
        expected = self._write_valid_seed_fixture(run_dir)
        curve_path = run_dir / "mse_by_round.csv"
        rows = list(csv.reader(curve_path.open()))
        rows[2][0] = "0"
        with curve_path.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        result = scorer.load_seed_result(run_dir, 0, expected)
        self.assertEqual(result.status, "invalid")
        self.assertIn("expected 1", result.note)

        other_run_dir = self.tmp / "wrong_config"
        expected = self._write_valid_seed_fixture(other_run_dir)
        config_path = other_run_dir / "effective_config.json"
        config = json.loads(config_path.read_text())
        config["critic_multiplier"] = 99
        config_path.write_text(json.dumps(config))
        result = scorer.load_seed_result(other_run_dir, 0, expected)
        self.assertEqual(result.status, "invalid")
        self.assertIn("critic_multiplier mismatch", result.note)

    def test_early_nonfinite_round_is_sticky_despite_finite_tail(self):
        run_dir = self.tmp / "sticky_nonfinite"
        expected = self._write_valid_seed_fixture(run_dir)
        curve_path = run_dir / "mse_by_round.csv"
        rows = list(csv.reader(curve_path.open()))
        rows[93][9] = "nan"
        rows[93][12] = "False"
        rows[93][13] = "True"
        with curve_path.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        result = scorer.load_seed_result(run_dir, 0, expected)
        self.assertEqual(result.status, "terminal_ineligible")
        self.assertTrue(result.diverged)
        self.assertIsNone(result.psi_last50_mean)

    def test_three_candidate_cell_clear_winner_reused_and_new_runs_mixed(self):
        """psi_rank1 (new run) clearly wins over mse_winner (reused from a
        finals-style path) and psi_rank2 (new run, deliberately diverged on
        one seed so it's excluded) -- exercises both resolve_new_run_dir and
        resolve_reused_run_dir against real files on disk end-to-end."""
        ds, method = "fake_ds", "fedgda_d"

        # mse_winner: reused_from_finals candidate, lr=0.03/cm=5, all 3 seeds
        # good but middling Psi -- lives under a *separate* finals-style root.
        finals_rows = []
        for seed in (0, 1, 2):
            run_dir = self.finals_dir / ds / method / f"seed_{seed}" / f"det_final_{ds}_{method}_seed{seed}_alpha0p5_lr0p03_cm5"
            write_run(
                run_dir, dataset=ds, method=method, seed=seed,
                learning_rate=0.03, critic_multiplier=5.0,
                psi_values=[3.0] * 500, mse_values=[0.3] * 500, diverged=False,
            )
            finals_rows.append({
                "dataset": ds, "method": method, "learning_rate": "0.03", "critic_multiplier": "5",
                "seed": str(seed), "alpha": "0.5",
                "run_id": run_dir.name, "final_result_dir": str(run_dir),
            })
        self._write_finals_manifest(finals_rows)

        # psi_rank1: new run, lr=0.1/cm=10, clearly highest Psi (10.0) across all 3 seeds.
        for seed in (0, 1, 2):
            run_dir = scorer.resolve_new_run_dir(ds, method, seed, 0.1, 10.0)
            write_run(
                run_dir, dataset=ds, method=method, seed=seed,
                learning_rate=0.1, critic_multiplier=10.0,
                psi_values=[10.0] * 500, mse_values=[0.1] * 500, diverged=False,
            )

        # psi_rank2: new run, lr=0.1/cm=5 -- seed 2 diverges, so this candidate
        # must be excluded even though its other two seeds look good (Psi=8.0).
        for seed in (0, 1):
            run_dir = scorer.resolve_new_run_dir(ds, method, seed, 0.1, 5.0)
            write_run(
                run_dir, dataset=ds, method=method, seed=seed,
                learning_rate=0.1, critic_multiplier=5.0,
                psi_values=[8.0] * 500, mse_values=[0.15] * 500, diverged=False,
            )
        run_dir = scorer.resolve_new_run_dir(ds, method, 2, 0.1, 5.0)
        write_run(
            run_dir, dataset=ds, method=method, seed=2,
            learning_rate=0.1, critic_multiplier=5.0,
            psi_values=[8.0] * 500, mse_values=[0.15] * 500, diverged=True,
        )

        summary = {
            "campaign": "test", "new_runs": 6,
            "plan": [{
                "dataset": ds, "method": method,
                "candidates": [
                    {"lr": 0.03, "cm": 5.0, "labels": ["mse_winner"], "reused_from_finals": True},
                    {"lr": 0.1, "cm": 5.0, "labels": ["psi_rank2"], "reused_from_finals": False},
                    {"lr": 0.1, "cm": 10.0, "labels": ["psi_rank1"], "reused_from_finals": False},
                ],
            }],
        }
        (self.campaign_dir / "adjudication_x_summary.json").write_text(json.dumps(summary))
        self._write_adjudication_manifest("x", summary)

        results = scorer.score_manifest("x")
        self.assertEqual(list(results.keys()), [f"{ds}/{method}"])
        cell = results[f"{ds}/{method}"]

        self.assertEqual(cell["outcome"], "promoted")
        self.assertEqual(cell["winner"]["candidate_id"], f"{ds}/{method}/lr0.1_cm10")
        self.assertIn(f"{ds}/{method}/lr0.1_cm5", cell["excluded_candidates"])
        self.assertEqual(len(cell["eligible_candidates"]), 2)  # mse_winner + psi_rank1, psi_rank2 excluded

        # Confirm the reused candidate's seeds were actually read from the
        # finals-style path (not silently treated as missing).
        reused = next(c for c in cell["all_candidates"] if c["candidate_id"] == f"{ds}/{method}/lr0.03_cm5")
        self.assertTrue(reused["eligible"])
        self.assertAlmostEqual(reused["median_psi"], 3.0)
        self.assertTrue(str(self.finals_dir) in reused["seeds"][0]["run_dir"])

    def test_missing_run_directory_is_ineligible_not_a_crash(self):
        """A candidate whose run never happened (no directory at all) must
        come back as artifacts_complete=False / ineligible, not raise."""
        ds, method = "fake_ds2", "fedogda_d"
        self._write_finals_manifest([])  # empty finals -- nothing to reuse
        for seed in (0, 1, 2):
            run_dir = scorer.resolve_new_run_dir(ds, method, seed, 0.01, 5.0)
            write_run(
                run_dir, dataset=ds, method=method, seed=seed,
                learning_rate=0.01, critic_multiplier=5.0,
                psi_values=[1.0] * 500, mse_values=[0.5] * 500, diverged=False,
            )
        # lr=0.02/cm=5's run directory is never created at all.

        summary = {
            "campaign": "test", "new_runs": 6,
            "plan": [{
                "dataset": ds, "method": method,
                "candidates": [
                    {"lr": 0.01, "cm": 5.0, "labels": ["psi_rank1"], "reused_from_finals": False},
                    {"lr": 0.02, "cm": 5.0, "labels": ["psi_rank2"], "reused_from_finals": False},
                ],
            }],
        }
        (self.campaign_dir / "adjudication_signal_summary.json").write_text(json.dumps(summary))
        self._write_adjudication_manifest("signal", summary)

        results = scorer.score_manifest("signal")
        cell = results[f"{ds}/{method}"]
        self.assertEqual(cell["outcome"], "incomplete")
        self.assertIsNone(cell["winner"])
        self.assertIn(f"{ds}/{method}/lr0.02_cm5", cell["excluded_candidates"])
        missing_candidate = next(c for c in cell["all_candidates"] if c["candidate_id"] == f"{ds}/{method}/lr0.02_cm5")
        self.assertFalse(missing_candidate["eligible"])
        self.assertFalse(missing_candidate["seeds"][0]["artifacts_complete"])


if __name__ == "__main__":
    unittest.main()
