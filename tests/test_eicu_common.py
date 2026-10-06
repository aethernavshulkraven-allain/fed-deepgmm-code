"""Tests for the canonical final-iterate stability diagnostic in eicu_common.py.

``final_iterate_stability`` canonicalizes a definition that was previously
reimplemented ad hoc (with differing return signatures) in
``analyze_fedogda_s_focused_v3.py``, ``analyze_optimistic_curve_screen_v1.py``,
``analyze_fedogda_s_step_fast_v5.py``, and ``analyze_fedogda_s_tuning_pilot.py``.
These tests pin the exact window/statistic/divergence semantics documented on
the function, plus the additional robustness (no crash on non-finite tail
values) that the four priors did not have.
"""

import csv
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import eicu_common  # noqa: E402


def write_mse_by_round(path, rows):
    fieldnames = ["round", "val_mse", "primary_val_mse", "finite", "diverged"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


class FinalIterateStabilityTest(unittest.TestCase):
    def test_window_defaults_to_last_50_rows(self):
        # 60 rows: first 10 noisy (should be excluded from the tail stat),
        # last 50 exactly constant at 1.0 (population std must be exactly 0).
        rows = []
        for round_idx in range(10):
            rows.append({"round": round_idx, "val_mse": 100.0 + round_idx, "finite": "True", "diverged": "False"})
        for round_idx in range(10, 60):
            rows.append({"round": round_idx, "val_mse": 1.0, "finite": "True", "diverged": "False"})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mse_by_round.csv")
            write_mse_by_round(path, rows)
            result = eicu_common.final_iterate_stability(path)
        self.assertEqual(result["n_rows_total"], 60)
        self.assertEqual(result["window_used"], 50)
        self.assertEqual(result["window_requested"], 50)
        self.assertAlmostEqual(result["tail_mean"], 1.0)
        self.assertAlmostEqual(result["tail_std"], 0.0)
        self.assertAlmostEqual(result["tail_range"], 0.0)
        self.assertFalse(result["diverged"])
        self.assertIsNone(result["first_diverged_round"])

    def test_window_shrinks_when_fewer_rows_than_window(self):
        rows = [
            {"round": i, "val_mse": float(i), "finite": "True", "diverged": "False"}
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mse_by_round.csv")
            write_mse_by_round(path, rows)
            result = eicu_common.final_iterate_stability(path)
        self.assertEqual(result["n_rows_total"], 5)
        self.assertEqual(result["window_used"], 5)
        # population std of [0,1,2,3,4]
        self.assertAlmostEqual(result["tail_std"], 1.4142135623730951, places=8)

    def test_single_row_window_has_zero_std(self):
        rows = [{"round": 0, "val_mse": 3.5, "finite": "True", "diverged": "False"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mse_by_round.csv")
            write_mse_by_round(path, rows)
            result = eicu_common.final_iterate_stability(path)
        self.assertEqual(result["window_used"], 1)
        self.assertEqual(result["tail_std"], 0.0)
        self.assertAlmostEqual(result["tail_mean"], 3.5)

    def test_empty_history_is_conservatively_diverged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mse_by_round.csv")
            write_mse_by_round(path, [])
            result = eicu_common.final_iterate_stability(path)
        self.assertEqual(result["n_rows_total"], 0)
        self.assertEqual(result["window_used"], 0)
        self.assertTrue(result["diverged"])
        self.assertIsNone(result["tail_std"])

    def test_divergence_flag_uses_full_history_not_just_tail(self):
        # Divergence happens at round 2 (outside a window=50 tail that would
        # only see rounds 10..59), then the run keeps producing finite rows.
        # The prior definition in all four scripts checks the FULL history,
        # so a run cannot "erase" an early divergence by recovering later.
        rows = []
        for round_idx in range(60):
            diverged = round_idx == 2
            finite = "False" if diverged else "True"
            rows.append({
                "round": round_idx,
                "val_mse": 1.0,
                "finite": finite,
                "diverged": "True" if diverged else "False",
            })
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mse_by_round.csv")
            write_mse_by_round(path, rows)
            result = eicu_common.final_iterate_stability(path)
        self.assertTrue(result["diverged"])
        self.assertEqual(result["first_diverged_round"], 2)
        # The tail window itself (rounds 10..59) is all-finite, so the
        # returned tail statistics are still computed normally.
        self.assertEqual(result["window_used"], 50)
        self.assertAlmostEqual(result["tail_std"], 0.0)

    def test_finite_false_without_diverged_flag_still_counts(self):
        rows = [
            {"round": 0, "val_mse": 1.0, "finite": "False", "diverged": "False"},
            {"round": 1, "val_mse": 1.0, "finite": "True", "diverged": "False"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mse_by_round.csv")
            write_mse_by_round(path, rows)
            result = eicu_common.final_iterate_stability(path)
        self.assertTrue(result["diverged"])
        self.assertEqual(result["first_diverged_round"], 0)

    def test_nonfinite_tail_values_are_skipped_not_fatal(self):
        # Unlike the four ad hoc priors (which call float() on every tail
        # value and raise on the first non-finite one), this canonical
        # implementation must not crash a batch analysis on one bad run.
        rows = [
            {"round": 0, "val_mse": "nan", "finite": "False", "diverged": "True"},
            {"round": 1, "val_mse": "inf", "finite": "False", "diverged": "True"},
            {"round": 2, "val_mse": 2.0, "finite": "True", "diverged": "False"},
            {"round": 3, "val_mse": 4.0, "finite": "True", "diverged": "False"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mse_by_round.csv")
            write_mse_by_round(path, rows)
            result = eicu_common.final_iterate_stability(path)
        self.assertEqual(result["window_used"], 4)
        self.assertEqual(result["n_tail_finite"], 2)
        self.assertEqual(result["n_tail_nonfinite"], 2)
        self.assertAlmostEqual(result["tail_mean"], 3.0)
        self.assertTrue(result["diverged"])

    def test_custom_metric_column_for_study_a_equal_client_selector(self):
        # Study A v2's confirmatory analyzer passes metric_column=
        # "primary_val_mse" (the equal-client selector), which can differ
        # from the pooled "val_mse" column the four legacy scripts use.
        rows = [
            {"round": 0, "val_mse": 100.0, "primary_val_mse": 1.0, "finite": "True", "diverged": "False"},
            {"round": 1, "val_mse": 200.0, "primary_val_mse": 3.0, "finite": "True", "diverged": "False"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mse_by_round.csv")
            write_mse_by_round(path, rows)
            result = eicu_common.final_iterate_stability(path, metric_column="primary_val_mse")
        self.assertEqual(result["metric_column"], "primary_val_mse")
        self.assertAlmostEqual(result["tail_mean"], 2.0)
        self.assertAlmostEqual(result["tail_min"], 1.0)
        self.assertAlmostEqual(result["tail_max"], 3.0)

    def test_tail_cv_is_std_over_abs_mean(self):
        rows = [
            {"round": 0, "val_mse": 8.0, "finite": "True", "diverged": "False"},
            {"round": 1, "val_mse": 12.0, "finite": "True", "diverged": "False"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mse_by_round.csv")
            write_mse_by_round(path, rows)
            result = eicu_common.final_iterate_stability(path)
        self.assertAlmostEqual(result["tail_mean"], 10.0)
        self.assertAlmostEqual(result["tail_std"], 2.0)
        self.assertAlmostEqual(result["tail_cv"], 0.2)

    def test_explicit_window_override(self):
        rows = [
            {"round": i, "val_mse": float(i), "finite": "True", "diverged": "False"}
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mse_by_round.csv")
            write_mse_by_round(path, rows)
            result = eicu_common.final_iterate_stability(path, window=3)
        self.assertEqual(result["window_requested"], 3)
        self.assertEqual(result["window_used"], 3)
        # last 3 values: 7, 8, 9
        self.assertAlmostEqual(result["tail_mean"], 8.0)


if __name__ == "__main__":
    unittest.main()
