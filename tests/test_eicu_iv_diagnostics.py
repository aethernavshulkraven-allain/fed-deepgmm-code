"""Tests for the numpy IV diagnostics.

``statsmodels``/``linearmodels`` are not installed, so correctness is established
against designs with a known answer: OLS is checked against a noiseless fit and
against the normal equations, and 2SLS is checked for recovery of a planted
coefficient under confounding that provably breaks OLS.
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from eicu_iv_diagnostics import (  # noqa: E402
    first_stage_diagnostics,
    ols,
    overlap_by_quantile,
    standardized_mean_differences,
    two_stage_least_squares,
)


def confounded_iv_design(n=20000, effect=1.5, instrument_strength=1.2, confounding=1.0, seed=0):
    """Y = effect*D + confounding*U + noise, with D driven by Z and U.

    U is unobserved, so OLS of Y on D is biased upward by ``confounding``, while
    2SLS using Z should recover ``effect``.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = instrument_strength * z + confounding * u + rng.normal(size=n) * 0.5
    y = effect * d + confounding * u + rng.normal(size=n) * 0.5
    return z, d, y, u


class OlsTest(unittest.TestCase):
    def test_recovers_exact_coefficients_without_noise(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, 3))
        beta = np.array([2.0, -1.0, 0.5])
        y = 4.0 + x @ beta

        fit = ols(x, y)
        np.testing.assert_allclose(fit["coef"], np.r_[4.0, beta], atol=1e-8)
        self.assertAlmostEqual(fit["r_squared"], 1.0, places=10)

    def test_matches_normal_equations(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(300, 2))
        y = rng.normal(size=300)

        fit = ols(x, y)
        design = np.hstack([np.ones((300, 1)), x])
        expected = np.linalg.solve(design.T @ design, design.T @ y)
        np.testing.assert_allclose(fit["coef"], expected, atol=1e-10)

    def test_robust_standard_errors_are_positive_and_shrink_with_n(self):
        small = ols(*_noisy(200, seed=2))
        large = ols(*_noisy(20000, seed=2))
        self.assertTrue((small["stderr"] > 0).all())
        self.assertLess(large["stderr"][1], small["stderr"][1])

    def test_reports_sample_size_and_rank(self):
        fit = ols(*_noisy(150, seed=3))
        self.assertEqual(fit["n"], 150)
        self.assertEqual(fit["k"], 2)


def _noisy(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    y = 1.0 + 2.0 * x[:, 0] + rng.normal(size=n)
    return x, y


class TwoStageLeastSquaresTest(unittest.TestCase):
    def test_recovers_the_planted_effect_under_confounding(self):
        z, d, y, _ = confounded_iv_design(effect=1.5, seed=0)
        fit = two_stage_least_squares(z, d, outcome=y)
        self.assertAlmostEqual(fit["effect"], 1.5, delta=0.05)

    def test_ols_is_biased_where_2sls_is_not(self):
        """The comparison that motivates using an IV at all."""
        z, d, y, _ = confounded_iv_design(effect=1.5, confounding=1.0, seed=1)

        naive = ols(d, y)["coef"][1]
        iv = two_stage_least_squares(z, d, outcome=y)["effect"]

        self.assertGreater(naive, 1.7)  # confounded upward
        self.assertAlmostEqual(iv, 1.5, delta=0.05)

    def test_recovers_effect_with_exogenous_controls(self):
        rng = np.random.default_rng(4)
        n = 20000
        x = rng.normal(size=(n, 2))
        z = rng.normal(size=n)
        u = rng.normal(size=n)
        d = 1.2 * z + u + x @ np.array([0.4, -0.3]) + rng.normal(size=n) * 0.5
        y = 2.0 * d + u + x @ np.array([1.0, 0.5]) + rng.normal(size=n) * 0.5

        fit = two_stage_least_squares(z, d, covariates=x, outcome=y)
        self.assertAlmostEqual(fit["effect"], 2.0, delta=0.05)

    def test_standard_error_covers_the_truth(self):
        z, d, y, _ = confounded_iv_design(effect=1.5, seed=5)
        fit = two_stage_least_squares(z, d, outcome=y)
        lower = fit["effect"] - 1.96 * fit["effect_stderr"]
        upper = fit["effect"] + 1.96 * fit["effect_stderr"]
        self.assertLess(lower, 1.5)
        self.assertGreater(upper, 1.5)

    def test_weak_instrument_widens_the_interval(self):
        strong = two_stage_least_squares(
            *_iv_arrays(instrument_strength=1.5, seed=6)
        )
        weak = two_stage_least_squares(*_iv_arrays(instrument_strength=0.05, seed=6))
        self.assertGreater(weak["effect_stderr"], strong["effect_stderr"])


def _iv_arrays(instrument_strength, seed):
    z, d, y, _ = confounded_iv_design(
        instrument_strength=instrument_strength, seed=seed
    )
    return z, d, None, y


class FirstStageTest(unittest.TestCase):
    def test_strong_instrument_has_large_partial_f(self):
        z, d, _, _ = confounded_iv_design(instrument_strength=1.5, seed=7)
        diag = first_stage_diagnostics(z, d)
        self.assertGreater(diag["partial_f"], 100.0)
        self.assertFalse(diag["weak_instrument_warning"])

    def test_irrelevant_instrument_is_flagged_weak(self):
        rng = np.random.default_rng(8)
        n = 5000
        z = rng.normal(size=n)
        d = rng.normal(size=n)  # no relationship to z at all
        diag = first_stage_diagnostics(z, d)
        self.assertLess(diag["partial_f"], 10.0)
        self.assertTrue(diag["weak_instrument_warning"])

    def test_partial_r2_between_zero_and_one(self):
        z, d, _, _ = confounded_iv_design(seed=9)
        diag = first_stage_diagnostics(z, d)
        self.assertGreater(diag["partial_r2"], 0.0)
        self.assertLess(diag["partial_r2"], 1.0)

    def test_first_stage_coefficient_matches_the_design(self):
        z, d, _, _ = confounded_iv_design(instrument_strength=1.2, seed=10)
        diag = first_stage_diagnostics(z, d)
        self.assertAlmostEqual(diag["instrument_coef"], 1.2, delta=0.05)

    def test_controls_reduce_partial_f_when_they_absorb_the_instrument(self):
        rng = np.random.default_rng(11)
        n = 5000
        z = rng.normal(size=n)
        control = z + rng.normal(size=n) * 0.01  # nearly collinear with z
        d = 1.0 * z + rng.normal(size=n)

        without = first_stage_diagnostics(z, d)["partial_f"]
        with_control = first_stage_diagnostics(z, d, covariates=control)["partial_f"]
        self.assertLess(with_control, without)


class BalanceTest(unittest.TestCase):
    def test_detects_imbalance_when_z_encodes_case_mix(self):
        rng = np.random.default_rng(12)
        n = 4000
        z = rng.normal(size=n)
        severity = 2.0 * z + rng.normal(size=n)  # instrument encodes patient mix
        benign = rng.normal(size=n)

        result = standardized_mean_differences(
            np.column_stack([severity, benign]), z, names=["severity", "benign"]
        )
        by_name = {r["covariate"]: r for r in result}
        self.assertGreater(abs(by_name["severity"]["smd"]), 1.0)
        self.assertLess(abs(by_name["benign"]["smd"]), 0.2)

    def test_results_are_sorted_by_absolute_imbalance(self):
        rng = np.random.default_rng(13)
        n = 2000
        z = rng.normal(size=n)
        covs = np.column_stack([rng.normal(size=n), 3.0 * z + rng.normal(size=n)])
        result = standardized_mean_differences(covs, z, names=["flat", "tilted"])
        self.assertEqual(result[0]["covariate"], "tilted")


class OverlapTest(unittest.TestCase):
    def test_treatment_rate_increases_with_the_instrument(self):
        z, d, _, _ = confounded_iv_design(instrument_strength=2.0, seed=14)
        d_binary = (d > np.median(d)).astype(float)
        rows = overlap_by_quantile(z, d_binary, n_bins=5)
        rates = [r["treatment_rate"] for r in rows]
        self.assertEqual(rates, sorted(rates))
        self.assertGreater(rates[-1] - rates[0], 0.5)

    def test_bins_partition_the_sample(self):
        z, d, _, _ = confounded_iv_design(seed=15)
        rows = overlap_by_quantile(z, (d > 0).astype(float), n_bins=5)
        self.assertEqual(sum(r["n"] for r in rows), len(z))

    def test_constant_instrument_yields_no_bins(self):
        z = np.ones(100)
        self.assertEqual(overlap_by_quantile(z, np.zeros(100)), [])


if __name__ == "__main__":
    unittest.main()
