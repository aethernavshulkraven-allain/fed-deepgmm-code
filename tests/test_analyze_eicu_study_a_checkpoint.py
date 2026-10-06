"""Tests for the standalone post-hoc per-client Study A evaluation script.

Deliberately built around a hand-constructed checkpoint + scenario, not a real
training run, so the arithmetic can be checked against known answers rather
than "looks plausible."
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example"))

from models.mlp_model import MLPModel  # noqa: E402

import analyze_eicu_study_a_checkpoint as post_hoc  # noqa: E402


def make_zero_width_model(input_dim, bias_value):
    """A model matching production's EICU_HIDDEN_WIDTHS architecture (so a
    state dict built here loads cleanly into build_models()'s reconstruction),
    with every weight zeroed except the final bias -- so g(x) == bias_value
    for any x, a fully hand-verifiable model.
    """
    model = MLPModel(
        input_dim=input_dim, layer_widths=post_hoc.EICU_HIDDEN_WIDTHS, activation=nn.LeakyReLU
    ).double()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.model[-1].bias.fill_(bias_value)
    return model


class AggregateTest(unittest.TestCase):
    def test_equal_client_and_sample_weighted_differ_under_imbalance(self):
        rows = [
            {"mse": 10.0, "n": 1},
            {"mse": 0.0, "n": 99},
        ]
        agg = post_hoc.aggregate(rows, "mse")
        self.assertAlmostEqual(agg["equal_client"], 5.0, places=10)
        self.assertAlmostEqual(agg["sample_weighted"], 0.1, places=10)

    def test_equal_client_and_sample_weighted_agree_under_balance(self):
        rows = [{"mse": 3.0, "n": 10}, {"mse": 7.0, "n": 10}]
        agg = post_hoc.aggregate(rows, "mse")
        self.assertAlmostEqual(agg["equal_client"], agg["sample_weighted"], places=10)

    def test_worst_and_median_are_correct(self):
        rows = [{"mse": v, "n": 1} for v in (1.0, 5.0, 9.0)]
        agg = post_hoc.aggregate(rows, "mse")
        self.assertEqual(agg["worst"], 9.0)
        self.assertEqual(agg["median"], 5.0)


class EvaluateTest(unittest.TestCase):
    """g is forced to a known constant function so predicted counterfactuals,
    structural error, and ATE error are all hand-computable.
    """

    def setUp(self):
        self.n_covariates = 2
        self.g = make_zero_width_model(self.n_covariates + 1, bias_value=2.0)  # g(x) == 2.0 always
        self.f = make_zero_width_model(self.n_covariates + 1, bias_value=1.0)  # f(z) == 1.0 always

    def _split(self, n=4):
        rng = np.random.default_rng(0)
        treatment = np.array([1.0, 0.0, 1.0, 0.0])[:n]
        covariates = rng.normal(size=(n, self.n_covariates))
        x = np.column_stack([treatment, covariates])
        z = np.column_stack([treatment, covariates])  # shape parity only, values unused
        true_g = np.full(n, 3.0)  # deliberately different from g's constant 2.0
        y = true_g + 0.5
        client_id = np.array([0, 0, 1, 1])[:n]
        return {
            "x": x,
            "z": z,
            "y": y.reshape(-1, 1),
            "g": true_g.reshape(-1, 1),
            "client_id": client_id,
            "g0_treated": np.full(n, 3.0).reshape(-1, 1),
            "g0_control": np.full(n, 1.0).reshape(-1, 1),
            "true_effect": np.full(n, 2.0).reshape(-1, 1),
        }

    def test_g_prediction_is_the_known_constant(self):
        evald = post_hoc.evaluate(self.g, self.f, self._split())
        np.testing.assert_allclose(evald["g_pred"], 2.0)

    def test_structural_squared_error_matches_hand_computation(self):
        # true_g == 3.0, g_pred == 2.0 -> squared error == 1.0 everywhere.
        evald = post_hoc.evaluate(self.g, self.f, self._split())
        np.testing.assert_allclose(evald["squared_error"], 1.0)

    def test_predicted_effect_is_zero_for_a_constant_g(self):
        # g(x_treated) == g(x_control) == 2.0 regardless of D -> pred_effect == 0.
        evald = post_hoc.evaluate(self.g, self.f, self._split())
        np.testing.assert_allclose(evald["pred_effect"], 0.0)

    def test_individual_effect_error_matches_true_effect_when_predicted_effect_is_zero(self):
        # individual_effect_error = pred_effect - true_effect = 0 - 2.0 = -2.0.
        evald = post_hoc.evaluate(self.g, self.f, self._split())
        np.testing.assert_allclose(evald["individual_effect_error"], -2.0)
        np.testing.assert_allclose(evald["true_effect"], 2.0)


class PerClientMetricsTest(unittest.TestCase):
    def _evald(self, client_id, squared_error, pred_effect, true_effect):
        pred_effect = np.asarray(pred_effect)
        true_effect = np.asarray(true_effect)
        n = len(client_id)
        return {
            "client_id": np.asarray(client_id),
            "squared_error": np.asarray(squared_error),
            "pred_effect": pred_effect,
            "true_effect": true_effect,
            "individual_effect_error": pred_effect - true_effect,
            "f_pred": np.ones(n),
            "y": np.ones(n),
            "g_pred": np.ones(n),
        }

    def test_groups_correctly_by_client_id(self):
        evald = self._evald(
            client_id=[0, 0, 1],
            squared_error=[1.0, 3.0, 10.0],
            pred_effect=[0.5, 0.5, 0.5],
            true_effect=[0.5, 0.5, 0.5],
        )
        rows = post_hoc.per_client_metrics(evald, {"0": 100, "1": 200})
        by_client = {r["hospital_id"]: r for r in rows}
        self.assertEqual(by_client[100]["n"], 2)
        self.assertAlmostEqual(by_client[100]["mse"], 2.0)  # mean(1.0, 3.0)
        self.assertEqual(by_client[200]["n"], 1)
        self.assertAlmostEqual(by_client[200]["mse"], 10.0)

    def test_abs_ate_error_matches_hand_computation(self):
        # true_ate = mean(2.0, 2.0) = 2.0; pred_ate = mean(-3.0, -5.0) = -4.0
        # -> |pred_ate - true_ate| = 6.0.
        evald = self._evald(
            client_id=[0, 0],
            squared_error=[0.0, 0.0],
            pred_effect=[-3.0, -5.0],
            true_effect=[2.0, 2.0],
        )
        rows = post_hoc.per_client_metrics(evald, {"0": 7})
        self.assertAlmostEqual(rows[0]["true_ate"], 2.0)
        self.assertAlmostEqual(rows[0]["pred_ate"], -4.0)
        self.assertAlmostEqual(rows[0]["per_client_abs_ate_error"], 6.0)

    def test_ate_error_and_individual_effect_mae_diverge_under_sign_cancellation(self):
        """metric_policy.md: "ATE error can be small because positive and
        negative individual errors cancel; individual-effect MAE cannot."
        Row-level effect errors are -2.0 and +2.0 (individual_effect_mae must
        see both), but they average to 0 within the client (abs ATE error
        must be 0) -- the two metrics are deliberately made to disagree here.
        """
        evald = self._evald(
            client_id=[0, 0],
            squared_error=[0.0, 0.0],
            pred_effect=[0.0, 4.0],  # mean = 2.0
            true_effect=[2.0, 2.0],  # mean = 2.0
        )
        rows = post_hoc.per_client_metrics(evald, {"0": 7})
        self.assertAlmostEqual(rows[0]["per_client_abs_ate_error"], 0.0)
        self.assertAlmostEqual(rows[0]["individual_effect_mae"], 2.0)  # mean(|-2|, |2|)


class AggregateAteErrorTest(unittest.TestCase):
    def test_equal_client_ate_error_averages_signed_ates_before_abs(self):
        # client A: pred_ate=3.0, true_ate=1.0 (signed diff +2.0)
        # client B: pred_ate=-1.0, true_ate=1.0 (signed diff -2.0)
        # Naively averaging |diff| per client would give 2.0, but
        # metric_policy.md's formula averages the signed ATEs first:
        # mean(pred_ate) - mean(true_ate) = 1.0 - 1.0 = 0.0 -> |0.0| = 0.0.
        rows = [
            {"pred_ate": 3.0, "true_ate": 1.0, "n": 1, "per_client_abs_ate_error": 2.0},
            {"pred_ate": -1.0, "true_ate": 1.0, "n": 1, "per_client_abs_ate_error": 2.0},
        ]
        agg = post_hoc.aggregate_ate_error(rows)
        self.assertAlmostEqual(agg["equal_client"], 0.0)
        # The per-client distribution stats still see the (larger) per-client values.
        self.assertAlmostEqual(agg["median"], 2.0)
        self.assertAlmostEqual(agg["worst"], 2.0)

    def test_sample_weighted_ate_error_uses_n_weighted_means(self):
        rows = [
            {"pred_ate": 4.0, "true_ate": 0.0, "n": 1, "per_client_abs_ate_error": 4.0},
            {"pred_ate": 0.0, "true_ate": 0.0, "n": 9, "per_client_abs_ate_error": 0.0},
        ]
        agg = post_hoc.aggregate_ate_error(rows)
        # weighted pred_ate mean = (1*4 + 9*0)/10 = 0.4; weighted true_ate mean = 0.0
        self.assertAlmostEqual(agg["sample_weighted"], 0.4)
        # equal-client mean pred_ate = (4+0)/2 = 2.0
        self.assertAlmostEqual(agg["equal_client"], 2.0)


class EndToEndCliTest(unittest.TestCase):
    """Hand-builds a checkpoint + scenario (no real training run) and checks
    the CLI's output files against hand-computed expectations.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="study_a_posthoc_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_end_to_end_matches_hand_computation(self):
        input_dim = 3  # 1 treatment + 2 covariates
        g = make_zero_width_model(input_dim, bias_value=0.0)  # g(x) == 0.0
        f = make_zero_width_model(input_dim, bias_value=1.0)  # f(z) == 1.0

        checkpoint_path = os.path.join(self.tmpdir, "best_validation.pt")
        torch.save(
            {
                "round": 5,
                "checkpoint_type": "best_validation",
                "g_state_dict": g.state_dict(),
                "f_state_dict": f.state_dict(),
                "effective_config": {"input_dim_g": input_dim, "input_dim_f": input_dim},
            },
            checkpoint_path,
        )

        n = 4
        treatment = np.array([1.0, 0.0, 1.0, 0.0])
        covariates = np.zeros((n, 2))
        x = np.column_stack([treatment, covariates])
        true_g = np.array([2.0, 2.0, 2.0, 2.0])  # g0 is constant 2.0 regardless of D here
        y = true_g.copy()
        client_id = np.array([10, 10, 20, 20])

        scenario_path = os.path.join(self.tmpdir, "toy_seed0.npz")
        np.savez(
            scenario_path,
            splits=["test"],
            test_x=x,
            test_z=x,
            test_y=y.reshape(-1, 1),
            test_g=true_g.reshape(-1, 1),
            test_client_id=client_id,
            test_g0_treated=np.full(n, 2.0).reshape(-1, 1),
            test_g0_control=np.full(n, 2.0).reshape(-1, 1),
            test_true_effect=np.zeros((n, 1)),  # constant g0 -> zero true effect
        )
        import json

        meta_path = os.path.join(self.tmpdir, "toy_seed0_metadata.json")
        with open(meta_path, "w") as handle:
            json.dump({"client_code_to_hospital": {"10": 111, "20": 222}}, handle)

        exit_code = post_hoc.main(
            ["--checkpoint", checkpoint_path, "--scenario", scenario_path, "--split", "test", "--out", self.tmpdir]
        )
        self.assertEqual(exit_code, 0)

        # g predicts 0.0 everywhere, true_g is 2.0 -> squared error 4.0 everywhere,
        # identical for both (equally sized, n=2 each) clients.
        summary_path = os.path.join(self.tmpdir, "per_client_eval_best_validation_test_summary.json")
        with open(summary_path) as handle:
            summary = json.load(handle)
        self.assertEqual(summary["n_clients"], 2)
        self.assertAlmostEqual(summary["aggregates"]["mse"]["equal_client"], 4.0, places=8)
        self.assertAlmostEqual(summary["aggregates"]["mse"]["sample_weighted"], 4.0, places=8)
        # g is constant in D -> predicted effect is 0, matching the true (also
        # constant-in-D) g0 exactly -> both ATE error and individual-effect
        # MAE are 0 (a case where they can't diverge, since there's no error).
        self.assertAlmostEqual(summary["aggregates"]["absolute_ate_error"]["equal_client"], 0.0, places=8)
        self.assertAlmostEqual(summary["aggregates"]["individual_effect_mae"]["equal_client"], 0.0, places=8)

        csv_path = os.path.join(self.tmpdir, "per_client_eval_best_validation_test.csv")
        self.assertTrue(os.path.exists(csv_path))
        report_path = os.path.join(self.tmpdir, "per_client_eval_best_validation_test.md")
        self.assertTrue(os.path.exists(report_path))

    def test_missing_input_dims_raises(self):
        with self.assertRaises(ValueError):
            post_hoc.build_models({"input_dim_g": 0, "input_dim_f": 0})

    def test_checkpoint_missing_required_keys_raises(self):
        path = os.path.join(self.tmpdir, "bad.pt")
        torch.save({"round": 0}, path)
        with self.assertRaises(ValueError):
            post_hoc.load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
