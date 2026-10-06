"""Tests for federated aggregation weighting (P0 fix).

The paper's federated objective (Section 3) is the equal-client average
``U(theta,tau) = (1/N) sum_i U^i(theta,tau)``. The pre-existing implementation
of ``FedAvgAPI._aggregate``/``_aggregate_reg`` instead weighted each client by
its sample count (``w_i = n_i / sum_j n_j``), i.e. ordinary FedAvg -- silently
letting large clients dominate the global model, which is exactly the failure
mode this fix targets for eICU's wildly unequal hospital sizes.

These tests protect three things:
1. ``compute_client_weights`` computes both schemes correctly, including under
   partial participation (K < N).
2. ``weighted_average_state_dicts`` -- the function both FedGDA's direct
   aggregation and FedOGDA's pseudogradient/delta derive from -- reproduces the
   *exact* original inline formula for ``sample_size`` mode (so every existing
   result in the repo stays reproducible), and gives a visibly, numerically
   different answer for ``uniform_clients`` under unequal client sizes.
3. ``get_effective_config`` resolves/defaults the new field correctly.
"""

import os
import sys
import unittest
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from fedgmm.sp_decentralized_mnist_lr_example.experiment_utils import (  # noqa: E402
    AGGREGATION_WEIGHTING_CHOICES,
    DEFAULT_AGGREGATION_WEIGHTING,
    EFFECTIVE_CONFIG_FIELDS,
    check_eicu_aggregation_weighting,
    check_eicu_objective_mode,
    compute_client_weights,
    get_effective_config,
    weighted_average_state_dicts,
)


class ComputeClientWeightsTest(unittest.TestCase):
    def test_uniform_ignores_sample_counts(self):
        weights = compute_client_weights([1, 99, 500], "uniform_clients")
        self.assertEqual(weights, [1 / 3, 1 / 3, 1 / 3])

    def test_uniform_weight_is_one_over_k_not_one_over_n(self):
        """Partial participation: K participating clients, not N total clients."""
        # Only 2 of some larger total-client-count sample this round.
        weights = compute_client_weights([7, 3], "uniform_clients")
        self.assertEqual(weights, [0.5, 0.5])

    def test_uniform_sums_to_one(self):
        weights = compute_client_weights([1, 2, 3, 4, 5], "uniform_clients")
        self.assertAlmostEqual(sum(weights), 1.0, places=12)

    def test_sample_size_matches_proportions(self):
        weights = compute_client_weights([1, 99], "sample_size")
        self.assertAlmostEqual(weights[0], 0.01, places=12)
        self.assertAlmostEqual(weights[1], 0.99, places=12)

    def test_sample_size_sums_to_one(self):
        weights = compute_client_weights([3, 5, 12], "sample_size")
        self.assertAlmostEqual(sum(weights), 1.0, places=12)

    def test_sample_size_matches_legacy_inline_division(self):
        """weight_i = n_i / sum_j n_j, exactly what the old inline code computed."""
        counts = [4, 6, 10]
        total = sum(counts)
        expected = [c / total for c in counts]
        self.assertEqual(compute_client_weights(counts, "sample_size"), expected)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            compute_client_weights([1, 2], "equal_weight_typo")

    def test_empty_counts_raises(self):
        with self.assertRaises(ValueError):
            compute_client_weights([], "uniform_clients")

    def test_sample_size_zero_total_raises(self):
        with self.assertRaises(ValueError):
            compute_client_weights([0, 0], "sample_size")

    def test_choices_constant_matches_implemented_modes(self):
        for mode in AGGREGATION_WEIGHTING_CHOICES:
            # Must not raise for any advertised choice.
            compute_client_weights([1, 2], mode)


class WeightedAverageStateDictsTest(unittest.TestCase):
    def _legacy_inline_aggregate(self, w_locals):
        """Literal transcription of the original fedavg_api.py `_aggregate` body,
        kept here only as a reference oracle to prove bit-exact equivalence.
        """
        training_num = sum(num for num, _ in w_locals)
        sample_num, g = w_locals[0]
        g = dict(g)
        for k in g.keys():
            for i in range(len(w_locals)):
                local_sample_number, local_g = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    g[k] = local_g[k] * w
                else:
                    g[k] = g[k] + local_g[k] * w
        return g

    def test_sample_size_mode_is_bit_exact_with_the_legacy_formula(self):
        """The change this test protects: existing (pre-fix) results must stay
        reproducible under the default `sample_size` mode.
        """
        w_locals = [
            (1, {"w": torch.tensor([10.0, -3.0])}),
            (99, {"w": torch.tensor([0.0, 1.0])}),
            (40, {"w": torch.tensor([5.0, 5.0])}),
        ]
        legacy = self._legacy_inline_aggregate(w_locals)

        counts = [n for n, _ in w_locals]
        weights = compute_client_weights(counts, "sample_size")
        state_dicts = [sd for _, sd in w_locals]
        result = weighted_average_state_dicts(state_dicts, weights)

        self.assertTrue(torch.equal(result["w"], legacy["w"]))

    def test_uniform_and_sample_weighted_give_visibly_different_known_answers(self):
        """The exact test the co-author asked for: unequal client sizes where
        uniform vs. sample-weighted aggregation produce visibly different,
        hand-computable answers.
        """
        # Client A: 1 sample, parameter value 10.0. Client B: 99 samples, value 0.0.
        # dtype=float64 to match production (every model in this repo is `.double()`d).
        sample_counts = [1, 99]
        state_dicts = [
            {"w": torch.tensor(10.0, dtype=torch.float64)},
            {"w": torch.tensor(0.0, dtype=torch.float64)},
        ]

        uniform_weights = compute_client_weights(sample_counts, "uniform_clients")
        uniform_result = weighted_average_state_dicts(state_dicts, uniform_weights)

        sample_weights = compute_client_weights(sample_counts, "sample_size")
        sample_result = weighted_average_state_dicts(state_dicts, sample_weights)

        # Known answers, by hand: uniform = 0.5*10 + 0.5*0 = 5.0;
        # sample-weighted = 0.01*10 + 0.99*0 = 0.1.
        self.assertAlmostEqual(uniform_result["w"].item(), 5.0, places=10)
        self.assertAlmostEqual(sample_result["w"].item(), 0.1, places=10)

        # Not just numerically distinguishable -- a large, obvious gap: under
        # sample-weighted aggregation the 99-sample client (holding the value
        # farthest from truth in this construction) would dominate the global
        # model almost completely; uniform prevents that by construction.
        self.assertGreater(abs(uniform_result["w"].item() - sample_result["w"].item()), 4.5)

    def test_multiple_keys_are_aggregated_independently(self):
        state_dicts = [
            {"a": torch.tensor(1.0), "b": torch.tensor(100.0)},
            {"a": torch.tensor(3.0), "b": torch.tensor(300.0)},
        ]
        weights = [0.5, 0.5]
        result = weighted_average_state_dicts(state_dicts, weights)
        self.assertAlmostEqual(result["a"].item(), 2.0)
        self.assertAlmostEqual(result["b"].item(), 200.0)

    def test_does_not_mutate_the_first_clients_state_dict(self):
        """The legacy code mutated w_locals[0]'s dict in place, aliasing the
        return value with a client's own state. The replacement must not.
        """
        original = torch.tensor([1.0, 2.0])
        client_a = {"w": original}
        client_b = {"w": torch.tensor([3.0, 4.0])}
        weighted_average_state_dicts([client_a, client_b], [0.5, 0.5])
        self.assertTrue(torch.equal(client_a["w"], original))

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            weighted_average_state_dicts([{"w": torch.tensor(1.0)}], [0.5, 0.5])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            weighted_average_state_dicts([], [])

    def test_works_with_plain_floats_not_only_tensors(self):
        """compute_client_weights/weighted_average_state_dicts are dependency-
        light pure functions -- confirm they don't secretly require torch.
        """
        state_dicts = [{"v": 2.0}, {"v": 8.0}]
        result = weighted_average_state_dicts(state_dicts, [0.5, 0.5])
        self.assertAlmostEqual(result["v"], 5.0)


class EffectiveConfigAggregationWeightingTest(unittest.TestCase):
    def test_defaults_to_sample_size_when_unset(self):
        """Backward compatibility: configs written before this option existed
        must resolve to the legacy behavior, not silently switch to uniform.
        """
        config = get_effective_config(SimpleNamespace(dataset="abs"))
        self.assertEqual(config["aggregation_weighting"], "sample_size")
        self.assertEqual(DEFAULT_AGGREGATION_WEIGHTING, "sample_size")

    def test_explicit_uniform_clients_is_resolved(self):
        config = get_effective_config(
            SimpleNamespace(dataset="eicu_semisynth", aggregation_weighting="uniform_clients")
        )
        self.assertEqual(config["aggregation_weighting"], "uniform_clients")

    def test_field_is_part_of_the_effective_config_schema(self):
        self.assertIn("aggregation_weighting", EFFECTIVE_CONFIG_FIELDS)
        config = get_effective_config(SimpleNamespace())
        for field in EFFECTIVE_CONFIG_FIELDS:
            self.assertIn(field, config)


class EicuAggregationWeightingGuardTest(unittest.TestCase):
    def test_non_eicu_dataset_allows_sample_size(self):
        check_eicu_aggregation_weighting("abs", "sample_size", "")

    def test_eicu_confirmatory_role_rejects_sample_size(self):
        with self.assertRaises(ValueError):
            check_eicu_aggregation_weighting("eicu_semisynth", "sample_size", "")

    def test_eicu_confirmatory_role_rejects_sample_size_even_with_wrong_role_label(self):
        with self.assertRaises(ValueError):
            check_eicu_aggregation_weighting("eicu_semisynth", "sample_size", "confirmatory")

    def test_eicu_uniform_clients_always_allowed(self):
        check_eicu_aggregation_weighting("eicu_semisynth", "uniform_clients", "")
        check_eicu_aggregation_weighting("eicu_semisynth", "uniform_clients", "aggregation_ablation")

    def test_eicu_ablation_role_authorizes_sample_size(self):
        check_eicu_aggregation_weighting("eicu_semisynth", "sample_size", "aggregation_ablation")

    def test_eicu_ablation_role_does_not_authorize_other_weightings(self):
        # campaign_role alone is not a blanket bypass -- the exception is
        # specifically for sample_size, the one value the ablation role
        # exists to test.
        with self.assertRaises(ValueError):
            check_eicu_aggregation_weighting("eicu_semisynth", "not_a_real_mode", "aggregation_ablation")


class EicuObjectiveModeGuardTest(unittest.TestCase):
    def test_non_eicu_dataset_allows_legacy(self):
        check_eicu_objective_mode("abs", "legacy")

    def test_eicu_rejects_legacy(self):
        with self.assertRaises(ValueError):
            check_eicu_objective_mode("eicu_semisynth", "legacy")

    def test_eicu_allows_paper_aligned(self):
        check_eicu_objective_mode("eicu_semisynth", "paper_aligned")


if __name__ == "__main__":
    unittest.main()
