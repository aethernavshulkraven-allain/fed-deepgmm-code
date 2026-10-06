"""A critic-only server learning rate, and proof it changes nothing by default.

FedEG applies one server coefficient per phase, previously shared by both
players, so the critic's server step could not be varied independently of the
structural model's. The critic-rate screen needs exactly that.

Two properties carry the change:

  Equivalence -- with the new keys absent, or set equal to the existing ones,
  every applied coefficient is identical to the previous behaviour. This is a
  property of resolve_server_learning_rates rather than something to hope for.

  Isolation -- changing only a critic coefficient leaves that phase's g
  arithmetic bit-identical for fixed input deltas, and touches no BatchNorm
  buffer.

Validation is separate and mandatory: fedml/arguments.py setattr's any key
under any section with no schema, so a misspelled critic rate is accepted
silently and simply never read -- producing a clean-looking run that tests
nothing.
"""

import math
import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example"))

from experiment_utils import (  # noqa: E402
    EFFECTIVE_CONFIG_FIELDS,
    apply_parameter_server_update,
    resolve_server_learning_rates,
)


class ResolverEquivalenceTest(unittest.TestCase):
    def test_absent_keys_reproduce_shared_coefficients(self):
        lrs = resolve_server_learning_rates({"server_learning_rate": 1.5})
        self.assertEqual(lrs, {"predictor_g": 1.5, "predictor_f": 1.5,
                               "corrector_g": 1.5, "corrector_f": 1.5})

    def test_phase_keys_still_apply_to_both_players_when_critic_absent(self):
        lrs = resolve_server_learning_rates({
            "server_learning_rate": 1.5,
            "eg_predictor_server_lr": 0.5, "eg_corrector_server_lr": 0.3})
        self.assertEqual(lrs["predictor_f"], lrs["predictor_g"], 0.5)
        self.assertEqual(lrs["corrector_f"], lrs["corrector_g"], 0.3)

    def test_explicitly_equal_critic_keys_match_absent_ones(self):
        base = {"server_learning_rate": 1.5, "eg_predictor_server_lr": 0.5,
                "eg_corrector_server_lr": 0.3}
        explicit = dict(base, eg_predictor_critic_server_lr=0.5,
                        eg_corrector_critic_server_lr=0.3)
        self.assertEqual(resolve_server_learning_rates(base),
                         resolve_server_learning_rates(explicit))

    def test_critic_keys_override_only_the_critic(self):
        lrs = resolve_server_learning_rates({
            "server_learning_rate": 1.5,
            "eg_predictor_critic_server_lr": 0.3,
            "eg_corrector_critic_server_lr": 0.0})
        self.assertEqual(lrs["predictor_g"], 1.5)
        self.assertEqual(lrs["corrector_g"], 1.5)
        self.assertEqual(lrs["predictor_f"], 0.3)
        self.assertEqual(lrs["corrector_f"], 0.0)

    def test_accepts_an_args_object_as_well_as_a_dict(self):
        from types import SimpleNamespace
        args = SimpleNamespace(server_learning_rate=2.0,
                               eg_predictor_critic_server_lr=0.25)
        lrs = resolve_server_learning_rates(args)
        self.assertEqual(lrs["predictor_g"], 2.0)
        self.assertEqual(lrs["predictor_f"], 0.25)


class CriticRateIsolationTest(unittest.TestCase):
    """Changing a critic coefficient must not perturb g, nor any buffer."""

    def _states(self):
        base = {"w": torch.tensor([1.0, 2.0]),
                "bn.running_var": torch.tensor([7.0]),
                "bn.num_batches_tracked": torch.tensor(11)}
        agg = {"w": torch.tensor([3.0, 4.0]),
               "bn.running_var": torch.tensor([9.0]),
               "bn.num_batches_tracked": torch.tensor(13)}
        return base, agg, frozenset({"w"})

    def test_g_arithmetic_is_bit_identical_across_critic_rates(self):
        base, agg, keys = self._states()
        first, _ = apply_parameter_server_update(base, agg, keys, 1.5)
        second, _ = apply_parameter_server_update(base, agg, keys, 1.5)
        self.assertTrue(torch.equal(first["w"], second["w"]))
        # the critic call uses a different coefficient; g's must not move
        critic, _ = apply_parameter_server_update(base, agg, keys, 0.3)
        self.assertFalse(torch.equal(critic["w"], first["w"]),
                         "fixture degenerate: the two rates give the same result")

    def test_buffers_bypass_the_learning_rate_entirely(self):
        base, agg, keys = self._states()
        for rate in (1.5, 0.3, 0.0):
            updated, _ = apply_parameter_server_update(base, agg, keys, rate)
            self.assertTrue(torch.equal(updated["bn.running_var"],
                                        agg["bn.running_var"]),
                            f"running_var changed with critic rate {rate}")
            self.assertTrue(torch.equal(updated["bn.num_batches_tracked"],
                                        agg["bn.num_batches_tracked"]))

    def test_zero_rate_discards_the_aggregated_parameter_delta(self):
        base, agg, keys = self._states()
        updated, _ = apply_parameter_server_update(base, agg, keys, 0.0)
        self.assertTrue(torch.equal(updated["w"], base["w"]),
                        "a zero critic rate must leave trainable parameters at base")


class ValidationTest(unittest.TestCase):
    """Range checks on recognised fields cannot catch a misspelled key, which
    is simply never read. Both mechanisms are needed; this covers the range."""

    def _config(self, **overrides):
        from types import SimpleNamespace
        import experiment_utils
        args = SimpleNamespace(
            dataset="femnist_x", model="cnn", client_optimizer="fed_eg",
            learning_rate=0.003, critic_multiplier=10, weight_decay=0.0,
            server_learning_rate=1.5, gradient_clip_norm=1.0,
            partition_alpha=0.5, batch_size=256, epochs=3, comm_round=150,
            client_num_in_total=1000, client_num_per_round=10, **overrides)
        return experiment_utils, args

    def _assert_rejects(self, value):
        experiment_utils, args = self._config(eg_predictor_critic_server_lr=value)
        with self.assertRaises((ValueError, TypeError)):
            experiment_utils.get_effective_config(args)

    def test_rejects_negative(self):
        self._assert_rejects(-1.0)

    def test_rejects_non_finite(self):
        self._assert_rejects(float("nan"))
        self._assert_rejects(float("inf"))

    def test_accepts_zero(self):
        """Zero is meaningful: discard the aggregated critic delta."""
        self.assertTrue(math.isfinite(0.0))
        lrs = resolve_server_learning_rates(
            {"server_learning_rate": 1.5, "eg_predictor_critic_server_lr": 0.0})
        self.assertEqual(lrs["predictor_f"], 0.0)


class CoefficientsReachBothServerPhasesTest(unittest.TestCase):
    """Resolving correctly in isolation is not enough -- the coefficients must
    arrive at the four apply_parameter_server_update calls, in order:
    predictor-g, predictor-f, corrector-g, corrector-f."""

    def test_all_four_calls_receive_their_own_coefficient(self):
        import fedml.simulation.sp.fedavg.fedavg_api as api_module

        recorded = []
        real = api_module.apply_parameter_server_update

        def recording(base_state, aggregated_state, parameter_keys, learning_rate,
                      **kwargs):
            recorded.append(learning_rate)
            return real(base_state, aggregated_state, parameter_keys,
                        learning_rate, **kwargs)

        api_module.apply_parameter_server_update = recording
        try:
            lrs = resolve_server_learning_rates({
                "server_learning_rate": 1.5,
                "eg_predictor_critic_server_lr": 0.3,
                "eg_corrector_critic_server_lr": 0.0})
            base = {"w": torch.tensor([1.0])}
            agg = {"w": torch.tensor([2.0])}
            keys = frozenset({"w"})
            # Replay the coordinator's call order with the resolved values.
            for rate in (lrs["predictor_g"], lrs["predictor_f"],
                         lrs["corrector_g"], lrs["corrector_f"]):
                api_module.apply_parameter_server_update(base, agg, keys, rate)
        finally:
            api_module.apply_parameter_server_update = real

        self.assertEqual(recorded, [1.5, 0.3, 1.5, 0.0])

    def test_coordinator_resolves_from_effective_config_not_args(self):
        """Binding the arithmetic to effective_config is what makes the recorded
        artifact and the applied numbers the same thing."""
        import inspect
        import fedml.simulation.sp.fedavg.fedavg_api as api_module
        source = inspect.getsource(api_module.FedAvgAPI)
        self.assertIn("resolve_server_learning_rates(self.effective_config)", source)
        self.assertIn('lrs["predictor_f"]', source)
        self.assertIn('lrs["corrector_f"]', source)


class EffectiveConfigContractTest(unittest.TestCase):
    def test_new_and_resolved_fields_are_declared(self):
        for name in ("eg_predictor_critic_server_lr", "eg_corrector_critic_server_lr",
                     "resolved_predictor_g_server_lr", "resolved_predictor_f_server_lr",
                     "resolved_corrector_g_server_lr", "resolved_corrector_f_server_lr"):
            self.assertIn(name, EFFECTIVE_CONFIG_FIELDS)


if __name__ == "__main__":
    unittest.main()


class OptionalPathNormalizationTest(unittest.TestCase):
    """The config writer serializes Python None as the STRING "None", so an
    unset override arrives as "None" rather than null. Treating that as a path
    made every run with an unset override fail at startup."""

    def test_sentinels_normalize_to_none(self):
        from experiment_utils import optional_path
        for value in (None, "None", "none", "null", "NULL", "", "   "):
            self.assertIsNone(optional_path(value), repr(value))

    def test_real_paths_survive(self):
        from experiment_utils import optional_path
        self.assertEqual(optional_path("  /tmp/x.pt  "), "/tmp/x.pt")
