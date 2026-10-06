import os
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_ROOT = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")
sys.path.insert(0, EXAMPLE_ROOT)

from experiment_utils import get_effective_config  # noqa: E402
from optimizers.extragradient import ExtraGradient  # noqa: E402
from fedml.simulation.sp.fedavg.client import Client  # noqa: E402
from fedml.ml.trainer.my_model_trainer_classification import ModelTrainerCLS  # noqa: E402
from fedml.simulation.sp.fedavg.fedavg_api import FedAvgAPI  # noqa: E402


class CountingObjective:
    def __init__(self):
        self.calls = 0

    def calc_objective(self, g, f, x, z, y):
        self.calls += 1
        g_value = g(x).pow(2).mean() + 0.25 * f(z).pow(2).mean()
        f_value = f(z).pow(2).mean() + 0.5 * g(x).pow(2).mean()
        return g_value, f_value


def build_zo_trainer(learning_rate):
    trainer = object.__new__(ModelTrainerCLS)
    trainer.g = nn.Linear(1, 1, bias=False).double()
    trainer.f = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        trainer.g.weight.fill_(0.4)
        trainer.f.weight.fill_(-0.7)
    trainer.g_optimizer = torch.optim.SGD(trainer.g.parameters(), lr=learning_rate)
    trainer.f_optimizer = torch.optim.SGD(trainer.f.parameters(), lr=learning_rate)
    trainer.game_objective = CountingObjective()
    trainer.args = SimpleNamespace()
    trainer.id = 0
    return trainer


def zo_args(mu=1e-3, directions=2):
    return SimpleNamespace(
        epochs=1,
        zo_mu=mu,
        zo_num_directions=directions,
        gradient_clip_norm=1.0,
        dataloader_pin_memory=False,
    )


def one_batch():
    values = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    return [(None, None, values, values.clone(), values.clone())]


class EffectiveConfigDispatchTest(unittest.TestCase):
    def test_all_supported_optimizers_have_explicit_variants(self):
        expected = {
            "sgd": "fedgda_d",
            "ogda": "fedogda_d",
            "fed_eg": "fed_eg_d",
            "fed_eg_double": "fed_eg_double_d",
            "fed_zo_eg": "fed_zo_eg_d",
        }
        for optimizer, variant in expected.items():
            with self.subTest(optimizer=optimizer):
                config = get_effective_config(
                    SimpleNamespace(dataset="abs", client_optimizer=optimizer, batch_size=0)
                )
                self.assertEqual(config["variant"], variant)

    def test_unknown_optimizer_fails_instead_of_falling_back_to_sgd(self):
        with self.assertRaisesRegex(ValueError, "client_optimizer must be one of"):
            get_effective_config(
                SimpleNamespace(dataset="abs", client_optimizer="adam", batch_size=0)
            )

    def test_zeroth_order_parameters_are_validated(self):
        with self.assertRaisesRegex(ValueError, "zo_mu must be positive"):
            get_effective_config(
                SimpleNamespace(
                    dataset="abs",
                    client_optimizer="fed_zo_eg",
                    batch_size=0,
                    zo_mu=0.0,
                )
            )
        with self.assertRaisesRegex(ValueError, "zo_num_directions"):
            get_effective_config(
                SimpleNamespace(
                    dataset="abs",
                    client_optimizer="fed_zo_eg",
                    batch_size=0,
                    zo_num_directions=0,
                )
            )


class ZerothOrderTrainerTest(unittest.TestCase):
    def test_each_direction_uses_two_forwards_and_restores_perturbations(self):
        trainer = build_zo_trainer(learning_rate=0.0)
        g_before = trainer.g.weight.detach().clone()
        f_before = trainer.f.weight.detach().clone()

        torch.manual_seed(11)
        trainer.train_gmm_zo(one_batch(), torch.device("cpu"), zo_args(directions=3))

        self.assertEqual(trainer.game_objective.calls, 6)
        torch.testing.assert_close(trainer.g.weight, g_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(trainer.f.weight, f_before, rtol=0.0, atol=0.0)

    def test_seed_reproduces_forward_only_update(self):
        first = build_zo_trainer(learning_rate=0.1)
        second = build_zo_trainer(learning_rate=0.1)

        torch.manual_seed(123)
        first.train_gmm_zo(one_batch(), torch.device("cpu"), zo_args())
        torch.manual_seed(123)
        second.train_gmm_zo(one_batch(), torch.device("cpu"), zo_args())

        torch.testing.assert_close(first.g.weight, second.g.weight, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first.f.weight, second.f.weight, rtol=0.0, atol=0.0)

    def test_rademacher_entries_are_signed_and_independent(self):
        parameters = [torch.zeros(64, dtype=torch.float64)]
        torch.manual_seed(7)
        first = ModelTrainerCLS._rademacher_directions(parameters)[0]
        second = ModelTrainerCLS._rademacher_directions(parameters)[0]
        self.assertTrue(set(first.tolist()) <= {-1.0, 1.0})
        self.assertTrue(set(second.tolist()) <= {-1.0, 1.0})
        self.assertFalse(torch.equal(first, second))


class LocalExtraGradientTest(unittest.TestCase):
    def test_correction_gradient_is_applied_from_original_parameters(self):
        trainer = build_zo_trainer(learning_rate=0.1)
        trainer.g_optimizer = ExtraGradient(trainer.g.parameters(), lr=0.1)
        trainer.f_optimizer = ExtraGradient(trainer.f.parameters(), lr=0.1)
        args = SimpleNamespace(
            epochs=1,
            gradient_clip_norm=100.0,
            dataloader_pin_memory=False,
        )

        trainer.train_gmm_eg(one_batch(), torch.device("cpu"), args)

        self.assertEqual(trainer.game_objective.calls, 2)
        torch.testing.assert_close(
            trainer.g.weight, torch.tensor([[0.3]], dtype=torch.float64)
        )
        torch.testing.assert_close(
            trainer.f.weight, torch.tensor([[-0.525]], dtype=torch.float64)
        )

    def test_client_dispatch_keeps_fedeg_and_double_distinct(self):
        class Trainer:
            def __init__(self):
                self.calls = []

            def set_g_model_params(self, params):
                pass

            def set_f_model_params(self, params):
                pass

            def train_gmm(self, data, device, args):
                self.calls.append("gda")

            def train_gmm_eg(self, data, device, args):
                self.calls.append("eg")

            def get_g_model_params(self):
                return {}

            def get_f_model_params(self):
                return {}

        for optimizer, expected in (("fed_eg", "gda"), ("fed_eg_double", "eg")):
            with self.subTest(optimizer=optimizer):
                trainer = Trainer()
                client = Client(
                    0, [], None, 1,
                    SimpleNamespace(client_optimizer=optimizer),
                    torch.device("cpu"), trainer,
                )
                client.train({}, {})
                self.assertEqual(trainer.calls, [expected])


class FakeClient:
    def __init__(self, client_idx):
        self.client_idx = client_idx
        self.calls = []

    def train(self, g_global, f_global):
        self.calls.append(("exact", g_global, f_global))
        return [g_global, f_global]

    def train_zo(self, g_global, f_global):
        self.calls.append(("zo", g_global, f_global))
        return [g_global, f_global]

    def get_sample_number(self):
        return self.client_idx + 10


class CorrectionBarrierTest(unittest.TestCase):
    def test_exact_and_zo_corrections_reuse_sampled_client_order(self):
        for optimizer, expected_call in (
            ("fed_eg", "exact"),
            ("fed_eg_double", "exact"),
            ("fed_zo_eg", "zo"),
        ):
            with self.subTest(optimizer=optimizer):
                api = object.__new__(FedAvgAPI)
                api.args = SimpleNamespace(client_optimizer=optimizer)
                api.client_list = [FakeClient(4), FakeClient(9)]
                g = {"weight": torch.tensor([1.0])}
                f = {"weight": torch.tensor([2.0])}

                results = api._run_correction_client_updates([4, 9], g, f)

                self.assertEqual([item[0] for item in results], [14, 19])
                self.assertEqual(
                    [client.calls[0][0] for client in api.client_list],
                    [expected_call, expected_call],
                )

    def test_client_order_mismatch_fails_at_barrier(self):
        api = object.__new__(FedAvgAPI)
        api.args = SimpleNamespace(client_optimizer="fed_eg")
        api.client_list = [FakeClient(3)]
        with self.assertRaisesRegex(RuntimeError, "client order"):
            api._run_correction_client_updates(
                [8], {"weight": torch.tensor([1.0])}, {"weight": torch.tensor([2.0])}
            )


if __name__ == "__main__":
    unittest.main()
