"""Reference lifetime across FL warm-up, clients, and serialized EG phases."""

import copy
from contextlib import nullcontext
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "fedgmm", "sp_decentralized_mnist_lr_example")
sys.path.insert(0, EXAMPLE)

from game_objectives.simple_moment_objective import PaperAlignedMomentObjective
from model_selection.learning_eval_nostop import (
    FHistoryLearningEvalGradientDecentNoStop,
    FHistoryLearningEvalSGDNoStop,
)
from optimizers.Customsgd import CustomSGD
from optimizers.extragradient import ExtraGradient
from fedml.ml.trainer.my_model_trainer_classification import ModelTrainerCLS
from fedml.simulation.sp.fedavg.client import Client
from fedml.simulation.sp.fedavg.fedavg_api import FedAvgAPI
from fedml.simulation.sp.fedavg.multiprocess_client import _execute_worker_task
from fedml.simulation.sp.fedavg.single_gpu_client import SingleGPUClientExecutor


class TracedObjective(PaperAlignedMomentObjective):
    def __init__(self):
        super().__init__()
        self.snapshots = []
        self.references_seen = []

    def set_theta_tilde(self, g, state_dict=None):
        super().set_theta_tilde(g, state_dict=state_dict)
        self.snapshots.append(copy.deepcopy(self._g_tilde.state_dict()))

    def calc_objective(self, *args):
        self.references_seen.append(copy.deepcopy(self._g_tilde.state_dict()))
        return super().calc_objective(*args)


def assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key


def batch():
    x = torch.tensor([[-2.], [-1.], [1.], [2.]], dtype=torch.float64)
    return (None, None, x, x.abs(), x * 0.4)


def make_trainer(algorithm):
    torch.manual_seed(19)
    trainer = object.__new__(ModelTrainerCLS)
    trainer.g = nn.Linear(1, 1).double()
    trainer.g.register_buffer("reference_marker", torch.tensor(19.))
    trainer.f = nn.Linear(1, 1).double()
    optimizer = ExtraGradient if algorithm == "fed_eg_double" else CustomSGD
    trainer.g_optimizer = optimizer(trainer.g.parameters(), lr=.001)
    trainer.f_optimizer = optimizer(trainer.f.parameters(), lr=.01)
    trainer.game_objective = TracedObjective()
    trainer.args = SimpleNamespace(
        epochs=2, client_optimizer=algorithm, gradient_clip_norm=1.,
        stop_on_numerical_failure=True, zo_mu=.001, zo_num_directions=2,
    )
    trainer.id = 0
    return trainer


@pytest.mark.parametrize("learner", [
    FHistoryLearningEvalSGDNoStop(num_epochs=2, batch_size=2, eval_freq=1),
    FHistoryLearningEvalGradientDecentNoStop(num_iter=2, eval_freq=1),
])
def test_warmup_initializes_one_reference_per_candidate_evaluation(learner):
    trainer = make_trainer("fed_eg")
    _, _, x, y, z = batch()
    objective = trainer.game_objective
    for candidate in range(2):
        before = copy.deepcopy(trainer.g.state_dict())
        objective.references_seen.clear()
        eps_history, f_history = learner.eval(
            x, z, y, x, z, y, trainer.g, trainer.f,
            trainer.g_optimizer, trainer.f_optimizer, objective,
        )
        assert len(objective.snapshots) == candidate + 1
        assert len(eps_history) == len(f_history) == 2
        assert all(torch.isfinite(value).all() for value in eps_history + f_history)
        for reference in objective.references_seen:
            assert_state_equal(reference, before)
        assert not torch.equal(trainer.g.weight, before["weight"])


def test_explicit_reference_copies_batchnorm_buffers_without_mutating_live_g():
    g = nn.Sequential(nn.Linear(1, 2), nn.BatchNorm1d(2), nn.Linear(2, 1)).double()
    reference = copy.deepcopy(g.state_dict())
    reference["1.running_mean"].fill_(3.)
    reference["1.running_var"].fill_(7.)
    reference["1.num_batches_tracked"].fill_(11)
    live = copy.deepcopy(g.state_dict())
    objective = PaperAlignedMomentObjective()
    objective.set_theta_tilde(g, state_dict=reference)
    assert_state_equal(objective._g_tilde.state_dict(), reference)
    assert_state_equal(g.state_dict(), live)
    assert not objective._g_tilde.training
    assert all(not p.requires_grad for p in objective._g_tilde.parameters())
    reference["1.running_mean"].zero_()
    assert torch.equal(objective._g_tilde[1].running_mean, torch.full((2,), 3., dtype=torch.float64))


class CPUWorkerExecutor:
    """Exercise the production task handler without CUDA or worker affinity."""
    def __init__(self, trainer):
        self.prototype = trainer
        self.tasks = []
        self.trainers = []

    def run(self, tasks):
        self.tasks.extend(copy.deepcopy(tasks))
        results = []
        for task in tasks:
            trainer = copy.deepcopy(self.prototype)
            self.trainers.append(trainer)
            with mock.patch(
                "fedml.simulation.sp.fedavg.multiprocess_client._release_worker_cuda_cache"
            ):
                results.append(_execute_worker_task(
                    task, trainer, trainer.args, torch.device("cpu")
                ))
        return results


def make_api(algorithm, serialized):
    trainer = make_trainer(algorithm)
    api = object.__new__(FedAvgAPI)
    api.args = trainer.args
    api.objective_mode = "paper_aligned"
    api.device = torch.device("cpu")
    api.train_data_local_dict = {0: [batch()], 1: [batch()]}
    api.test_data_local_dict = {0: None, 1: None}
    api.train_data_local_num_dict = {0: 4, 1: 4}
    api.client_list = [Client(i, [batch()], None, 4, trainer.args,
                              api.device, copy.deepcopy(trainer)) for i in range(2)]
    api.client_executor = CPUWorkerExecutor(trainer) if serialized else None
    return api, trainer


@pytest.mark.parametrize("algorithm", ["fed_eg", "fed_eg_double", "fed_zo_eg"])
@pytest.mark.parametrize("serialized", [False, True])
def test_round_reference_survives_both_phases_and_next_round(algorithm, serialized):
    api, trainer = make_api(algorithm, serialized)
    g_base = copy.deepcopy(trainer.g.state_dict())
    f_base = copy.deepcopy(trainer.f.state_dict())
    api._run_primary_client_updates([0, 1], g_base, f_base, None)
    g_lookahead = {k: value + .1 for k, value in g_base.items()}
    f_lookahead = {k: value + .2 for k, value in f_base.items()}
    corrected = api._run_correction_client_updates(
        [0, 1], g_lookahead, f_lookahead, g_reference=g_base
    )
    trainers = (api.client_executor.trainers if serialized else
                [client.model_trainer for client in api.client_list])
    for local in trainers:
        for reference in local.game_objective.references_seen:
            assert_state_equal(reference, g_base)
    if serialized:
        for task in api.client_executor.tasks[2:]:
            assert_state_equal(task["g_reference"], g_base)
            assert all(v.device.type == "cpu" for v in task["g_reference"].values())
    next_g, next_f = corrected[0][1]
    api._run_primary_client_updates([0, 1], next_g, next_f, None)
    next_trainers = (api.client_executor.trainers[-2:] if serialized else trainers)
    for local in next_trainers:
        assert_state_equal(local.game_objective.snapshots[-1], next_g)
    assert_state_equal(g_base, trainer.g.state_dict())


def test_paper_correction_rejects_missing_round_reference():
    api, trainer = make_api("fed_eg", False)
    with pytest.raises(ValueError, match="round-base g reference"):
        api._run_correction_client_updates(
            [0, 1], trainer.g.state_dict(), trainer.f.state_dict()
        )


@pytest.mark.parametrize("algorithm", ["fed_eg", "fed_eg_double", "fed_zo_eg"])
def test_stream_slot_forwards_reference_without_changing_completion_barrier(algorithm):
    # Test the stream slot's existing input contract. The independent
    # coordinator/stream train_data vs gmm_train_epochs mismatch is not fixed here.
    trainer = make_trainer(algorithm)
    reference = copy.deepcopy(trainer.g.state_dict())
    executor = object.__new__(SingleGPUClientExecutor)
    executor.device = torch.device("cpu")
    executor.args = trainer.args
    executor.trainers = [trainer]
    executor.streams = [mock.Mock()]
    task = {
        "client_idx": 0, "train_data": [batch()], "sample_number": 4,
        "phase": "correction", "use_zeroth_order": algorithm == "fed_zo_eg",
        "g_global": {k: value + .1 for k, value in reference.items()},
        "f_global": copy.deepcopy(trainer.f.state_dict()), "g_reference": reference,
    }
    with mock.patch.object(torch.cuda, "device", return_value=nullcontext()), \
            mock.patch.object(torch.cuda, "stream", return_value=nullcontext()):
        executor._run_task(0, task)
    assert_state_equal(trainer.game_objective._g_tilde.state_dict(), reference)
    executor.streams[0].synchronize.assert_called_once_with()
