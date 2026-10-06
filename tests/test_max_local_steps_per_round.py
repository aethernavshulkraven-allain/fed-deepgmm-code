"""Tests for the ``max_local_steps_per_round`` cap in ``train_gmm``.

Task 2 (eICU Study A Full lock package review): cap local optimizer steps
per communication round so uniform (1/171) aggregation weights genuinely
represent equal local progress, instead of large hospitals (up to 55 local
batches/round) dominating small ones (as few as 2).

Two properties matter:

1. Default behavior (``max_local_steps_per_round`` unset/0) must be
   byte-for-byte unchanged: every batch, every epoch, in original order,
   every round -- the frozen demo campaign never sets this knob and must
   never see a behavior change.
2. When the cap is set, it must not just *limit* the batches used in one
   round -- ``client_data`` for eICU/zoo datasets is a plain, fixed-order
   Python list materialized once at data-load time (see
   fedml/data/data_loader.py), not a reshuffling DataLoader. A naive
   "always take the first N" implementation would train on only the first
   N batches of that fixed list forever, permanently starving the rest of
   a large client's data. The cap must instead *rotate* which batches are
   used each round, so that every batch is eventually used within
   ceil(n_batches / cap) rounds.
"""

import os
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_ROOT = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")
sys.path.insert(0, EXAMPLE_ROOT)

from experiment_utils import check_max_local_steps_participation  # noqa: E402
from fedml.ml.trainer.my_model_trainer_classification import ModelTrainerCLS  # noqa: E402


class RecordingObjective:
    """calc_objective that records which batch (by its encoded index) was
    used, in consumption order, while still returning real, differentiable
    scalars so backward()/step() run exactly as in production."""

    def __init__(self):
        self.seen = []

    def calc_objective(self, g, f, x, z, y):
        self.seen.append(int(x.item()))
        g_value = (g(x) - y).pow(2).mean()
        f_value = (f(z) - y).pow(2).mean()
        return g_value, f_value


def build_trainer():
    trainer = object.__new__(ModelTrainerCLS)
    trainer.g = nn.Linear(1, 1, bias=False).double()
    trainer.f = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        trainer.g.weight.fill_(0.4)
        trainer.f.weight.fill_(-0.3)
    trainer.g_optimizer = torch.optim.SGD(trainer.g.parameters(), lr=1e-3)
    trainer.f_optimizer = torch.optim.SGD(trainer.f.parameters(), lr=1e-3)
    trainer.game_objective = RecordingObjective()
    trainer.id = 0
    return trainer


def make_batches(n):
    """n distinct one-element "batches"; batch i encodes index i in x so
    RecordingObjective can report exactly which original batches were used."""
    batches = []
    for i in range(n):
        x = torch.tensor([[float(i)]], dtype=torch.float64)
        y = torch.tensor([[1.0]], dtype=torch.float64)
        z = torch.tensor([[float(i)]], dtype=torch.float64)
        batches.append((None, None, x, y, z))
    return batches


def base_args(**overrides):
    kwargs = dict(
        epochs=1,
        gradient_clip_norm=1.0,
        dataloader_pin_memory=False,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


class DefaultUnlimitedBehaviorTest(unittest.TestCase):
    def test_unset_cap_processes_every_batch_in_order_every_round(self):
        trainer = build_trainer()
        batches = make_batches(5)
        args = base_args()  # no max_local_steps_per_round attribute at all
        device = torch.device("cpu")

        trainer.train_gmm(batches, device, args)
        self.assertEqual(trainer.game_objective.seen, [0, 1, 2, 3, 4])

        # A second "round" (same trainer instance, same fixed batch list --
        # exactly how fedavg_api.py reuses one Client/trainer for the whole
        # run) must behave identically: no rotation, no truncation.
        trainer.game_objective.seen = []
        trainer.train_gmm(batches, device, args)
        self.assertEqual(trainer.game_objective.seen, [0, 1, 2, 3, 4])

    def test_zero_cap_is_equivalent_to_unset(self):
        trainer = build_trainer()
        batches = make_batches(5)
        args = base_args(max_local_steps_per_round=0)
        device = torch.device("cpu")

        trainer.train_gmm(batches, device, args)
        self.assertEqual(trainer.game_objective.seen, [0, 1, 2, 3, 4])

    def test_two_local_epochs_unset_cap_repeats_full_list_each_epoch(self):
        trainer = build_trainer()
        batches = make_batches(3)
        args = base_args(epochs=2)
        device = torch.device("cpu")

        trainer.train_gmm(batches, device, args)
        self.assertEqual(trainer.game_objective.seen, [0, 1, 2, 0, 1, 2])


class CapLimitsAndRotatesTest(unittest.TestCase):
    def test_cap_limits_steps_taken_in_one_round(self):
        trainer = build_trainer()
        batches = make_batches(5)
        args = base_args(max_local_steps_per_round=2)
        device = torch.device("cpu")

        trainer.train_gmm(batches, device, args)
        self.assertEqual(len(trainer.game_objective.seen), 2)

    def test_cap_rotates_across_rounds_and_covers_every_batch(self):
        """The bug this guards against: always taking the fixed list's
        first `cap` batches every round would freeze `seen` at [0, 1] on
        every call. Batches 2, 3, 4 must eventually be used."""
        trainer = build_trainer()
        batches = make_batches(5)
        args = base_args(max_local_steps_per_round=2)
        device = torch.device("cpu")

        rounds = []
        for _ in range(3):  # ceil(5 / 2) = 3 rounds needed to cover all 5
            trainer.game_objective.seen = []
            trainer.train_gmm(batches, device, args)
            rounds.append(list(trainer.game_objective.seen))

        # Exact deterministic rotation: round 1 -> [0,1], round 2 -> [2,3],
        # round 3 -> [4,0] (wraps around the fixed-order list).
        self.assertEqual(rounds, [[0, 1], [2, 3], [4, 0]])

        covered = set()
        for r in rounds:
            covered.update(r)
        self.assertEqual(covered, {0, 1, 2, 3, 4})

        # Not every round is identical -- the actual regression this test
        # exists to catch.
        self.assertNotEqual(rounds[0], rounds[1])
        self.assertNotEqual(rounds[1], rounds[2])

    def test_cap_equal_to_batch_count_is_a_noop_rotation(self):
        """The smallest eligible full-eICU client has exactly 2 local
        batches at batch_size=64 (Task 3's >=65-row gate), matching this
        campaign's cap of 2 -- every batch is used every round."""
        trainer = build_trainer()
        batches = make_batches(2)
        args = base_args(max_local_steps_per_round=2)
        device = torch.device("cpu")

        for _ in range(3):
            trainer.game_objective.seen = []
            trainer.train_gmm(batches, device, args)
            self.assertEqual(sorted(trainer.game_objective.seen), [0, 1])

    def test_cap_persists_rotation_state_per_trainer_instance(self):
        """Two independent trainers (i.e. two different clients) rotate
        independently -- state lives on `self`, not module-level."""
        batches = make_batches(4)
        args = base_args(max_local_steps_per_round=1)
        device = torch.device("cpu")

        trainer_a = build_trainer()
        trainer_b = build_trainer()

        trainer_a.train_gmm(batches, device, args)
        trainer_a.train_gmm(batches, device, args)
        a_seen = trainer_a.game_objective.seen  # accumulated across 2 calls

        trainer_b.train_gmm(batches, device, args)
        b_seen = trainer_b.game_objective.seen  # accumulated across 1 call

        self.assertEqual(a_seen, [0, 1])
        self.assertEqual(b_seen, [0])


class ParticipationGuardTest(unittest.TestCase):
    """check_max_local_steps_participation: the rotation offset lives on
    each client's persistent, per-client SP trainer instance, so the cap
    is only safe under SP execution with full participation (this
    campaign's actual, locked configuration). Guards against silently
    reintroducing the "same first N batches forever" bug under a future
    config change (partial sampling, or the multiprocess executor)."""

    def test_zero_cap_never_raises_regardless_of_mode(self):
        check_max_local_steps_participation("multi_gpu_processes", True, 0, 5, 171)
        check_max_local_steps_participation("sp", False, 0, 171, 171)

    def test_full_participation_sp_is_allowed(self):
        check_max_local_steps_participation("sp", False, 2, 171, 171)
        # unset client_execution_mode + enable_multiprocessing=False also
        # resolves to sp, matching _create_client_executor's own fallback.
        check_max_local_steps_participation("", False, 2, 171, 171)

    def test_multiprocess_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "client_execution_mode='sp'"):
            check_max_local_steps_participation("multi_gpu_processes", False, 2, 171, 171)

    def test_enable_multiprocessing_flag_raises_even_with_blank_mode(self):
        with self.assertRaisesRegex(ValueError, "client_execution_mode='sp'"):
            check_max_local_steps_participation("", True, 2, 171, 171)

    def test_partial_participation_raises(self):
        with self.assertRaisesRegex(ValueError, "full participation"):
            check_max_local_steps_participation("sp", False, 2, 50, 171)


if __name__ == "__main__":
    unittest.main()
