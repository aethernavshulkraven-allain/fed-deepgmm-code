"""Fixed-statistics BatchNorm as a local-training diagnostic arm.

The median client holds 11 samples, so BatchNorm estimates its statistics from
very few points. local_bn_mode="frozen" puts only the BatchNorm modules into
eval during local updates -- stored running statistics, no updates to them --
while every weight including the BN affine parameters stays trainable.

This is a DIAGNOSTIC, not a proposed fix: it changes the forward map and its
Jacobian, so an improvement under it is evidence about a normalization-
mediated mechanism, not proof of one. Note also that LeakySoftmaxCNN already
bypasses BatchNorm for single-sample batches, so the smallest clients are on
frozen statistics regardless.

Two properties carry the change: "batch" must be bit-identical to the current
code, and "frozen" must actually alter training rather than silently doing
nothing.
"""

import copy
import os
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example"))

from fedml.ml.trainer.my_model_trainer_classification import ModelTrainerCLS  # noqa: E402
from game_objectives.simple_moment_objective import OptimalMomentObjective    # noqa: E402
from models.cnn_models import DefaultCNN                                      # noqa: E402
from models.mlp_model import MLPModel                                         # noqa: E402
from optimizers.Customsgd import CustomSGD                                    # noqa: E402


def make_batch(n=6, seed=17):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand((n, 1, 28, 28), generator=gen, dtype=torch.float64)
    z = torch.randn((n, 1), generator=gen, dtype=torch.float64)
    y = torch.randn((n, 1), generator=gen, dtype=torch.float64)
    return (None, None, x, y, z)


def make_trainer(bn_mode, seed=11):
    torch.manual_seed(seed)
    t = object.__new__(ModelTrainerCLS)
    t.g = DefaultCNN(cuda=False).double()
    t.f = MLPModel(input_dim=1, layer_widths=[20], activation=nn.LeakyReLU).double()
    t.g_optimizer = CustomSGD(t.g.parameters(), lr=0.003, momentum=0.0)
    t.f_optimizer = CustomSGD(t.f.parameters(), lr=0.03, momentum=0.0)
    t.game_objective = OptimalMomentObjective(lambda_1=0.1)
    t.args = SimpleNamespace(
        epochs=1, client_optimizer="fed_eg", gradient_clip_norm=1.0,
        stop_on_numerical_failure=True, zo_mu=1e-3, zo_num_directions=1,
        max_local_steps_per_round=0, dataloader_pin_memory=False,
        local_bn_mode=bn_mode)
    t.id = 0
    return t


class DefaultIsBitIdenticalTest(unittest.TestCase):
    def test_batch_mode_matches_an_unset_flag(self):
        results = []
        for mode in ("batch", None):
            t = make_trainer("batch")
            if mode is None:
                delattr(t.args, "local_bn_mode")
            t.train_gmm([make_batch()], torch.device("cpu"), t.args)
            results.append(copy.deepcopy(t.g.state_dict()))
        for key in results[0]:
            self.assertTrue(torch.equal(results[0][key], results[1][key]),
                            f"{key} differs between explicit 'batch' and an unset flag")

    def test_batch_mode_leaves_bn_in_training(self):
        t = make_trainer("batch")
        switched = t.apply_local_bn_mode((t.g, t.f), "batch")
        self.assertEqual(switched, 0)


class FrozenModeActuallyIntervenesTest(unittest.TestCase):
    def test_frozen_switches_the_batchnorm_modules(self):
        t = make_trainer("frozen")
        t.g.train()
        switched = t.apply_local_bn_mode((t.g, t.f), "frozen")
        self.assertGreater(switched, 0, "no BatchNorm module was switched")
        bns = [m for m in t.g.modules()
               if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        self.assertTrue(bns)
        for module in bns:
            self.assertFalse(module.training, "BatchNorm still in training mode")

    def test_affine_parameters_stay_trainable(self):
        t = make_trainer("frozen")
        t.g.train()
        t.apply_local_bn_mode((t.g, t.f), "frozen")
        for name, param in t.g.named_parameters():
            self.assertTrue(param.requires_grad, f"{name} was frozen; only stats should be")

    def test_frozen_changes_training(self):
        """A diagnostic that silently does nothing is worse than none."""
        out = {}
        for mode in ("batch", "frozen"):
            t = make_trainer(mode)
            t.train_gmm([make_batch()], torch.device("cpu"), t.args)
            out[mode] = copy.deepcopy(t.g.state_dict())
        diffs = [k for k in out["batch"]
                 if not torch.equal(out["batch"][k], out["frozen"][k])]
        self.assertTrue(diffs, "frozen mode produced an identical update; it is a no-op")

    def test_frozen_does_not_move_running_statistics(self):
        t = make_trainer("frozen")
        before = {k: v.clone() for k, v in t.g.state_dict().items()
                  if "running_" in k or "num_batches_tracked" in k}
        self.assertTrue(before, "fixture has no BatchNorm buffers to check")
        t.train_gmm([make_batch()], torch.device("cpu"), t.args)
        after = t.g.state_dict()
        for key, value in before.items():
            self.assertTrue(torch.equal(value, after[key]),
                            f"{key} moved under frozen statistics")

    def test_invalid_mode_is_rejected(self):
        t = make_trainer("batch")
        with self.assertRaises(ValueError):
            t.apply_local_bn_mode((t.g,), "off")


class CalibratedStatisticsTest(unittest.TestCase):
    """The frozen arm pinned running_var=1, the untrained default -- it tested
    "normalize by an uninformed prior forever", not "use calibrated statistics
    instead of noisy ones". Calibration measures the ACTUAL pre-BatchNorm
    statistics at the current weights from a fixed panel of training images."""

    def panel(self, n=64):
        gen = torch.Generator().manual_seed(3)
        return torch.rand((n, 1, 28, 28), generator=gen, dtype=torch.float64)

    def observed_bn_input_stats(self, g, panel):
        caught = {}
        h = g.bn.register_forward_pre_hook(
            lambda m, i: caught.__setitem__("t", i[0].detach()))
        was = g.training
        g.eval()
        with torch.no_grad():
            g(panel)
        g.train(was)
        h.remove()
        flat = caught["t"].transpose(0, 1).reshape(caught["t"].shape[1], -1)
        return flat.mean(dim=1), flat.var(dim=1, unbiased=False)

    def test_buffers_match_the_measured_statistics(self):
        t = make_trainer("calibrated")
        panel = self.panel()
        want_mean, want_var = self.observed_bn_input_stats(t.g, panel)
        n = t.calibrate_bn_statistics(t.g, panel)
        self.assertGreater(n, 0, "nothing was calibrated")
        self.assertTrue(torch.allclose(t.g.bn.running_mean, want_mean, atol=1e-12))
        self.assertTrue(torch.allclose(t.g.bn.running_var, want_var, atol=1e-12))

    def test_calibration_replaces_the_uninformed_prior(self):
        """The default buffers are mean 0 / var 1 and carry no feature
        information. Calibration must move them somewhere real."""
        t = make_trainer("calibrated")
        self.assertTrue(torch.allclose(t.g.bn.running_var,
                                       torch.ones_like(t.g.bn.running_var)))
        t.calibrate_bn_statistics(t.g, self.panel())
        self.assertFalse(torch.allclose(t.g.bn.running_var,
                                        torch.ones_like(t.g.bn.running_var)),
                         "running_var is still the untrained default")
        self.assertTrue(torch.isfinite(t.g.bn.running_var).all())
        self.assertTrue((t.g.bn.running_var >= 0).all())

    def test_calibration_does_not_touch_parameters_or_training_flags(self):
        t = make_trainer("calibrated")
        before = {k: v.clone() for k, v in t.g.state_dict().items()
                  if "running_" not in k and "num_batches" not in k}
        flags = [(m, m.training) for m in t.g.modules()]
        t.g.train()
        t.calibrate_bn_statistics(t.g, self.panel())
        after = t.g.state_dict()
        for k, v in before.items():
            self.assertTrue(torch.equal(v, after[k]), f"{k} moved during calibration")
        for module, _ in flags:
            self.assertTrue(module.training, "training flag not restored")

    def test_calibration_does_not_consume_global_rng(self):
        t = make_trainer("calibrated")
        state = torch.get_rng_state()
        t.calibrate_bn_statistics(t.g, self.panel())
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_absent_panel_is_a_no_op(self):
        t = make_trainer("calibrated")
        self.assertEqual(t.calibrate_bn_statistics(t.g, None), 0)

    def test_calibrated_changes_training_and_holds_statistics_fixed(self):
        out = {}
        for mode in ("batch", "calibrated"):
            t = make_trainer(mode)
            t.args._fedgmm_calibration_panel = self.panel()
            before_var = t.g.state_dict()["bn.running_var"].clone()
            t.train_gmm([make_batch()], torch.device("cpu"), t.args)
            out[mode] = (copy.deepcopy(t.g.state_dict()), before_var)
        diffs = [k for k in out["batch"][0]
                 if not torch.equal(out["batch"][0][k], out["calibrated"][0][k])]
        self.assertTrue(diffs, "calibrated mode was a no-op")
        # statistics are calibrated once at phase start, then held
        self.assertFalse(torch.allclose(out["calibrated"][0]["bn.running_var"],
                                        out["calibrated"][1]),
                         "buffers were never calibrated")

    def test_unknown_mode_still_rejected(self):
        t = make_trainer("batch")
        with self.assertRaises(ValueError):
            t.apply_local_bn_mode((t.g,), "sometimes")


if __name__ == "__main__":
    unittest.main()
