"""The pooled harness must refuse to run if it computes the cancelled gradient.

The federated trainer's gradient tests certify the federated trainer. They say
nothing about a freshly written pooled loop -- which is exactly where the defect
happened: scripts/run_centralized_control_femnist_x_20260921.py placed two
unrestricted backward() calls before both optimizer steps, so each network
received the gradient of L_g + L_f = M + (-M + P) = P, the moment terms
cancelled, and two days of conclusions were drawn from runs that trained on the
penalty alone.

So the pooled harness carries its own gate, and the gate exercises the SAME
function the training loop uses. These tests check the gate passes on the real
implementation and, more importantly, that it FAILS when the defect is
reintroduced -- a gate that cannot fail certifies nothing.
"""

import importlib.util
import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example"))


def load_harness():
    path = os.path.join(REPO_ROOT, "scripts", "run_pooled_control_femnist_x_20260922.py")
    spec = importlib.util.spec_from_file_location("pooled_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PooledGradientGateTest(unittest.TestCase):
    def setUp(self):
        self.pooled = load_harness()
        self.device = torch.device("cpu")

    def test_gate_passes_on_the_real_implementation(self):
        gate = self.pooled.assert_gradient_gate(self.device)
        self.assertGreater(gate["g_gap_vs_cancelled"], 1e-6)
        self.assertGreater(gate["f_gap_vs_cancelled"], 1e-6)

    def test_gate_fires_when_the_k1_defect_is_reintroduced(self):
        original = self.pooled.game_gradients

        def defective(objective, g, f, xb, zb, yb):
            g_params = [p for p in g.parameters() if p.requires_grad]
            f_params = [p for p in f.parameters() if p.requires_grad]
            for param in g_params + f_params:
                param.grad = None
            g_obj, f_obj = objective.calc_objective(g, f, xb, zb, yb)
            g_obj.backward(retain_graph=True)   # the defect: both losses land
            f_obj.backward()                    # in both parameter sets
            return ([p.grad.clone() for p in g_params],
                    [p.grad.clone() for p in f_params], g_obj, f_obj)

        self.pooled.game_gradients = defective
        try:
            with self.assertRaises(RuntimeError) as ctx:
                self.pooled.assert_gradient_gate(self.device)
            # The equality check against the independent reference now catches
            # this before the K1-specific comparison is reached, so a pure K1
            # defect surfaces as a reference mismatch. The K1 check is retained
            # as defence in depth and names the historical defect explicitly if
            # some future path reaches it.
            message = str(ctx.exception)
            self.assertTrue(
                "do not match the independent" in message or "penalty-only" in message,
                f"K1 rejected for an unexpected reason: {message}")
        finally:
            self.pooled.game_gradients = original

    def _assert_gate_rejects(self, transform, label):
        original = self.pooled.game_gradients

        def mutated(objective, g, f, xb, zb, yb):
            gg, fg, g_obj, f_obj = original(objective, g, f, xb, zb, yb)
            return transform(gg), transform(fg), g_obj, f_obj

        self.pooled.game_gradients = mutated
        try:
            with self.assertRaises(RuntimeError, msg=f"{label} was not rejected"):
                self.pooled.assert_gradient_gate(self.device)
        finally:
            self.pooled.game_gradients = original

    def test_gate_rejects_zero_gradients(self):
        """Differing from penalty-only is necessary but NOT sufficient: zeros
        differ from the cancelled gradient and passed an earlier gate."""
        self._assert_gate_rejects(
            lambda grads: [torch.zeros_like(t) for t in grads], "zero gradients")

    def test_gate_rejects_sign_reversed_gradients(self):
        """Sign reversal also differs from penalty-only, and would turn descent
        into ascent for both players."""
        self._assert_gate_rejects(lambda grads: [-t for t in grads], "sign reversal")

    def test_gate_rejects_rescaled_gradients(self):
        self._assert_gate_rejects(lambda grads: [t * 2.0 for t in grads], "2x scaling")

    def test_gate_rejects_a_dropped_participant(self):
        original = self.pooled.game_gradients

        def mutated(objective, g, f, xb, zb, yb):
            gg, fg, g_obj, f_obj = original(objective, g, f, xb, zb, yb)
            gg[0] = None
            return gg, fg, g_obj, f_obj

        self.pooled.game_gradients = mutated
        try:
            with self.assertRaises(RuntimeError) as ctx:
                self.pooled.assert_gradient_gate(self.device)
            self.assertIn("participates", str(ctx.exception))
        finally:
            self.pooled.game_gradients = original

    def test_gate_matches_an_independent_reference_exactly(self):
        """The primary check is equality to gradients computed by a different
        mechanism -- backward() with interleaved zero_grad -- not merely
        inequality with a known-bad value."""
        gate = self.pooled.assert_gradient_gate(self.device)
        self.assertLess(gate["g_max_abs_diff_vs_reference"], 1e-10)
        self.assertLess(gate["f_max_abs_diff_vs_reference"], 1e-10)
        self.assertGreater(gate["g_grad_scale"], 1e-12)
        self.assertGreater(gate["f_grad_scale"], 1e-12)

    def test_game_gradients_leaves_the_other_player_untouched(self):
        """Isolation is structural here, not a property of call ordering."""
        from game_objectives.simple_moment_objective import OptimalMomentObjective
        from models.cnn_models import DefaultCNN
        from models.mlp_model import MLPModel
        import torch.nn as nn

        torch.manual_seed(5)
        g = DefaultCNN(cuda=False).double()
        f = MLPModel(input_dim=1, layer_widths=[20], activation=nn.LeakyReLU).double()
        for param in list(g.parameters()) + list(f.parameters()):
            param.grad = torch.zeros_like(param)
        gen = torch.Generator().manual_seed(2)
        xb = torch.rand((4, 1, 28, 28), generator=gen, dtype=torch.float64)
        zb = torch.randn((4, 1), generator=gen, dtype=torch.float64)
        yb = torch.randn((4, 1), generator=gen, dtype=torch.float64)

        self.pooled.game_gradients(
            OptimalMomentObjective(lambda_1=0.1), g, f, xb, zb, yb)
        for param in list(g.parameters()) + list(f.parameters()):
            self.assertTrue(torch.all(param.grad == 0),
                            "game_gradients wrote into .grad; it must only return")


if __name__ == "__main__":
    unittest.main()
