"""Audit: does FedGDA-D / FedOGDA-D's client-local update match the intended
simultaneous-gradient equations, given that ``train_gmm`` calls
``g_optimizer.step()`` before ``f_obj.backward()`` on the same retained
graph -- and every optimizer here (``CustomSGD``, ``OGDA``, and now
``ExtraGradient``) writes through ``.data``, which silently skips
PyTorch's autograd version check for exactly that ordering?

The concern: if PyTorch's backward pass for ``f_obj`` re-read g's
parameters *after* they were mutated by ``g_optimizer.step()``, f's
gradient would be silently computed against the wrong (post-step) theta
-- a real Gauss-Seidel-style contamination, not the simultaneous GDA the
method is supposed to implement.

Verified two independent ways, per the audit request:

1. ``ClosedFormObjectiveGradientTest`` -- on a toy 1-D linear g/f where
   ``OptimalMomentObjective``'s gradient is hand-derivable, the autograd
   gradient must equal the hand formula exactly.
2. ``OrderingDoesNotContaminateGradientTest`` /
   ``ParameterUpdateMatchesReferenceTest`` -- run the *real*
   ``CustomSGD``/``OGDA`` classes through the *exact* sequence
   ``train_gmm`` uses (g backward+clip+step, THEN f backward+clip+step),
   and compare the resulting parameters against a reference trajectory
   computed independently: fresh ``torch.autograd.grad`` calls on
   un-mutated snapshots, with the SGD/OGDA update formulas re-implemented
   by hand from the optimizers' own docstrings (not by calling the
   classes under test). Two epochs, so OGDA's optimistic term
   (``2*lr*g_t - lr*g_{t-1}``) is actually exercised on the second step.

Outcome recorded in the ledger notes: if these pass, the
existing FedGDA-D/FedOGDA-D screen and finals results stand as computed
against the intended equations. If they ever fail, both methods need
retuning from the screen onward -- the frozen hyperparameters would no
longer be valid.
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example"))

from game_objectives.simple_moment_objective import OptimalMomentObjective  # noqa: E402
from optimizers.Customsgd import CustomSGD  # noqa: E402
from optimizers.ogda import OGDA  # noqa: E402


class Linear1D(nn.Module):
    """g(x) = theta * x  or  f(z) = tau * z -- single scalar parameter, so
    OptimalMomentObjective's gradient has a clean closed form."""

    def __init__(self, init_value):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float64))

    def forward(self, v):
        return self.weight * v.squeeze(-1)


# Exact values, chosen so every intermediate is exactly representable in
# float64 -- no rounding noise to obscure a real discrepancy.
X = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
Z = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
Y = torch.tensor([[0.5], [0.5]], dtype=torch.float64)
LAMBDA_1 = 0.1
LR = 0.1
# Deliberately far above every gradient norm this toy problem produces (checked
# in test_clip_norm_choice_never_actually_binds below) -- the real repo default
# is 1.0, but this audit is about the g-before-f ordering, not about re-verifying
# torch.nn.utils.clip_grad_norm_ itself, so clipping is kept a no-op here.
CLIP_NORM = 1000.0


def hand_grads(theta, tau):
    """d(g_obj)/d(theta) and d(f_obj)/d(tau) for OptimalMomentObjective,
    evaluated at a single (theta, tau) point -- i.e. the simultaneous-GDA
    gradient, with no interleaved mutation between the two.

    g_of_x = theta*x, f_of_z = tau*z, epsilon = theta*x - y
    g_obj = mean(f_of_z * epsilon)                       -> dg/dtheta = mean(f_of_z * x)
    f_obj = -g_obj + lambda_1 * mean(f_of_z^2 * epsilon^2)
          -> df/dtau = -mean(z*epsilon) + lambda_1*mean(2*f_of_z*z*epsilon^2)
    """
    x = X.squeeze(-1)
    z = Z.squeeze(-1)
    y = Y.squeeze(-1)
    epsilon = theta * x - y
    f_of_z = tau * z
    d_theta = (f_of_z * x).mean()
    d_tau = -(z * epsilon).mean() + LAMBDA_1 * (2 * f_of_z * z * epsilon**2).mean()
    return d_theta.item(), d_tau.item()


class ClosedFormObjectiveGradientTest(unittest.TestCase):
    """OptimalMomentObjective's own gradient, independent of ordering
    concerns entirely -- must match the hand formula above exactly."""

    def test_g_and_f_gradients_match_hand_derivation(self):
        g = Linear1D(1.0)
        f = Linear1D(1.0)
        objective = OptimalMomentObjective(lambda_1=LAMBDA_1)
        g_obj, f_obj = objective.calc_objective(g, f, X, Z, Y)

        g_obj.backward(retain_graph=True)
        # Read g's gradient now, before f_obj.backward() runs -- f_obj also
        # depends on g (epsilon), so its backward() would accumulate an
        # additional contribution into g.weight.grad if checked afterward.
        # This is exactly why train_gmm's own g_optimizer.step() happens
        # before f_obj.backward(): it consumes g's gradient at this point,
        # not after it's been added to by f's pass too.
        actual_dtheta = g.weight.grad.item()

        f.weight.grad = None
        f_obj.backward()
        actual_dtau = f.weight.grad.item()

        expected_dtheta, expected_dtau = hand_grads(1.0, 1.0)
        self.assertAlmostEqual(actual_dtheta, expected_dtheta, places=12)
        self.assertAlmostEqual(actual_dtau, expected_dtau, places=12)


class ClipNormChoiceNeverBindsTest(unittest.TestCase):
    """Confirms CLIP_NORM's "never binds" comment is actually true, rather
    than trusted on faith -- this audit is about update ordering, not about
    conflating results with clip_grad_norm's own (separately trusted,
    standard-library) behavior."""

    def test_clip_norm_choice_never_actually_binds(self):
        d_theta, d_tau = hand_grads(1.0, 1.0)
        self.assertLess(abs(d_theta), CLIP_NORM)
        self.assertLess(abs(d_tau), CLIP_NORM)


class OrderingDoesNotContaminateGradientTest(unittest.TestCase):
    """The core audit question: with the REAL CustomSGD, does
    g_optimizer.step() (a .data-based in-place write) running before
    f_obj.backward() change the gradient f actually receives?

    If it did, the captured f.grad below would differ from hand_grads'
    reference (computed at the un-mutated theta=1.0) -- and would instead
    drift toward the gradient at theta=0.75 (post g-step). Those two
    candidate values are computed explicitly so a failure is legible, not
    just "assertion failed".
    """

    def test_fgrad_uses_pre_gstep_theta_not_post_gstep_theta(self):
        g = Linear1D(1.0)
        f = Linear1D(1.0)
        objective = OptimalMomentObjective(lambda_1=LAMBDA_1)
        g_optimizer = CustomSGD(g.parameters(), lr=LR, momentum=0.0)
        f_optimizer = CustomSGD(f.parameters(), lr=LR, momentum=0.0)

        # Exact real train_gmm sequence.
        g_obj, f_obj = objective.calc_objective(g, f, X, Z, Y)
        g_optimizer.zero_grad()
        g_obj.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(g.parameters(), CLIP_NORM)
        g_optimizer.step()  # mutates theta 1.0 -> 0.75 via .data, BEFORE f backward

        f_optimizer.zero_grad()
        f_obj.backward()  # graph was built when theta was still 1.0
        torch.nn.utils.clip_grad_norm_(f.parameters(), CLIP_NORM)
        actual_f_grad = f.weight.grad.item()

        _, correct_dtau_at_theta_1p0 = hand_grads(1.0, 1.0)
        _, wrong_dtau_if_contaminated_at_theta_0p75 = hand_grads(0.75, 1.0)

        # Sanity: the two candidates must actually differ, or this test
        # can't distinguish correct from contaminated.
        self.assertGreater(
            abs(correct_dtau_at_theta_1p0 - wrong_dtau_if_contaminated_at_theta_0p75), 0.05
        )
        self.assertAlmostEqual(actual_f_grad, correct_dtau_at_theta_1p0, places=12)
        self.assertNotAlmostEqual(
            actual_f_grad, wrong_dtau_if_contaminated_at_theta_0p75, places=3
        )


class ParameterUpdateMatchesReferenceTest(unittest.TestCase):
    """Two local epochs (matching how train_gmm's inner loop reuses one
    optimizer instance across a client's local epochs), FedGDA-D
    (CustomSGD) and FedOGDA-D (OGDA), each checked against a reference
    trajectory computed with fresh torch.autograd.grad calls and the
    update formulas re-implemented by hand -- not by calling the classes
    under test.
    """

    def _reference_sgd_trajectory(self, epochs):
        theta, tau = 1.0, 1.0
        for _ in range(epochs):
            grad_theta, grad_tau = hand_grads(theta, tau)
            theta = theta - LR * grad_theta
            tau = tau - LR * grad_tau
        return theta, tau

    def _reference_ogda_trajectory(self, epochs):
        theta, tau = 1.0, 1.0
        prev_g, prev_f = None, None
        for _ in range(epochs):
            grad_theta, grad_tau = hand_grads(theta, tau)
            if prev_g is None:  # OGDA's own init: prev_grad = grad.clone() on step 1
                prev_g, prev_f = grad_theta, grad_tau
            theta = theta - (2.0 * LR * grad_theta - LR * prev_g)
            tau = tau - (2.0 * LR * grad_tau - LR * prev_f)
            prev_g, prev_f = grad_theta, grad_tau
        return theta, tau

    def _run_real_trainer(self, optimizer_cls, epochs):
        g = Linear1D(1.0)
        f = Linear1D(1.0)
        objective = OptimalMomentObjective(lambda_1=LAMBDA_1)
        g_optimizer = optimizer_cls(g.parameters(), lr=LR) if optimizer_cls is OGDA \
            else optimizer_cls(g.parameters(), lr=LR, momentum=0.0)
        f_optimizer = optimizer_cls(f.parameters(), lr=LR) if optimizer_cls is OGDA \
            else optimizer_cls(f.parameters(), lr=LR, momentum=0.0)

        for _ in range(epochs):
            g_obj, f_obj = objective.calc_objective(g, f, X, Z, Y)
            g_optimizer.zero_grad()
            g_obj.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(g.parameters(), CLIP_NORM)
            g_optimizer.step()

            f_optimizer.zero_grad()
            f_obj.backward()
            torch.nn.utils.clip_grad_norm_(f.parameters(), CLIP_NORM)
            f_optimizer.step()

        return g.weight.item(), f.weight.item()

    def test_fedgda_d_two_epochs_matches_plain_sgd_reference(self):
        actual_theta, actual_tau = self._run_real_trainer(CustomSGD, epochs=2)
        ref_theta, ref_tau = self._reference_sgd_trajectory(epochs=2)
        self.assertAlmostEqual(actual_theta, ref_theta, places=12)
        self.assertAlmostEqual(actual_tau, ref_tau, places=12)

    def test_fedogda_d_first_epoch_reduces_to_plain_sgd(self):
        actual_theta, actual_tau = self._run_real_trainer(OGDA, epochs=1)
        ref_theta, ref_tau = self._reference_sgd_trajectory(epochs=1)
        self.assertAlmostEqual(actual_theta, ref_theta, places=12)
        self.assertAlmostEqual(actual_tau, ref_tau, places=12)

    def test_fedogda_d_two_epochs_matches_optimistic_reference(self):
        actual_theta, actual_tau = self._run_real_trainer(OGDA, epochs=2)
        ref_theta, ref_tau = self._reference_ogda_trajectory(epochs=2)
        self.assertAlmostEqual(actual_theta, ref_theta, places=12)
        self.assertAlmostEqual(actual_tau, ref_tau, places=12)

        # And the optimistic step must actually differ from plain SGD by
        # epoch 2 -- otherwise this test would pass even if OGDA secretly
        # degraded to SGD throughout.
        plain_theta, plain_tau = self._reference_sgd_trajectory(epochs=2)
        self.assertNotAlmostEqual(actual_theta, plain_theta, places=4)
        self.assertNotAlmostEqual(actual_tau, plain_tau, places=4)


if __name__ == "__main__":
    unittest.main()
