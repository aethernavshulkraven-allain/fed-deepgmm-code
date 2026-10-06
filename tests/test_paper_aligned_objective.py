"""Tests for PaperAlignedMomentObjective (P0 item #1).

The paper's client-local objective (Section 2 / Study A spec) is:

    U^i_theta~(theta,tau) = M(theta,tau) - (1/4) * R(tau; theta~)
    M(theta,tau) = E[ f_tau(Z,W) * (Y - g_theta(D,W)) ]
    R(tau; theta~) = E[ f_tau(Z,W)^2 * (Y - g_theta~(D,W))^2 ]

"The structural player minimises the objective and the critic maximises it."
theta~ is the frozen previous global iterate, fixed for the whole round.

These tests exist because a sign error here is easy to make invisibly: get
epsilon's sign backwards and the code still runs, still sort of trains, and
can even still look like it's "recovering" something without actually
implementing min_theta sup_tau of the stated U. So correctness is checked two
ways, not one:

1. Closed-form: on a toy linear g/f model where the gradient of M is
   hand-derivable, the autograd gradient must equal the hand-derived formula
   exactly (not "close, probably fine" -- exactly, up to floating point).
2. Empirical: one small gradient-descent step on g_objective must decrease M;
   one small gradient-descent step on f_objective must increase U -- proving
   the implementation drives theta and tau in the directions the paper
   actually assigns to each player, not the other way around.

Plus: theta~ is genuinely frozen (no gradient, doesn't move when the live
model does), lambda is exactly 1/4 (not the legacy tunable 0.1), and the
legacy objective is completely untouched by any of this.
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example"))

from game_objectives.simple_moment_objective import (  # noqa: E402
    OptimalMomentObjective,
    PaperAlignedMomentObjective,
)


class Linear1D(nn.Module):
    """g(x) = theta * x or f(z) = tau * z, no bias -- so the single parameter's
    gradient has a clean, hand-derivable closed form.
    """

    def __init__(self, init_value):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float64))

    def forward(self, v):
        return self.weight * v.squeeze(-1)


def toy_batch(seed=0, n=17):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, dtype=torch.float64, generator=gen)
    z = torch.randn(n, 1, dtype=torch.float64, generator=gen)
    y = torch.randn(n, 1, dtype=torch.float64, generator=gen)
    return x, z, y


class ClosedFormGradientTest(unittest.TestCase):
    """g(x)=theta*x, f(z)=tau*z. Hand-derived:

        M(theta,tau) = mean( tau*z * (y - theta*x) )
        dM/dtheta    = mean( -tau*z*x )   [independent of theta]
        dM/dtau      = mean( z*(y-theta*x) )

    Since g_objective = M exactly (reg term doesn't depend on theta when
    theta~ is frozen), dg_objective/dtheta must equal dM/dtheta exactly.
    """

    def setUp(self):
        self.x, self.z, self.y = toy_batch()
        self.theta0 = 0.37
        self.tau0 = -1.4

    def _build(self, theta_val, tau_val):
        g = Linear1D(theta_val)
        f = Linear1D(tau_val)
        objective = PaperAlignedMomentObjective(lambda_1=0.25)
        objective.set_theta_tilde(g)  # round-0 theta~ = current theta
        return g, f, objective

    def test_g_objective_gradient_matches_hand_derivation(self):
        g, f, objective = self._build(self.theta0, self.tau0)
        g_obj, _ = objective.calc_objective(g, f, self.x, self.z, self.y)
        g_obj.backward()

        with torch.no_grad():
            expected = (-self.tau0 * self.z.squeeze(-1) * self.x.squeeze(-1)).mean()

        self.assertAlmostEqual(g.weight.grad.item(), expected.item(), places=10)

    def test_f_objective_gradient_direction_matches_hand_derivation(self):
        """dM/dtau (theta~ fixed, so its epsilon is constant w.r.t. tau) plus
        the reg term's own tau-gradient, matching -M + lambda*R differentiated
        w.r.t. tau by hand.
        """
        g, f, objective = self._build(self.theta0, self.tau0)
        _, f_obj = objective.calc_objective(g, f, self.x, self.z, self.y)
        f_obj.backward()

        with torch.no_grad():
            epsilon_live = self.y.squeeze(-1) - self.theta0 * self.x.squeeze(-1)
            epsilon_frozen = epsilon_live  # theta~ == theta at round 0
            z_flat = self.z.squeeze(-1)
            # f_objective = -M + lambda*R
            #   dM/dtau = mean(z*epsilon_live)
            #   R = mean((tau*z)^2 * epsilon_frozen^2)
            #   dR/dtau = mean(2*tau*z^2*epsilon_frozen^2)
            d_neg_M = -(z_flat * epsilon_live).mean()
            d_reg = 0.25 * (2 * self.tau0 * z_flat**2 * epsilon_frozen**2).mean()
            expected = d_neg_M + d_reg

        self.assertAlmostEqual(f.weight.grad.item(), expected.item(), places=10)


class DescentAscentDirectionTest(unittest.TestCase):
    """The property the co-author's review specifically asked to have verified:
    g must descend M, f must ascend U = M - lambda*R.
    """

    def _paper_M(self, g, f, x, z, y):
        with torch.no_grad():
            epsilon = torch.squeeze(y) - torch.squeeze(g(x))
            return torch.squeeze(f(z)).mul(epsilon).mean().item()

    def _paper_U(self, g, f, g_tilde, x, z, y, lam=0.25):
        with torch.no_grad():
            m = self._paper_M(g, f, x, z, y)
            eps_frozen = torch.squeeze(y) - torch.squeeze(g_tilde(x))
            r = (torch.squeeze(f(z)) ** 2).mul(eps_frozen ** 2).mean().item()
            return m - lam * r

    def test_one_gradient_descent_step_on_g_decreases_M(self):
        x, z, y = toy_batch(seed=1)
        g = Linear1D(0.8)
        f = Linear1D(1.1)
        objective = PaperAlignedMomentObjective(lambda_1=0.25)
        objective.set_theta_tilde(g)

        m_before = self._paper_M(g, f, x, z, y)
        g_obj, _ = objective.calc_objective(g, f, x, z, y)
        g_obj.backward()
        with torch.no_grad():
            g.weight -= 1e-3 * g.weight.grad  # plain gradient-descent step
        m_after = self._paper_M(g, f, x, z, y)

        self.assertLess(m_after, m_before)

    def test_one_gradient_descent_step_on_f_increases_U(self):
        x, z, y = toy_batch(seed=2)
        g = Linear1D(0.5)
        f = Linear1D(0.9)
        objective = PaperAlignedMomentObjective(lambda_1=0.25)
        objective.set_theta_tilde(g)
        g_tilde_ref = objective._g_tilde  # same frozen snapshot U uses

        u_before = self._paper_U(g, f, g_tilde_ref, x, z, y)
        _, f_obj = objective.calc_objective(g, f, x, z, y)
        f_obj.backward()
        with torch.no_grad():
            f.weight -= 1e-3 * f.weight.grad  # descent on f_objective == ascent on U
        u_after = self._paper_U(g, f, g_tilde_ref, x, z, y)

        self.assertGreater(u_after, u_before)


class FrozenThetaTildeTest(unittest.TestCase):
    def test_theta_tilde_carries_no_gradient(self):
        g = Linear1D(1.0)
        objective = PaperAlignedMomentObjective()
        objective.set_theta_tilde(g)
        for param in objective._g_tilde.parameters():
            self.assertFalse(param.requires_grad)

    def test_mutating_live_g_after_snapshot_does_not_move_theta_tilde(self):
        x, z, y = toy_batch(seed=3)
        g = Linear1D(1.0)
        f = Linear1D(1.0)
        objective = PaperAlignedMomentObjective(lambda_1=0.25)
        objective.set_theta_tilde(g)
        frozen_value_before = objective._g_tilde.weight.item()

        with torch.no_grad():
            g.weight += 5.0  # simulate several local training steps moving g

        frozen_value_after = objective._g_tilde.weight.item()
        self.assertEqual(frozen_value_before, frozen_value_after)
        self.assertNotEqual(g.weight.item(), frozen_value_after)

    def test_reg_term_uses_frozen_epsilon_not_live_epsilon(self):
        """After g moves post-snapshot, R must still reflect the frozen g~,
        not the live (now different) g -- otherwise theta~ isn't fixed at all.
        """
        x, z, y = toy_batch(seed=4)
        g = Linear1D(0.2)
        f = Linear1D(1.0)
        objective = PaperAlignedMomentObjective(lambda_1=0.25)
        objective.set_theta_tilde(g)

        with torch.no_grad():
            epsilon_frozen_expected = torch.squeeze(y) - 0.2 * torch.squeeze(x)
            reg_expected = (
                (torch.squeeze(f(z)) ** 2).mul(epsilon_frozen_expected ** 2).mean()
            )

        with torch.no_grad():
            g.weight += 3.0  # move the live model far from theta~

        _, f_obj = objective.calc_objective(g, f, x, z, y)
        with torch.no_grad():
            m = self._m(g, f, x, z, y)
        reg_actual = (f_obj + m) / objective._lambda_1  # f_obj = -M + lambda*R

        self.assertAlmostEqual(reg_actual.item(), reg_expected.item(), places=8)

    def _m(self, g, f, x, z, y):
        epsilon = torch.squeeze(y) - torch.squeeze(g(x))
        return torch.squeeze(f(z)).mul(epsilon).mean()

    def test_calc_objective_before_set_theta_tilde_raises(self):
        g = Linear1D(1.0)
        f = Linear1D(1.0)
        x, z, y = toy_batch(seed=5)
        objective = PaperAlignedMomentObjective()
        with self.assertRaises(RuntimeError):
            objective.calc_objective(g, f, x, z, y)

    def test_lambda_defaults_to_one_quarter(self):
        self.assertEqual(PaperAlignedMomentObjective()._lambda_1, 0.25)

    def test_theta_tilde_is_a_separate_object_not_an_alias(self):
        g = Linear1D(1.0)
        objective = PaperAlignedMomentObjective()
        objective.set_theta_tilde(g)
        self.assertIsNot(objective._g_tilde, g)
        self.assertIsNot(objective._g_tilde.weight, g.weight)


class LegacyObjectiveUnchangedTest(unittest.TestCase):
    """OptimalMomentObjective (legacy mode) must be untouched by any of this."""

    def test_legacy_still_uses_g_minus_y_and_live_epsilon_in_reg(self):
        x, z, y = toy_batch(seed=6)
        g = Linear1D(0.4)
        f = Linear1D(0.6)
        objective = OptimalMomentObjective(lambda_1=0.1)

        g_obj, f_obj = objective.calc_objective(g, f, x, z, y)

        with torch.no_grad():
            epsilon = torch.squeeze(g(x)) - torch.squeeze(y)  # legacy sign: g - y
            moment = torch.squeeze(f(z)).mul(epsilon).mean()
            f_reg = 0.1 * (torch.squeeze(f(z)) ** 2).mul(epsilon ** 2).mean()

        self.assertAlmostEqual(g_obj.item(), moment.item(), places=10)
        self.assertAlmostEqual(f_obj.item(), (-moment + f_reg).item(), places=10)

    def test_legacy_has_no_set_theta_tilde(self):
        self.assertFalse(hasattr(OptimalMomentObjective(), "set_theta_tilde"))


if __name__ == "__main__":
    unittest.main()
