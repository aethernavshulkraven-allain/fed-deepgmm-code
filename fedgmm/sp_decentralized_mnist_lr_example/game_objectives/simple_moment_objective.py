import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from game_objectives.abstract_objective import AbstractObjective


class SimpleMomentObjective(AbstractObjective):
    def __init__(self):
        AbstractObjective.__init__(self)

    def calc_objective(self, g, f, x, z, y):
        g_of_x = torch.squeeze(g(x))
        f_of_z = torch.squeeze(f(z))
        y = torch.squeeze(y)
        moment = f_of_z.mul(y - g_of_x).mean()
        return moment, -moment


class NormalizedMomentObjective(AbstractObjective):
    def __init__(self, lambda_1=100.0, lambda_2=1.0, lambda_3=0.0):
        AbstractObjective.__init__(self)
        self._lambda_1 = lambda_1
        self._lambda_2 = lambda_2
        self._lambda_3 = lambda_3

    def __str__(self):
        return ("NormalizedObjective::lambda_1=%f:lambda_2=%f:lambda_3=%f"
                % (self._lambda_1, self._lambda_2, self._lambda_3))

    def calc_objective(self, g, f, x, z, y):
        epsilon = y - torch.squeeze(g(x))
        f_of_z = torch.squeeze(f(z))
        # f_of_z = f_of_z - f_of_z.mean()
        raw_moment = f_of_z.mul(epsilon).mean()
        denominator = (f_of_z ** 2).mul(epsilon ** 2).mean() ** 0.5
        # denominator = denominator.detach()
        moment_norm = raw_moment / denominator
        # print(denominator)
        f_reg = self._lambda_1 * (F.relu(f_of_z.abs() - 0.3)).mean()
        f_reg += self._lambda_2 * (f_of_z.mean() ** 2)
        # f_reg /= denominator
        # g_reg = 0.02 * ((epsilon - epsilon.mean()) ** 2).mean()
        g_reg = self._lambda_3 * F.relu((epsilon ** 2).mean() ** 0.5 - 1.0)
        # f_reg = 0
        # g_reg = 0

        return moment_norm + g_reg, -moment_norm + f_reg


class RegularizedMomentObjective(AbstractObjective):
    def __init__(self, lambda_1=0.1, lambda_2=1.0, lambda_3=0.0):
        AbstractObjective.__init__(self)
        self._lambda_1 = lambda_1
        self._lambda_2 = lambda_2
        self._lambda_3 = lambda_3

    def __str__(self):
        return ("RegObjective::lambda_1=%f:lambda_2=%f:lambda_3=%f"
                % (self._lambda_1, self._lambda_2, self._lambda_3))

    def calc_objective(self, g, f, x, z, y):
        g_of_x = torch.squeeze(g(x))
        f_of_z = torch.squeeze(f(z))
        y = torch.squeeze(y)
        moment = f_of_z.mul(y - g_of_x).mean()
        regularizer_1 = torch.nn.functional.mse_loss(f_of_z - f_of_z.mean(), torch.tensor(5.0).double().to(moment.device)) 
        regularizer_2 = f_of_z.mean()**2
        return moment, -moment + self._lambda_1*regularizer_1 + self._lambda_2*regularizer_2


class HingeRegularizedMomentObjective(RegularizedMomentObjective):
    def __str__(self):
        return ("HingeRegObjective::lambda_1=%f:lambda_2=%f:lambda_3=%f"
                % (self._lambda_1, self._lambda_2, self._lambda_3))

    def calc_objective(self, g, f, x, z, y):
        g_of_x = torch.squeeze(g(x))
        f_of_z = torch.squeeze(f(z))
        y = torch.squeeze(y)
        moment = f_of_z.mul(y - g_of_x).mean()
        regularizer_1 = (torch.nn.functional.relu(f_of_z.abs()-0.3)).mean() 
        regularizer_2 = f_of_z.mean()**2
        # g_reg = 100.0 * (F.relu(y - g_of_x).abs() - 0.3).mean()
        g_reg = self._lambda_3 * F.relu(((y - g_of_x) ** 2).mean() ** 0.5 - 1.0)
        return moment + g_reg, -moment + self._lambda_1*regularizer_1 + self._lambda_2*regularizer_2


class OptimalMomentObjective(AbstractObjective):
    def __init__(self, lambda_1=0.1):
        AbstractObjective.__init__(self)
        self._lambda_1 = lambda_1

    def __str__(self):
        return "OptimalObjective::lambda_1=%f" % self._lambda_1
    
    def calc_objective(self, g, f, x, z, y):
        g_of_x = torch.squeeze(g(x))
        f_of_z = torch.squeeze(f(z))
        y = torch.squeeze(y)
        epsilon = g_of_x - y
        
        
        moment = f_of_z.mul(epsilon).mean()
        
        f_reg = self._lambda_1 * (f_of_z ** 2).mul(epsilon ** 2).mean()

        # f_reg = self._lambda_1 * (f_of_z ** 2).mul(epsilon ** 2).mean()
        g_reg = 0.0
        
        return moment + g_reg  , -moment + f_reg

        # return (epsilon ** 2).mean(), -moment + f_reg


class PaperAlignedMomentObjective(AbstractObjective):
    """U^i_theta~(theta,tau) = M(theta,tau) - lambda_1 * R(tau; theta~).

    where, using the paper's sign convention (epsilon = Y - g(X), not g(X) - Y):

        M(theta,tau) = E[ f_tau(Z,W) * (Y - g_theta(D,W)) ]
        R(tau)       = E[ f_tau(Z,W)^2 * (Y - g_theta~(D,W))^2 ]

    theta~ ("g_tilde") is the **frozen previous global structural iterate**,
    fixed for the whole communication round -- it does not depend on the live
    theta being optimized this round, and carries no gradient. Supply the same
    round-base state via ``set_theta_tilde`` for every client phase; this class
    intentionally does not fall back to the live model if it hasn't been set,
    because that would silently reproduce a different (non-frozen) objective.

    The structural player minimizes U over theta; since R does not depend on
    theta (theta~ is frozen), that is equivalent to minimizing M alone, so
    g_objective = M directly. The critic maximizes U over tau, which as a loss
    to be *minimized* by the f-optimizer is f_objective = -M + lambda_1*R. Both
    directions are checked against hand-derived closed-form gradients in
    tests/test_paper_aligned_objective.py.
    """

    def __init__(self, lambda_1=0.25):
        AbstractObjective.__init__(self)
        self._lambda_1 = lambda_1
        self._g_tilde = None

    def __str__(self):
        return "PaperAlignedObjective::lambda_1=%f" % self._lambda_1

    def set_theta_tilde(self, g, state_dict=None):
        """Snapshot ``g``'s current parameters as the frozen theta~ reference.

        Each client phase receives the global iterate at the start of the
        round; its value stays fixed while theta moves locally. For an
        EG correction, state_dict supplies the round base rather than the
        live look-ahead state. Parameters AND buffers are copied; g is untouched.
        """
        g_tilde = copy.deepcopy(g)
        if state_dict is not None:
            g_tilde.load_state_dict(state_dict, strict=True)
        for param in g_tilde.parameters():
            param.requires_grad_(False)
        g_tilde.eval()
        self._g_tilde = g_tilde

    def calc_objective(self, g, f, x, z, y):
        if self._g_tilde is None:
            raise RuntimeError(
                "PaperAlignedMomentObjective.calc_objective called before "
                "set_theta_tilde: theta~ (the frozen previous global iterate) "
                "has not been snapshotted for this round."
            )

        y = torch.squeeze(y)
        f_of_z = torch.squeeze(f(z))
        epsilon_live = y - torch.squeeze(g(x))

        moment = f_of_z.mul(epsilon_live).mean()

        with torch.no_grad():
            epsilon_frozen = y - torch.squeeze(self._g_tilde(x))
        reg = (f_of_z ** 2).mul(epsilon_frozen ** 2).mean()

        g_objective = moment
        f_objective = -moment + self._lambda_1 * reg
        return g_objective, f_objective
