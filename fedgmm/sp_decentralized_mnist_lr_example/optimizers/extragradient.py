import torch
from torch.optim import Optimizer


class ExtraGradient(Optimizer):
    """SGD-style extragradient with an explicit look-ahead barrier.

    ``extrapolation()`` saves the current parameters and applies the predictor
    gradient. ``step()`` restores those saved parameters and applies the
    correction gradient evaluated at the look-ahead point. If ``step()`` is
    called without a preceding extrapolation it behaves like ordinary SGD;
    this keeps the optimizer compatible with the repository's model-selection
    helpers.
    """

    def __init__(self, params, lr, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

    @staticmethod
    def _direction(parameter, group):
        direction = parameter.grad.detach()
        if group["weight_decay"] != 0.0:
            direction = direction.add(parameter.detach(), alpha=group["weight_decay"])
        return direction

    @torch.no_grad()
    def extrapolation(self):
        """Save the base iterate and move to the predictor point.

        Writes go through ``.data`` rather than an autograd-tracked in-place
        op. Every other optimizer in this repo (CustomSGD, SGDA, OGDA) does
        the same -- their objectives are evaluated jointly (f's loss graph
        traces back through g's forward output, e.g.
        ``game_objectives/simple_moment_objective.py``), and the model-
        selection and client-training loops both call ``g_optimizer.step()``
        before ``f_obj.backward()``. A version-tracked write there makes
        autograd correctly refuse to run that second backward. ``.data``
        keeps this optimizer a drop-in match for the others' behavior in
        those shared call sites instead of being the first one to surface
        that ordering.
        """
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "eg_base" in state:
                    raise RuntimeError(
                        "ExtraGradient extrapolation called twice without correction"
                    )
                state["eg_base"] = parameter.detach().clone()
                parameter.data.add_(
                    self._direction(parameter, group), alpha=-group["lr"]
                )

    @torch.no_grad()
    def step(self, closure=None):
        """Apply the correction gradient from the saved base iterate."""
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                base = state.pop("eg_base", None)
                direction = self._direction(parameter, group)
                if base is not None:
                    parameter.data.copy_(base)
                parameter.data.add_(direction, alpha=-group["lr"])
