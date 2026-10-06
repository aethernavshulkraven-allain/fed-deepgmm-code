

class OptimizerFactory(object):
    def __init__(self, optimizer_class, **optimizer_kwargs):
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

    def __str__(self):
        out = self.optimizer_class.__name__ + "::"
        for k, v in self.optimizer_kwargs.items():
            out += ":%r=%r" % (k, v)
        return out

    def __call__(self, model):
        return self.optimizer_class(model.parameters(), **self.optimizer_kwargs)
# from opacus.optimizers import DPOptimizer
# import torch

# class DPOAdam(DPOptimizer, torch.optim.Adam):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, noise_multiplier=1.0, max_grad_norm=1.0):
#         # Initialize the base Adam optimizer within DPOptimizer
#         super(DPOAdam, self).__init__(
#             optimizer=torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
#             noise_multiplier=noise_multiplier,
#             max_grad_norm=max_grad_norm,
#             expected_batch_size=1024
#         )

#     def backward(self, loss):
#         """
#         Computes gradients for all model parameters with respect to the loss.
#         Per-sample gradients will be computed for DP compliance.
#         """
#         loss.backward()

#     def step(self, closure=None):
#         """
#         Perform a single optimization step:
#         1. Clip per-sample gradients.
#         2. Add noise to the clipped gradients.
#         3. Update model parameters.
#         """
#         return super(DPOAdam, self).step(closure)

#     def state_dict(self):
#         """
#         Returns the state of the optimizer as a dict.
#         """
#         return super(DPOAdam, self).state_dict()

#     def load_state_dict(self, state_dict):
#         """
#         Loads the optimizer state.
#         """
#         super(DPOAdam, self).load_state_dict(state_dict)

