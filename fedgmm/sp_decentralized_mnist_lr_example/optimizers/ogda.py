import torch
from torch.optim import Optimizer

class OGDA(Optimizer):
    """
    Implements Optimistic Gradient Descent Ascent (OGDA).
    
    The update rule is:
    x_{t+1} = x_t - 2 * lr * g_t + lr * g_{t-1}
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        maximize (bool, optional): whether to maximize the objective (ascent) (default: False)
    """

    def __init__(self, params, lr, maximize=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        defaults = dict(lr=lr, maximize=maximize)
        super(OGDA, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Previous gradient initialized to zeros
                    # state['prev_grad'] = torch.zeros_like(p.data)
                    state['prev_grad'] = grad.clone()

                prev_grad = state['prev_grad']
                lr = group['lr']
                maximize = group['maximize']
                
                state['step'] += 1

                # OGDA Update: 
                # If minimize: x = x - (2 * lr * grad - lr * prev_grad)
                # If maximize: x = x + (2 * lr * grad - lr * prev_grad)
                
                update = 2.0 * lr * grad - lr * prev_grad
                
                if maximize:
                    p.data.add_(update)
                else:
                    p.data.add_(-update)

                # Store current gradient as prev_grad for next step
                prev_grad.copy_(grad)

        return loss
