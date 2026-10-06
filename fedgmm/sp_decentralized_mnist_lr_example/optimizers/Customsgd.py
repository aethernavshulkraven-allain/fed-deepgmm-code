import torch

class CustomSGD(torch.optim.Optimizer):
    """Implements stochastic gradient descent (optionally with momentum).

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
    """

    def __init__(self, params, lr, momentum=0, weight_decay=0, dampening=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening)
        super(CustomSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = buf
                p.data.add_(d_p, alpha=-lr)

        return loss
from torch.optim.optimizer import Optimizer, required
class SGDA(Optimizer):
    """
    Implements Stochastic Gradient Descent Ascent.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        maximize (bool, optional): Whether to apply gradient ascent (default: False)
        
    Example:
        >>> optimizer = SGDA(model.parameters(), lr=0.1, weight_decay=0.01, maximize=False)
        >>> optimizer.zero_grad()
        >>> loss = model(input)
        >>> loss.backward()
        >>> optimizer.step()
    """
    
    def __init__(self, params, lr=required, weight_decay=0, maximize=False):
        # if lr is not required and not 0.0 <= lr:
        #     raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        
        defaults = dict(lr=lr, weight_decay=weight_decay, maximize=maximize)
        super(SGDA, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Returns:
            The loss from the closure if it is not None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                if group['maximize']:
                    p.data.add_(grad, alpha=group['lr'])
                else:
                    p.data.add_(grad, alpha=-group['lr'])

        return loss


