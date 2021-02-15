import numpy as np
import torch

from torch.optim import Optimizer

from .utils import (_validate_range, _broadcastablize,
                    _move_coeff_to_same_device, _reduce_array_if_possible_for,
                    _zero_grad_if_cuda, index_array_or_return_scalar)
from .partial import PartiallyFusedOptimizer


class Adadelta(Optimizer):
  """Implements Adadelta algorithm.

  It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.

  Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
      parameter groups
    rho (float or a list/tuple/np.array/torch.Tensor of floats, optional):
      coefficient used for computing a running average of squared
      gradients (default: 0.9)
    eps (float or a list/tuple/np.array/torch.Tensor of floats, optional): term
      added to the denominator to improve numerical stability (default: 1e-6)
    lr (float or a list/tuple/np.array/torch.Tensor of floats, optional):
      coefficient that scale delta before it is applied to the parameters
      (default: 1.0)
    weight_decay (float or a list/tuple/np.array/torch.Tensor of floats,
      optional): weight decay (L2 penalty) (default: 0)

  __ https://arxiv.org/abs/1212.5701
  """

  def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0, B=1):
    _validate_range('learning rate', lr, 0.0, float('inf'))
    _validate_range('rho value', rho, 0.0, 1.0)
    _validate_range('epsilon value', eps, 0.0, float('inf'))
    _validate_range('weight_decay value', weight_decay, 0.0, float('inf'))
    lr, rho, eps, weight_decay = _reduce_array_if_possible_for(
        lr, rho, eps, weight_decay)

    defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
    super(Adadelta, self).__init__(params, defaults)
    _broadcastablize(self, 'lr', B)
    _broadcastablize(self, 'rho', B)
    _broadcastablize(self, 'eps', B)
    _broadcastablize(self, 'weight_decay', B)

  def zero_grad(self):
    if not _zero_grad_if_cuda(self):
      super(Adadelta, self).zero_grad()

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Arguments:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad
        if grad.is_sparse:
          raise RuntimeError('Adadelta does not support sparse gradients')
        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          state['square_avg'] = torch.zeros_like(
              p, memory_format=torch.preserve_format)
          state['acc_delta'] = torch.zeros_like(
              p, memory_format=torch.preserve_format)
          _move_coeff_to_same_device(group, 'lr', p)
          _move_coeff_to_same_device(group, 'rho', p)
          _move_coeff_to_same_device(group, 'eps', p)
          _move_coeff_to_same_device(group, 'weight_decay', p)

        square_avg, acc_delta = state['square_avg'], state['acc_delta']
        lr, rho, eps, weight_decay = (group['lr'], group['rho'], group['eps'],
                                      group['weight_decay'])

        state['step'] += 1

        if isinstance(weight_decay, dict) or weight_decay != 0:
          if isinstance(weight_decay, dict):
            grad = grad + weight_decay[p] * p
          else:
            grad = grad.add(p, alpha=weight_decay)

        if isinstance(rho, dict):
          square_avg.mul_(rho[p]).add_((1 - rho[p]) * grad * grad)
        else:
          square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
        if isinstance(eps, dict):
          std = square_avg.add(eps[p]).sqrt_()
          delta = acc_delta.add(eps[p]).sqrt_().div_(std).mul_(grad)
        else:
          std = square_avg.add(eps).sqrt_()
          delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
        if isinstance(lr, dict):
          p.add_(-lr[p] * delta)
        else:
          p.add_(delta, alpha=-lr)
        if isinstance(rho, dict):
          acc_delta.mul_(rho[p]).add_((1 - rho[p]) * delta * delta)
        else:
          acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)

    return loss


class PartiallyFusedAdadelta(PartiallyFusedOptimizer):

  def __init__(
      self,
      fused_params,
      unfused_params,
      lr=1.0,
      rho=0.9,
      eps=1e-6,
      weight_decay=0,
      B=1,
  ):
    fused_adadelta = Adadelta(
        fused_params,
        lr=lr,
        rho=rho,
        eps=eps,
        weight_decay=weight_decay,
        B=B,
    )
    unfused_adadelta = [
        torch.optim.Adadelta(
            params,
            lr=index_array_or_return_scalar(lr, b),
            rho=index_array_or_return_scalar(rho, b),
            eps=index_array_or_return_scalar(eps, b),
            weight_decay=index_array_or_return_scalar(weight_decay, b),
        ) for b, params in enumerate(unfused_params)
    ]
    super(PartiallyFusedAdadelta, self).__init__(
        fused_adadelta,
        unfused_adadelta,
    )
