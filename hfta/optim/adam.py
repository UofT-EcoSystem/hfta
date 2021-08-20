import math
import torch

from torch.optim import Optimizer

from .utils import (_validate_range, _broadcastablize,
                    _move_coeff_to_same_device, _reduce_array_if_possible_for,
                    _zero_grad_if_cuda, index_array_or_return_scalar)
from .partial import PartiallyFusedOptimizer


class Adam(Optimizer):
  r"""Implements Adam algorithm.

  It has been proposed in `Adam: A Method for Stochastic Optimization`_.
  The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

  Args:
    params (iterable): iterable of parameters to optimize or dicts defining
      parameter groups
    lr (float or a list/tuple/np.array/torch.Tensor of floats, optional):
      learning rate (default: 1e-3)
    betas (Tuple[float or a list/..., float or a list/...], optional):
      coefficients used for computing running averages of gradient and its
      square (default: (0.9, 0.999))
    eps (float or a list/tuple/np.array/torch.Tensor of floats, optional): term
      added to the denominator to improve numerical stability (default: 1e-8)
    weight_decay (float or a list/..., optional): weight decay (L2 penalty)
      (default: 0)
    amsgrad (boolean, optional): whether to use the AMSGrad variant of this
      algorithm from the paper `On the Convergence of Adam and Beyond`_
      (default: False)

  .. _Adam\: A Method for Stochastic Optimization:
      https://arxiv.org/abs/1412.6980
  .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
  .. _On the Convergence of Adam and Beyond:
      https://openreview.net/forum?id=ryQu7f-RZ
  """

  def __init__(
      self,
      params,
      lr=1e-3,
      betas=(0.9, 0.999),
      eps=1e-8,
      weight_decay=0,
      amsgrad=False,
      B=1,
  ):
    _validate_range('learning rate', lr, 0.0, float('inf'))
    _validate_range('epsilon value', eps, 0.0, float('inf'))
    _validate_range('beta parameter at index 0', betas[0], 0.0, 1.0)
    _validate_range('beta parameter at index 1', betas[1], 0.0, 1.0)
    _validate_range('weight_decay value', weight_decay, 0.0, float('inf'))
    lr, eps, beta1, beta2, weight_decay = _reduce_array_if_possible_for(
        lr, eps, betas[0], betas[1], weight_decay)
    betas = (beta1, beta2)

    defaults = dict(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,  # TODO(wangshangsam): amsgrad array support.
    )
    super(Adam, self).__init__(params, defaults)
    _broadcastablize(self, 'lr', B)
    _broadcastablize(self, 'eps', B)
    _broadcastablize(self, 'betas', B, is_tuple=True)
    _broadcastablize(self, 'weight_decay', B)

  def __setstate__(self, state):
    super(Adam, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('amsgrad', False)

  def zero_grad(self):
    if not _zero_grad_if_cuda(self):
      super(Adam, self).zero_grad()

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      beta1, beta2 = group['betas']

      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad
        if grad.is_sparse:
          raise RuntimeError('Adam does not support sparse gradients, please '
                             'consider SparseAdam instead')
        amsgrad = group['amsgrad']

        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(
              p, memory_format=torch.preserve_format)
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = torch.zeros_like(
              p, memory_format=torch.preserve_format)
          if amsgrad:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state['max_exp_avg_sq'] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
          _move_coeff_to_same_device(group, 'lr', p)
          _move_coeff_to_same_device(group, 'eps', p)
          _move_coeff_to_same_device(group, 'betas', p, is_tuple=True)
          _move_coeff_to_same_device(group, 'weight_decay', p)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
          max_exp_avg_sq = state['max_exp_avg_sq']
        lr, eps, weight_decay = group['lr'], group['eps'], group['weight_decay']

        state['step'] += 1

        # Start of functional computation
        def get_bias_correction(beta, step):
          if isinstance(beta, dict):
            bias_correction = 1 - beta[p]**step
          else:
            bias_correction = 1 - beta**step
          return bias_correction

        bias_correction1 = get_bias_correction(beta1, state['step'])
        bias_correction2 = get_bias_correction(beta2, state['step'])
        sqrt_bias_correction2 = bias_correction2.sqrt() if isinstance(
            bias_correction2, torch.Tensor) else math.sqrt(bias_correction2)

        if isinstance(weight_decay, dict) or weight_decay != 0:
          if isinstance(weight_decay, dict):
            grad = grad + weight_decay[p] * p
          else:
            grad = grad.add(p, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        if isinstance(beta1, dict):
          exp_avg.mul_(beta1[p]).add_((1 - beta1[p]) * grad)
        else:
          exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if isinstance(beta2, dict):
          exp_avg_sq.mul_(beta2[p]).add_((1 - beta2[p]) * grad * grad)
        else:
          exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if amsgrad:
          # Maintains the maximum of all 2nd moment running avg. till now
          torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
          # Use the max. for normalizing running avg. of gradient
          if isinstance(eps, dict):
            denom = (max_exp_avg_sq.sqrt() / sqrt_bias_correction2).add_(eps[p])
          else:
            denom = (max_exp_avg_sq.sqrt() / sqrt_bias_correction2).add_(eps)
        else:
          if isinstance(eps, dict):
            denom = (exp_avg_sq.sqrt() / sqrt_bias_correction2).add_(eps[p])
          else:
            denom = (exp_avg_sq.sqrt() / sqrt_bias_correction2).add_(eps)

        if isinstance(lr, dict):
          step_size = lr[p] / bias_correction1
        else:
          step_size = lr / bias_correction1

        if isinstance(step_size, torch.Tensor):
          p.add_(-step_size * (exp_avg / denom))
        else:
          p.addcdiv_(exp_avg, denom, value=-step_size)

    return loss


class PartiallyFusedAdam(PartiallyFusedOptimizer):

  def __init__(
      self,
      fused_params,
      unfused_params,
      lr=1e-3,
      betas=(0.9, 0.999),
      eps=1e-8,
      weight_decay=0,
      amsgrad=False,
      B=1,
  ):
    fused_params = list(fused_params)
    if len(fused_params) == 0:
      fused_adam = None
    else:
      fused_adam = Adam(
          fused_params,
          lr=lr,
          betas=betas,
          eps=eps,
          weight_decay=weight_decay,
          amsgrad=amsgrad,
          B=B,
      )
    unfused_adams = [
        torch.optim.Adam(
            params,
            lr=index_array_or_return_scalar(lr, b),
            betas=(
                index_array_or_return_scalar(betas[0], b),
                index_array_or_return_scalar(betas[1], b),
            ),
            eps=index_array_or_return_scalar(eps, b),
            weight_decay=index_array_or_return_scalar(weight_decay, b),
            amsgrad=amsgrad,
        ) for b, params in enumerate(unfused_params)
    ]
    super(PartiallyFusedAdam, self).__init__(fused_adam, unfused_adams)
