import math
import torch
from torch import Tensor
from typing import List, Optional, Union

from .utils import Coefficient, is_coefficient


def adadelta(
    params: List[Tensor],
    grads: List[Tensor],
    square_avgs: List[Tensor],
    acc_deltas: List[Tensor],
    *,
    lr: Union[float, Coefficient],
    rho: Union[float, Coefficient],
    eps: Union[float, Coefficient],
    weight_decay: [float, Coefficient],
):
  r"""Functional API that performs Adadelta algorithm computation.
    See :class:`~torch.optim.Adadelta` for details.
    """

  for (param, grad, square_avg, acc_delta) in zip(params, grads, square_avgs,
                                                  acc_deltas):
    if is_coefficient(weight_decay) or weight_decay != 0:
      if is_coefficient(weight_decay):
        grad = grad + weight_decay[param] * param
      else:
        grad = grad.add(param, alpha=weight_decay)

    if is_coefficient(rho):
      square_avg.mul_(rho[param]).add_((1 - rho[param]) * grad * grad)
    else:
      square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
    if is_coefficient(eps):
      std = square_avg.add(eps[param]).sqrt_()
      delta = acc_delta.add(eps[param]).sqrt_().div_(std).mul_(grad)
    else:
      std = square_avg.add(eps).sqrt_()
      delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
    if is_coefficient(lr):
      param.add_(-lr[param] * delta)
    else:
      param.add_(delta, alpha=-lr)
    if is_coefficient(rho):
      acc_delta.mul_(rho[param]).add_((1 - rho[param]) * delta * delta)
    else:
      acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)


def adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    *,
    amsgrad: bool,
    beta1: Union[float, Coefficient],
    beta2: Union[float, Coefficient],
    lr: Union[float, Coefficient],
    weight_decay: Union[float, Coefficient],
    eps: Union[float, Coefficient],
):
  r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

  for i, param in enumerate(params):

    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step = state_steps[i]

    if is_coefficient(beta1):
      bias_correction1 = 1 - beta1[param]**step
    else:
      bias_correction1 = 1 - beta1**step
    if is_coefficient(beta2):
      sqrt_bias_correction2 = (1 - beta2[param]**step).sqrt()
    else:
      sqrt_bias_correction2 = math.sqrt(1 - beta2**step)

    if is_coefficient(weight_decay) or weight_decay != 0:
      if is_coefficient(weight_decay):
        grad = grad + weight_decay[param] * param
      else:
        grad = grad.add(param, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    if is_coefficient(beta1):
      exp_avg.mul_(beta1[param]).add_((1 - beta1[param]) * grad)
    else:
      exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    if is_coefficient(beta2):
      exp_avg_sq.mul_(beta2[param]).add_(
          (1 - beta2[param]) * grad * grad.conj())
    else:
      exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
    if amsgrad:
      # Maintains the maximum of all 2nd moment running avg. till now
      torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
      # Use the max. for normalizing running avg. of gradient
      if is_coefficient(eps):
        denom = (max_exp_avg_sqs[i].sqrt() / sqrt_bias_correction2).add_(
            eps[param])
      else:
        denom = (max_exp_avg_sqs[i].sqrt() / sqrt_bias_correction2).add_(eps)
    else:
      if is_coefficient(eps):
        denom = (exp_avg_sq.sqrt() / sqrt_bias_correction2).add_(eps[param])
      else:
        denom = (exp_avg_sq.sqrt() / sqrt_bias_correction2).add_(eps)

    if is_coefficient(lr):
      step_size = lr[param] / bias_correction1
    else:
      step_size = lr / bias_correction1

    if torch.is_tensor(step_size):
      param.add_(-step_size * (exp_avg / denom))
    else:
      param.addcdiv_(exp_avg, denom, value=-step_size)
