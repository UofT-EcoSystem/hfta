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
