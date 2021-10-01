import numpy as np
import torch

from . import _functional as F
from torch.optim import Optimizer

from .utils import (make_coefficient, reduce_array_if_possible_for,
                    index_array_or_return_scalar)
from .partial import PartiallyFusedOptimizer


class Adadelta(Optimizer):
  r"""Implements Adadelta algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)},
                \: f(\theta) \text{ (objective)}, \: \rho \text{ (decay)},
                \: \lambda \text{ (weight decay)}                                                \\
            &\textbf{initialize} :  v_0  \leftarrow 0 \: \text{ (square avg)},
                \: u_0 \leftarrow 0 \: \text{ (accumulate variables)}                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm} v_t      \leftarrow v_{t-1} \rho + g^2_t (1 - \rho)                    \\
            &\hspace{5mm}\Delta x_t    \leftarrow   \frac{\sqrt{u_{t-1} +
                \epsilon }}{ \sqrt{v_t + \epsilon}  }g_t \hspace{21mm}                           \\
            &\hspace{5mm} u_t  \leftarrow   u_{t-1}  \rho +
                 \Delta x^2_t  (1 - \rho)                                                        \\
            &\hspace{5mm}\theta_t      \leftarrow   \theta_{t-1} - \gamma  \Delta x_t            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `ADADELTA: An Adaptive Learning Rate Method`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float or a list/tuple/np.array/torch.Tensor of floats, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float or a list/tuple/np.array/torch.Tensor of floats, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float or a list/tuple/np.array/torch.Tensor of floats, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float or a list/tuple/np.array/torch.Tensor of floats, optional): weight decay (L2 penalty) (default: 0)

    .. _ADADELTA\: An Adaptive Learning Rate Method:
        https://arxiv.org/abs/1212.5701
    """

  def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0, B=1):
    lr, rho, eps, weight_decay = reduce_array_if_possible_for(
        lr, rho, eps, weight_decay)
    lr = make_coefficient('learning rate', lr, lb=0.0, ub=float('inf'))
    rho = make_coefficient('rho value', rho, lb=0.0, ub=1.0)
    eps = make_coefficient('epsilon value', eps, lb=0.0, ub=float('inf'))
    weight_decay = make_coefficient('weight_decay value',
                                    weight_decay,
                                    lb=0.0,
                                    ub=float('inf'))

    defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
    super(Adadelta, self).__init__(params, defaults)

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
      params_with_grad = []
      grads = []
      square_avgs = []
      acc_deltas = []
      lr, rho, eps, weight_decay = group['lr'], group['rho'], group[
          'eps'], group['weight_decay']

      for p in group['params']:
        if p.grad is None:
          continue
        params_with_grad.append(p)
        if p.grad.is_sparse:
          raise RuntimeError('Adadelta does not support sparse gradients')
        grads.append(p.grad)

        state = self.state[p]

        # Lazy state initialization
        if len(state) == 0:
          state['step'] = 0
          state['square_avg'] = torch.zeros_like(
              p, memory_format=torch.preserve_format)
          state['acc_delta'] = torch.zeros_like(
              p, memory_format=torch.preserve_format)

        square_avgs.append(state['square_avg'])
        acc_deltas.append(state['acc_delta'])

        state['step'] += 1

      F.adadelta(params_with_grad,
                 grads,
                 square_avgs,
                 acc_deltas,
                 lr=lr,
                 rho=rho,
                 eps=eps,
                 weight_decay=weight_decay)

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
    fused_params = list(fused_params)
    if len(fused_params) == 0:
      fused_adadelta = None
    else:
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
