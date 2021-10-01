import math
import torch
from . import _functional as F
from torch.optim import Optimizer

from .utils import (make_coefficient, reduce_array_if_possible_for,
                    index_array_or_return_scalar)
from .partial import PartiallyFusedOptimizer


class Adam(Optimizer):
  r"""Implements Adam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: amsgrad                      \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: \widehat{v_0}^{max}\leftarrow 0\\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float or a list/tuple/np.array/torch.Tensor of floats, optional): learning rate (default: 1e-3)
        betas (Tuple[float or a list/tuple/np.array/torch.Tensor of floats, float or a list/tuple/np.array/torch.Tensor of floats], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float or a list/tuple/np.array/torch.Tensor of floats, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float or a list/tuple/np.array/torch.Tensor of floats, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

  def __init__(self,
               params,
               lr=1e-3,
               betas=(0.9, 0.999),
               eps=1e-8,
               weight_decay=0,
               amsgrad=False,
               B=1):
    lr, eps, beta1, beta2, weight_decay = reduce_array_if_possible_for(
        lr, eps, betas[0], betas[1], weight_decay)
    betas = (beta1, beta2)
    lr = make_coefficient('learning rate', lr, lb=0.0, ub=float('inf'))
    eps = make_coefficient('epsilon value', eps, lb=0.0, ub=float('inf'))
    betas = make_coefficient('beta parameter at index',
                             betas,
                             lb=0.0,
                             ub=1.0,
                             is_tuple=True)
    weight_decay = make_coefficient('weight_decay value',
                                    weight_decay,
                                    lb=0.0,
                                    ub=float('inf'))
    # TODO(wangshangsam): amsgrad array support.
    defaults = dict(lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                    amsgrad=amsgrad)
    super(Adam, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(Adam, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('amsgrad', False)

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
      exp_avgs = []
      exp_avg_sqs = []
      max_exp_avg_sqs = []
      state_steps = []
      beta1, beta2 = group['betas']

      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          if p.grad.is_sparse:
            raise RuntimeError(
                'Adam does not support sparse gradients, please consider SparseAdam instead'
            )
          grads.append(p.grad)

          state = self.state[p]
          # Lazy state initialization
          if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
            if group['amsgrad']:
              # Maintains max of all exp. moving avg. of sq. grad. values
              state['max_exp_avg_sq'] = torch.zeros_like(
                  p, memory_format=torch.preserve_format)

          exp_avgs.append(state['exp_avg'])
          exp_avg_sqs.append(state['exp_avg_sq'])

          if group['amsgrad']:
            max_exp_avg_sqs.append(state['max_exp_avg_sq'])

          # update the steps for each param group update
          state['step'] += 1
          # record the step after step update
          state_steps.append(state['step'])

      F.adam(params_with_grad,
             grads,
             exp_avgs,
             exp_avg_sqs,
             max_exp_avg_sqs,
             state_steps,
             amsgrad=group['amsgrad'],
             beta1=beta1,
             beta2=beta2,
             lr=group['lr'],
             weight_decay=group['weight_decay'],
             eps=group['eps'])
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
