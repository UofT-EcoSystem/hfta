import itertools
import numpy as np
import torch

from .utils import index_array_or_return_scalar, _zero_grad_if_cuda


class PartiallyFusedOptimizer:

  def __init__(
      self,
      fused_optimizer,
      unfused_optimizers,
  ):
    self._fused_optimizer = fused_optimizer
    self._unfused_optimizers = unfused_optimizers

  def zero_grad(self):
    if self._fused_optimizer is not None:
      self._fused_optimizer.zero_grad()
    for ufo in self._unfused_optimizers:
      if not _zero_grad_if_cuda(ufo):
        ufo.zero_grad()

  def step(self, closure=None):
    if self._fused_optimizer is not None:
      fused_ret = self._fused_optimizer.step(closure=closure)
    else:
      fused_ret = None
    unfused_rets = [
        ufo.step(closure=closure) for ufo in self._unfused_optimizers
    ]
    return fused_ret, unfused_rets

  @property
  def param_groups(self):
    if self._fused_optimizer is not None:
      return itertools.chain(
          self._fused_optimizer.param_groups,
          *(ufo.param_groups for ufo in self._unfused_optimizers),
      )
    else:
      return itertools.chain(*(ufo.param_groups for ufo in self._unfused_optimizers))


  @property
  def fused_param_groups(self):
    if self._fused_optimizer is not None:
      return self._fused_optimizer.param_groups
    else:
      return []

  @property
  def unfused_param_groups(self):
    """ Returns iterable of param_groups; each param_groups corresponds to a
    single unfused optimizer.
    """
    return (ufo.param_groups for ufo in self._unfused_optimizers)


class PartiallyFusedLRScheduler:

  def __init__(self, fused_lr_scheduler, unfused_lr_schedulers):
    self._fused_lr_scheduler = fused_lr_scheduler
    self._unfused_lr_schedulers = unfused_lr_schedulers

  def step(self, epoch=None):
    self._fused_lr_scheduler.step(epoch=epoch)
    for b, ufls in enumerate(self._unfused_lr_schedulers):
      ufls.step(epoch=index_array_or_return_scalar(epoch, b))

  def state_dict(self):
    return {
        'fused': self._fused_lr_scheduler.state_dict(),
        'unfused': [ufls.state_dict() for ufls in self._unfused_lr_schedulers]
    }

  def load_state_dict(self, state_dict):
    self._fused_lr_scheduler.load_state_dict(state_dict['fused'])
    for ufls, sd in zip(self._unfused_lr_schedulers, state_dict['unfused']):
      ufls.load_state_dict(sd)

  def get_last_lr(self):
    return (
        self._fused_lr_scheduler.get_last_lr(),
        [ufls.get_last_lr() for ufls in self._unfused_lr_schedulers],
    )

  def get_lr(self):
    return (
        self._fused_lr_scheduler.get_lr(),
        [ufls.get_lr() for ufls in self._unfused_lr_schedulers],
    )
