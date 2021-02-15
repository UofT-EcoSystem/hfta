import itertools
import numpy as np
import torch


class PartiallyFusedOptimizer:

  def __init__(
      self,
      fused_optimizer,
      unfused_optimizers,
  ):
    self._fused_optimizer = fused_optimizer
    self._unfused_optimizers = unfused_optimizers

  def step(self, *args, **kwargs):
    """
    fused_optimizer_args: tuple
    fused_optimizer_kwargs: dict
    unfused_optimizers_args: iterable of tuples
    unfused_optimizers_kwargs: iterable of dicts
    """
    fused_ret = self._fused_optimizer.step(*args, **kwargs)
    unfused_rets = [
        ufo.step(*args, **kwargs) for ufo in self._unfused_optimizers
    ]
    return fused_ret, unfused_rets

  @property
  def param_groups(self):
    return itertools.chain(
        self._fused_optimizer.param_groups,
        *(ufo.param_groups for ufo in self._unfused_optimizers),
    )
