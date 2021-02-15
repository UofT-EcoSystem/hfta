import functools
import torch.optim

from .adadelta import Adadelta, PartiallyFusedAdadelta
from .adam import Adam, PartiallyFusedAdam
from .lr_scheduler import StepLR
from .utils import (index_array_or_return_scalar,
                    consolidate_hyperparams_and_determine_B)

_OPTIMIZERS_MAP = {
    torch.optim.Adadelta: Adadelta,
    torch.optim.Adam: Adam,
}

_PARTIALLY_FUSED_OPTIMIZERS_MAP = {
    torch.optim.Adadelta: PartiallyFusedAdadelta,
    torch.optim.Adam: PartiallyFusedAdam,
}

_LR_SCHEDULER_MAP = {
    torch.optim.lr_scheduler.StepLR: StepLR,
}


def get_hfta_optim_for(torch_optim_class, B=1, partially_fused=False):
  if B > 0:
    if partially_fused:
      return functools.partial(
          _PARTIALLY_FUSED_OPTIMIZERS_MAP[torch_optim_class],
          B=B,
      )
    else:
      return functools.partial(_OPTIMIZERS_MAP[torch_optim_class], B=B)
  else:
    return torch_optim_class


def get_hfta_lr_scheduler_for(torch_lr_scheduler_class, B=1):
  if B > 0:
    return functools.partial(_LR_SCHEDULER_MAP[torch_lr_scheduler_class], B=B)
  else:
    return torch_lr_scheduler_class
