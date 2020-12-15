import random
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import sys
sys.path.append('../')
from ops import testcase_automator
from optim import (get_hfta_optim_for, get_hfta_lr_scheduler_for,
                   index_array_or_return_scalar)
from utils import (_TestNet, _init_test_nets_with_grads,
                   _take_step_on_test_optimizers, _verify_test_nets_params)


def _take_step_on_test_lr_schedulers(lr_scheduler_fused, lr_scheduler_array):
  B = len(lr_scheduler_array)
  random_type = random.choice([list, tuple, np.array, torch.as_tensor])
  random_epochs = random_type([random.randint(5, 15) for _ in range(B)])
  use_random_epoch = random.choice([True, False])

  for b, scheduler in enumerate(lr_scheduler_array):
    scheduler.step(epoch=index_array_or_return_scalar(random_epochs, b)
                   if use_random_epoch else None)
  lr_scheduler_fused.step(epoch=random_epochs if use_random_epoch else None)


def _init_initial_lr(optimizer_fused, optimizer_array):
  for optimizer in optimizer_array:
    for group in optimizer.param_groups:
      group['initial_lr'] = group['lr']
  for group in optimizer_fused.param_groups:
    if isinstance(group['lr'], dict):
      group['initial_lr'] = {
          p: lr.detach().clone() for p, lr in group['lr'].items()
      }
    else:
      group['initial_lr'] = group['lr']


def testcase_StepLR(B=3, step_size=2, gamma=0.1, last_epoch=-1):
  lr = random.choice([torch.rand((B,)), random.random()])
  net_array = [_TestNet() for _ in range(B)]
  net_fused = _TestNet(B=B)
  optimizer_array = [
      optim.Adadelta(
          net_array[b].parameters(),
          lr=index_array_or_return_scalar(lr, b),
      ) for b in range(B)
  ]
  optimizer_fused = get_hfta_optim_for(optim.Adadelta, B=B)(
      net_fused.parameters(),
      lr=lr,
  )
  if not isinstance(last_epoch, int) or last_epoch != -1:
    _init_initial_lr(optimizer_fused, optimizer_array)
  lr_scheduler_array = [
      lr_scheduler.StepLR(
          optimizer_array[b],
          index_array_or_return_scalar(step_size, b),
          gamma=index_array_or_return_scalar(gamma, b),
          last_epoch=index_array_or_return_scalar(last_epoch, b),
      ) for b in range(B)
  ]
  lr_scheduler_fused = get_hfta_lr_scheduler_for(lr_scheduler.StepLR, B=B)(
      optimizer_fused,
      step_size,
      gamma=gamma,
      last_epoch=last_epoch,
  )
  _init_test_nets_with_grads(net_fused, net_array)
  for _ in range(10):
    _take_step_on_test_optimizers(optimizer_fused, optimizer_array)
    _take_step_on_test_lr_schedulers(lr_scheduler_fused, lr_scheduler_array)
  _verify_test_nets_params(net_fused, net_array)


if __name__ == '__main__':
  testcase_automator(
      testcase_StepLR,
      {
          'B': [1, 5, 8],
          'step_size': [
              (2, 3, 4),
              [2, 3, 4],
              np.array([2, 3, 4]),
              torch.as_tensor([2, 3, 4]),
          ],
          'gamma': [
              (0.1, 0.2, 0.3),
              [0.1, 0.2, 0.3],
              np.array([0.1, 0.2, 0.3]),
              torch.as_tensor([0.1, 0.2, 0.3]),
          ],
          'last_epoch': [
              5,
              (5, 6, 7),
              [5, 6, 7],
              np.array([5, 6, 7]),
              torch.as_tensor([5, 6, 7]),
          ],
      },
  )
