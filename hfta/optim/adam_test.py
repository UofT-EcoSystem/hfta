import numpy as np
import random
import torch
import torch.optim as optim

from hfta.ops import testcase_automator
from hfta.optim import get_hfta_optim_for, index_array_or_return_scalar
from utils import _TestNet, _optim_testing_procedure


def testcase_fused(
    B=3,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    amsgrad=False,
):
  net_array = [_TestNet() for _ in range(B)]
  net_fused = _TestNet(B=B)
  optimizer_array = [
      optim.Adam(
          net_array[b].parameters(),
          lr=index_array_or_return_scalar(lr, b),
          betas=(
              index_array_or_return_scalar(betas[0], b),
              index_array_or_return_scalar(betas[1], b),
          ),
          eps=index_array_or_return_scalar(eps, b),
          weight_decay=index_array_or_return_scalar(weight_decay, b),
          amsgrad=amsgrad,
      ) for b in range(B)
  ]
  optimizer_fused = get_hfta_optim_for(optim.Adam, B=B)(
      net_fused.parameters(),
      lr=lr,
      betas=betas,
      eps=eps,
      weight_decay=weight_decay,
      amsgrad=amsgrad,
  )
  _optim_testing_procedure(net_fused, net_array, optimizer_fused,
                           optimizer_array)


def testcase_partially_fused(B=3, amsgrad=False):
  net_array = [_TestNet() for _ in range(B)]
  net_fused = _TestNet(B=B, partially_fused=True)
  lr = [random.uniform(1e-4, 1e-2) for _ in range(B)]
  betas = (
      [random.uniform(0.8, 0.99) for _ in range(B)],
      [random.uniform(0.998, 0.9999) for _ in range(B)],
  )
  eps = [random.uniform(1e-9, 1e-7) for _ in range(B)]
  weight_decay = [random.uniform(0.0, 0.3) for _ in range(B)]
  optimizer_array = [
      optim.Adam(
          net_array[b].parameters(),
          lr=index_array_or_return_scalar(lr, b),
          betas=(
              index_array_or_return_scalar(betas[0], b),
              index_array_or_return_scalar(betas[1], b),
          ),
          eps=index_array_or_return_scalar(eps, b),
          weight_decay=index_array_or_return_scalar(weight_decay, b),
          amsgrad=amsgrad,
      ) for b in range(B)
  ]
  partially_fused_optimizer = get_hfta_optim_for(
      optim.Adam,
      B=B,
      partially_fused=True,
  )(
      net_fused.parameters(),
      net_fused.unfused_parameters(),
      lr=lr,
      betas=betas,
      eps=eps,
      weight_decay=weight_decay,
      amsgrad=amsgrad,
      B=B,
  )
  _optim_testing_procedure(net_fused, net_array, partially_fused_optimizer,
                           optimizer_array)


if __name__ == '__main__':
  testcase_automator(
      testcase_fused,
      {
          'B': [1, 5, 8],
          'lr': [
              [1e-3, 3e-3, 1e-2],
              (1e-3, 3e-3, 1e-2),
              np.array([1e-3, 3e-3, 1e-2]),
              torch.as_tensor([1e-3, 3e-3, 1e-2], dtype=torch.float),
              [1e-3, 1e-3, 1e-3],
              (1e-3, 1e-3, 1e-3),
              np.array([1e-3, 1e-3, 1e-3]),
              torch.as_tensor([1e-3, 1e-3, 1e-3], dtype=torch.float),
          ],
          'betas': [
              ([0.7, 0.8, 0.9], 0.999),
              ((0.7, 0.8, 0.9), 0.999),
              (np.array([0.7, 0.8, 0.9]), 0.999),
              (torch.as_tensor([0.7, 0.8, 0.9], dtype=torch.float), 0.999),
              ([0.9, 0.9, 0.9], 0.999),
              ((0.9, 0.9, 0.9), 0.999),
              (np.array([0.9, 0.9, 0.9]), 0.999),
              (torch.as_tensor([0.9, 0.9, 0.9], dtype=torch.float), 0.999),
              (0.9, [0.777, 0.888, 0.999]),
              (0.9, (0.777, 0.888, 0.999)),
              (0.9, np.array([0.777, 0.888, 0.999])),
              (0.9, torch.as_tensor([0.777, 0.888, 0.999], dtype=torch.float)),
              (0.9, [0.999, 0.999, 0.999]),
              (0.9, (0.999, 0.999, 0.999)),
              (0.9, np.array([0.999, 0.999, 0.999])),
              (0.9, torch.as_tensor([0.999, 0.999, 0.999], dtype=torch.float)),
              ([0.7, 0.8, 0.9], [0.777, 0.888, 0.999]),
              ((0.7, 0.8, 0.9), (0.777, 0.888, 0.999)),
              (np.array([0.7, 0.8, 0.9]), np.array([0.777, 0.888, 0.999])),
              (
                  torch.as_tensor([0.7, 0.8, 0.9], dtype=torch.float),
                  torch.as_tensor([0.777, 0.888, 0.999], dtype=torch.float),
              ),
              ([0.9, 0.9, 0.9], [0.999, 0.999, 0.999]),
              ((0.9, 0.9, 0.9), (0.999, 0.999, 0.999)),
              (np.array([0.9, 0.9, 0.9]), np.array([0.999, 0.999, 0.999])),
              (
                  torch.as_tensor([0.9, 0.9, 0.9], dtype=torch.float),
                  torch.as_tensor([0.999, 0.999, 0.999], dtype=torch.float),
              ),
          ],
          'eps': [
              [1e-7, 1e-8, 1e-9],
              (1e-7, 1e-8, 1e-9),
              np.array([1e-7, 1e-8, 1e-9]),
              torch.as_tensor([1e-7, 1e-8, 1e-9], dtype=torch.float),
              [1e-8, 1e-8, 1e-8],
              (1e-8, 1e-8, 1e-8),
              np.array([1e-8, 1e-8, 1e-8]),
              torch.as_tensor([1e-8, 1e-8, 1e-8], dtype=torch.float),
          ],
          'weight_decay': [
              [0.1, 0.03, 0.0],
              (0.1, 0.03, 0.0),
              np.array([0.1, 0.03, 0.0]),
              torch.as_tensor([0.1, 0.03, 0.0], dtype=torch.float),
              [0.0, 0.0, 0.0],
              (0, 0, 0),
              np.array([0, 0, 0]),
              torch.as_tensor([0.0, 0.0, 0.0], dtype=torch.float),
              0.3,
              0.0,
          ],
          'amsgrad': [True],
      },
  )
  testcase_automator(
      testcase_partially_fused,
      {
          'B': [1, 5, 8],
          'amsgrad': [True],
      },
  )
