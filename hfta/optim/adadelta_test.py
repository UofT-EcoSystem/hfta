import numpy as np
import torch
import torch.optim as optim

import sys
sys.path.append('../')
from ops import testcase_automator
from optim import get_hfta_optim_for, index_array_or_return_scalar
from utils import _TestNet, _optim_testing_procedure


def testcase(B=3, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
  net_array = [_TestNet() for _ in range(B)]
  net_fused = _TestNet(B=B)
  optimizer_array = [
      optim.Adadelta(
          net_array[b].parameters(),
          lr=index_array_or_return_scalar(lr, b),
          rho=index_array_or_return_scalar(rho, b),
          eps=index_array_or_return_scalar(eps, b),
          weight_decay=index_array_or_return_scalar(weight_decay, b),
      ) for b in range(B)
  ]
  optimizer_fused = get_hfta_optim_for(optim.Adadelta, B=B)(
      net_fused.parameters(),
      lr=lr,
      rho=rho,
      eps=eps,
      weight_decay=weight_decay,
  )
  _optim_testing_procedure(net_fused, net_array, optimizer_fused,
                           optimizer_array)


if __name__ == '__main__':
  testcase_automator(
      testcase,
      {
          'B': [1, 5, 8],
          'lr': [
              [0.5, 1.0, 2.0],
              (0.5, 1.0, 2.0),
              np.array([0.5, 1.0, 2.0]),
              torch.as_tensor([0.5, 1.0, 2.0], dtype=torch.float),
              [1.0, 1.0, 1.0],
              (1.0, 1.0, 1.0),
              np.array([1.0, 1.0, 1.0]),
              torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float),
          ],
          'rho': [
              [0.1, 0.9, 0.99],
              (0.1, 0.9, 0.99),
              np.array([0.1, 0.9, 0.99]),
              torch.as_tensor([0.1, 0.9, 0.99], dtype=torch.float),
              [0.9, 0.9, 0.9],
              (0.9, 0.9, 0.9),
              np.array([0.9, 0.9, 0.9]),
              torch.as_tensor([0.9, 0.9, 0.9], dtype=torch.float),
          ],
          'eps': [
              [1e-6, 1e-5, 1e-7],
              (1e-6, 1e-5, 1e-7),
              np.array([1e-6, 1e-5, 1e-7]),
              torch.as_tensor([1e-6, 1e-5, 1e-7], dtype=torch.float),
              [1e-6, 1e-6, 1e-6],
              (1e-6, 1e-6, 1e-6),
              np.array([1e-6, 1e-6, 1e-6]),
              torch.as_tensor([1e-6, 1e-6, 1e-6], dtype=torch.float),
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
      },
  )
