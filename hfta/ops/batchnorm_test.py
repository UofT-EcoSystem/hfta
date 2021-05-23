import numpy as np
import torch
import torch.nn as nn
import random
from hfta.ops import get_hfta_op_for, testcase_automator


def testcase_1d(
    num_features=128,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    track_running_stats=True,
    B=3,
    N=8,
    L=16,
    train_test_steps=10,
    training=True,
):
  C = num_features
  with torch.no_grad():
    args = (num_features,)
    kwargs = {
        'eps': eps,
        'momentum': momentum,
        'affine': affine,
        'track_running_stats': track_running_stats,
    }
    batchNormal1d_array = [nn.BatchNorm1d(*args, **kwargs) for _ in range(B)]
    if track_running_stats:
      rand_int = random.randint(0, 1024)
      for bn in batchNormal1d_array:
        nn.init.normal_(bn.running_mean)
        nn.init.normal_(bn.running_var)
        bn.num_batches_tracked.fill_(rand_int)
    batchNormal1d_fused = get_hfta_op_for(nn.BatchNorm1d, B=B)(*args, **kwargs)
    # Init weights and biases.
    for b in range(B):
      batchNormal1d_fused.snatch_parameters(batchNormal1d_array[b], b)

    if training:
      [bn.train() for bn in batchNormal1d_array]
      batchNormal1d_fused.train()
    else:
      [bn.eval() for bn in batchNormal1d_array]
      batchNormal1d_fused.eval()

    # check whether fused outputs are same in several training steps
    for i in range(train_test_steps):
      if L == 0:
        x_array = [torch.rand(N, C) for _ in range(B)]
        x_fused = torch.cat([x.unsqueeze(0) for x in x_array], dim=0)
      else:
        x_array = [torch.rand(N, C, L) for _ in range(B)]
        x_fused = torch.cat([x.unsqueeze(1) for x in x_array], dim=1)

      y_array = [batchNormal1d_array[b](x_array[b]) for b in range(B)]
      y_fused_actual = batchNormal1d_fused(x_fused)
      if L == 0:
        y_fused_expect = torch.cat([y.unsqueeze(0) for y in y_array], dim=0)
      else:
        y_fused_expect = torch.cat([y.unsqueeze(1) for y in y_array], dim=1)
      try:
        np.testing.assert_allclose(
            y_fused_actual.numpy(),
            y_fused_expect.numpy(),
            rtol=1e-4,
        )
      except AssertionError as e:
        print(e)


def testcase_2d(
    num_features=128,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    track_running_stats=True,
    B=3,
    N=8,
    HWin=28,
    train_test_steps=10,
    training=True,
):
  C = num_features
  with torch.no_grad():

    args = (num_features,)
    kwargs = {
        'eps': eps,
        'momentum': momentum,
        'affine': affine,
        'track_running_stats': track_running_stats,
    }
    batchNormal2d_array = [nn.BatchNorm2d(*args, **kwargs) for _ in range(B)]
    batchNormal2d_fused = get_hfta_op_for(nn.BatchNorm2d, B=B)(*args, **kwargs)
    if track_running_stats:
      rand_int = random.randint(0, 1024)
      for bn in batchNormal2d_array:
        nn.init.normal_(bn.running_mean)
        nn.init.normal_(bn.running_var)
        bn.num_batches_tracked.fill_(rand_int)
    # Init weights and biases.
    for b in range(B):
      batchNormal2d_fused.snatch_parameters(batchNormal2d_array[b], b)

    if training:
      [bn.train() for bn in batchNormal2d_array]
      batchNormal2d_fused.train()
    else:
      [bn.eval() for bn in batchNormal2d_array]
      batchNormal2d_fused.eval()

    # check whether fused outputs are same in several training steps
    for i in range(train_test_steps):
      x_array = [torch.rand(N, C, HWin, HWin) for _ in range(B)]
      x_fused = torch.cat([x.unsqueeze(1) for x in x_array], dim=1)

      y_array = [batchNormal2d_array[b](x_array[b]) for b in range(B)]
      y_fused_actual = batchNormal2d_fused(x_fused)
      y_fused_expect = torch.cat([y.unsqueeze(1) for y in y_array], dim=1)
      try:
        np.testing.assert_allclose(
            y_fused_actual.numpy(),
            y_fused_expect.numpy(),
            rtol=1e-4,
        )
      except AssertionError as e:
        print(e)


if __name__ == '__main__':
  testcase_automator(
      testcase_1d,
      {
          'num_features': [1, 16, 128],
          'B': [1, 2, 5, 10],
          'N': [16, 64],
          'L': [0, 1, 8, 64],
          'momentum': [0.01],
          'affine': [True, False],
          'track_running_stats': [True, False],
          'training': [True, False],
      },
  )
  testcase_automator(
      testcase_2d,
      {
          'num_features': [1, 16, 128],
          'B': [1, 2, 5, 10],
          'N': [16, 64],
          'HWin': [32, 128],
          'momentum': [0.01],
          'affine': [True, False],
          'track_running_stats': [True, False],
          'training': [True, False],
      },
  )
