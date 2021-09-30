import numpy as np
import torch
import torch.nn as nn

from hfta.ops import (get_hfta_op_for, testcase_automator, assert_allclose,
                      dump_error_msg)


def testcase_MaxPool2d(
    B=3,
    N=32,
    C=16,
    kernel_size=2,
    HWin=28,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False,
    device=torch.device('cpu'),
    dtype=torch.float,
):
  with torch.no_grad():
    x_array = [
        torch.rand(N, C, HWin, HWin, device=device, dtype=dtype)
        for _ in range(B)
    ]
    x_fused = torch.cat([x.unsqueeze(1) for x in x_array], dim=1)
    args = (kernel_size,)
    kwargs = {
        'stride': stride,
        'padding': padding,
        'dilation': dilation,
        'return_indices': return_indices,
        'ceil_mode': ceil_mode,
    }
    pool_array = [nn.MaxPool2d(*args, **kwargs) for _ in range(B)]
    pool_fused = get_hfta_op_for(nn.MaxPool2d, B=B)(*args, **kwargs)
    res_array = [pool_array[b](x_array[b]) for b in range(B)]
    res_fused_actual = pool_fused(x_fused)
    if return_indices:
      y_array, indices_array = tuple(zip(*res_array))
      y_fused_actual, indices_fused_actual = res_fused_actual
    else:
      y_array = res_array
      y_fused_actual = res_fused_actual
    y_fused_expect = torch.cat([y.unsqueeze(1) for y in y_array], dim=1)
    try:
      assert_allclose(
          y_fused_actual.cpu().numpy(),
          y_fused_expect.cpu().numpy(),
          rtol=1e-4,
      )
    except AssertionError as e:
      dump_error_msg(e)
    if return_indices:
      indices_fused_expect = torch.cat(
          [indices.unsqueeze(1) for indices in indices_array],
          dim=1,
      )
      try:
        assert_allclose(
            indices_fused_actual.cpu().numpy(),
            indices_fused_expect.cpu().numpy(),
            rtol=1e-4,
        )
      except AssertionError as e:
        dump_error_msg(e)


def testcase_AdaptiveAvgPool2d(
    B=3,
    N=32,
    C=16,
    HWin=28,
    output_size=(16, 16),
    device=torch.device('cpu'),
    dtype=torch.float,
):
  with torch.no_grad():
    x_array = [
        torch.rand(N, C, HWin, HWin, device=device, dtype=dtype)
        for _ in range(B)
    ]
    x_fused = torch.cat([x.unsqueeze(1) for x in x_array], dim=1)
    args = (output_size,)
    pool_array = [nn.AdaptiveAvgPool2d(*args) for _ in range(B)]
    pool_fused = get_hfta_op_for(nn.AdaptiveAvgPool2d, B=B)(*args)
    y_array = [pool_array[b](x_array[b]) for b in range(B)]
    y_fused_actual = pool_fused(x_fused)
    y_fused_expect = torch.cat([y.unsqueeze(1) for y in y_array], dim=1)
    try:
      assert_allclose(
          y_fused_actual.cpu().numpy(),
          y_fused_expect.cpu().numpy(),
          rtol=1e-4,
      )
    except AssertionError as e:
      dump_error_msg(e)


if __name__ == '__main__':
  testcase_automator(
      testcase_MaxPool2d,
      {
          'B': [1, 2, 5, 10],
          'N': [1, 8, 64],
          'C': [3, 64, 128],
          'kernel_size': [1, 3, 4],
          'HWin': [32, 256],
          'stride': [1, 3, 4],
          'padding': [1],
          'dilation': [2, 5],
          'return_indices': [True],
          'ceil_mode': [True],
          'device': [torch.device('cuda:0')],
          'dtype': [torch.double],
      },
  )
  testcase_automator(
      testcase_AdaptiveAvgPool2d,
      {
          'B': [1, 2, 5, 10],
          'N': [1, 8, 64],
          'C': [3, 64, 128],
          'HWin': [32, 256],
          'output_size': [
              (5, 5),
              (7, 16),
              (17, 6),
              (14, None),
              (None, 20),
              (None, None),
              9,
          ],
          'device': [torch.device('cuda:0')],
          'dtype': [torch.double],
      },
  )
