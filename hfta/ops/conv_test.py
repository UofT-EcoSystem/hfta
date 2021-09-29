import numpy as np
import torch
import torch.nn as nn

from hfta.ops import (get_hfta_op_for, testcase_automator, support_dtype,
                      assert_allclose, dump_error_msg)


def testcase_Conv1d(
    B=3,
    N=32,
    Cin=4,
    Cout=16,
    kernel_size=3,
    Lin=28,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros',
    device=torch.device('cpu'),
    dtype=torch.float,
):
  if not support_dtype(device, dtype):
    return
  with torch.no_grad():
    x_array = [
        torch.rand(N, Cin, Lin, device=device, dtype=dtype) for _ in range(B)
    ]
    x_fused = torch.cat([x.unsqueeze(1) for x in x_array], dim=1)
    args = (Cin, Cout, kernel_size)
    kwargs = {
        'stride': stride,
        'padding': padding,
        'dilation': dilation,
        'groups': groups,
        'bias': bias,
        'padding_mode': padding_mode,
        'device': device,
        'dtype': dtype,
    }
    conv_array = [nn.Conv1d(*args, **kwargs) for _ in range(B)]
    conv_fused = get_hfta_op_for(nn.Conv1d, B=B)(*args, **kwargs)
    # Init weights and biases.
    for b in range(B):
      conv_fused.snatch_parameters(conv_array[b], b)
    y_array = [conv_array[b](x_array[b]) for b in range(B)]
    y_fused_actual = conv_fused(x_fused)
    y_fused_expect = torch.cat([y.unsqueeze(1) for y in y_array], dim=1)
    try:
      assert_allclose(
          y_fused_actual.cpu().numpy(),
          y_fused_expect.cpu().numpy(),
          rtol=1e-4,
          population_threshold=1e-2,
      )
    except AssertionError as e:
      dump_error_msg(e)


def testcase_Conv2d(
    B=3,
    N=32,
    Cin=4,
    Cout=16,
    kernel_size=3,
    HWin=28,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros',
    device=torch.device('cpu'),
    dtype=torch.float,
):
  if not support_dtype(device, dtype):
    return
  with torch.no_grad():
    x_array = [
        torch.rand(N, Cin, HWin, HWin, device=device, dtype=dtype)
        for _ in range(B)
    ]
    x_fused = torch.cat([x.unsqueeze(1) for x in x_array], dim=1)
    args = (Cin, Cout, kernel_size)
    kwargs = {
        'stride': stride,
        'padding': padding,
        'dilation': dilation,
        'groups': groups,
        'bias': bias,
        'padding_mode': padding_mode,
        'device': device,
        'dtype': dtype,
    }
    conv_array = [nn.Conv2d(*args, **kwargs) for _ in range(B)]
    conv_fused = get_hfta_op_for(nn.Conv2d, B=B)(*args, **kwargs)
    # Init weights and biases.
    for b in range(B):
      conv_fused.snatch_parameters(conv_array[b], b)
    y_array = [conv_array[b](x_array[b]) for b in range(B)]
    y_fused_actual = conv_fused(x_fused)
    y_fused_expect = torch.cat([y.unsqueeze(1) for y in y_array], dim=1)
    try:
      assert_allclose(
          y_fused_actual.cpu().numpy(),
          y_fused_expect.cpu().numpy(),
          rtol=1e-4,
          population_threshold=1e-2,
      )
    except AssertionError as e:
      dump_error_msg(e)


def testcase_ConvTranspose2d(
    B=3,
    N=32,
    Cin=4,
    Cout=16,
    kernel_size=3,
    HWin=28,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    bias=True,
    dilation=1,
    padding_mode='zeros',
    output_size=None,
    device=torch.device('cpu'),
    dtype=torch.float,
):
  if not support_dtype(device, dtype):
    return
  with torch.no_grad():
    x_array = [
        torch.rand(N, Cin, HWin, HWin, device=device, dtype=dtype)
        for _ in range(B)
    ]
    x_fused = torch.cat([x.unsqueeze(1) for x in x_array], dim=1)
    args = (Cin, Cout, kernel_size)

    # Handle output_padding
    if output_padding != 0:
      stride = output_padding + 1
      dilation = output_padding + 1

    # Handle output_size argument for the forward function
    if output_size:
      # The hardcoded input 57 and 58 are the possible size given stride == 2
      stride = 2
      output_size_arg = output_size if len(output_size) == 2 else (
          output_size[0:1] + output_size[2:])
    else:
      output_size_arg = None

    kwargs = {
        'stride': stride,
        'padding': padding,
        'output_padding': output_padding,
        'groups': groups,
        'bias': bias,
        'dilation': dilation,
        'padding_mode': padding_mode,
        'device': device,
        'dtype': dtype,
    }
    conv_array = [nn.ConvTranspose2d(*args, **kwargs) for _ in range(B)]
    conv_fused = get_hfta_op_for(nn.ConvTranspose2d, B=B)(*args, **kwargs)
    # Init weights and biases.
    for b in range(B):
      conv_fused.snatch_parameters(conv_array[b], b)

    y_array = [
        conv_array[b](x_array[b], output_size=output_size_arg) for b in range(B)
    ]
    y_fused_actual = conv_fused(x_fused, output_size=output_size)
    y_fused_expect = torch.cat([y.unsqueeze(1) for y in y_array], dim=1)
    try:
      assert_allclose(
          y_fused_actual.cpu().numpy(),
          y_fused_expect.cpu().numpy(),
          rtol=1e-4,
          population_threshold=1e-2,
      )
      if output_size:
        assert (
            y_fused_actual.shape == y_fused_expect.shape
        ), "The actual output size ({}) is different from the expected output size ({}).".format(
            y_fused_actual.shape, y_fused_expect.shape)
    except AssertionError as e:
      dump_error_msg(e)


if __name__ == '__main__':
  # Conv1d unit tests
  testcase_automator(
      testcase_Conv1d,
      {
          'B': [1, 2, 5, 10],
          'N': [1, 8, 64],
          'Cin': [3, 128],
          'Cout': [1, 64],
          'kernel_size': [1, 5, 7],
          'Lin': [32, 128],
          'stride': [2],
          'padding': [2],
          'dilation': [2],
          'groups': [2],
          'bias': [False],
          'padding_mode': ['reflect', 'replicate', 'circular'],
          'device': [
              torch.device('cpu'),
              torch.device('cuda:0'),
          ],
          'dtype': [torch.half, torch.float, torch.double],
      },
  )

  # Conv2d unit tests
  testcase_automator(
      testcase_Conv2d,
      {
          'B': [1, 2, 5, 10],
          'N': [1, 8, 64],
          'Cin': [3, 128],
          'Cout': [1, 64],
          'kernel_size': [1, 5, 7],
          'HWin': [32, 128],
          'stride': [2],
          'padding': [2],
          'dilation': [2],
          'groups': [2],
          'bias': [False],
          'padding_mode': ['reflect', 'replicate', 'circular'],
          'device': [
              torch.device('cpu'),
              torch.device('cuda:0'),
          ],
          'dtype': [torch.half, torch.float, torch.double],
      },
  )

  # ConvTranspose2d unit tests
  testcase_automator(
      testcase_ConvTranspose2d,
      {
          'B': [1, 2, 5, 10],
          'N': [1, 8, 64],
          'Cin': [3, 128],
          'Cout': [1, 64],
          'kernel_size': [1, 5, 7],
          'HWin': [32, 128],
          'stride': [2],
          'padding': [1, 2],
          'output_padding': [1, 2],
          'groups': [2],
          'bias': [False],
          'dilation': [2],
          'output_size': [(32, 3, 16, 57, 58), (57, 58)],
          'device': [
              torch.device('cpu'),
              torch.device('cuda:0'),
          ],
          'dtype': [torch.half, torch.float, torch.double],
      },
  )
