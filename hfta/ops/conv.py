import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import _single, _pair, _reverse_repeat_tuple

from typing import Optional, List


class _ConvNd(nn.Module):

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      bias,
      padding_mode,
      B,
  ):
    super(_ConvNd, self).__init__()
    if in_channels % groups != 0:
      raise ValueError('in_channels must be divisible by groups')
    if out_channels % groups != 0:
      raise ValueError('out_channels must be divisible by groups')
    valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
    if padding_mode not in valid_padding_modes:
      raise ValueError("padding_mode must be one of {}, but got "
                       "padding_mode='{}'".format(valid_padding_modes,
                                                  padding_mode))
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.transposed = transposed
    self.output_padding = output_padding
    self.groups = groups
    self.padding_mode = padding_mode
    self.B = B
    # `_reversed_padding_repeated_twice` is the padding to be passed to
    # `F.pad` if needed (e.g., for non-zero padding types that are
    # implemented as two ops: padding + conv). `F.pad` accepts paddings in
    # reverse order than the dimension.
    self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
        self.padding,
        2,
    )
    if transposed:
      self.weight = nn.Parameter(
          torch.Tensor(B, in_channels, out_channels // groups, *kernel_size))
    else:
      self.weight = nn.Parameter(
          torch.Tensor(B, out_channels, in_channels // groups, *kernel_size))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(B, out_channels))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    for b in range(self.B):
      nn.init.kaiming_uniform_(self.weight[b], a=math.sqrt(5))
      if self.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[b])
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias[b], -bound, bound)

  def extra_repr(self):
    s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
         ', stride={stride}, B={B}')
    if self.padding != (0,) * len(self.padding):
      s += ', padding={padding}'
    if self.dilation != (1,) * len(self.dilation):
      s += ', dilation={dilation}'
    if self.output_padding != (0,) * len(self.output_padding):
      s += ', output_padding={output_padding}'
    if self.groups != 1:
      s += ', groups={groups}'
    if self.bias is None:
      s += ', bias=False'
    if self.padding_mode != 'zeros':
      s += ', padding_mode={padding_mode}'
    return s.format(**self.__dict__)

  def __setstate__(self, state):
    super(_ConvNd, self).__setstate__(state)
    if not hasattr(self, 'padding_mode'):
      self.padding_mode = 'zeros'

  def snatch_parameters(self, other, b):
    assert 0 <= b < self.B
    self.weight.data[b] = other.weight.data
    if self.bias is not None:
      self.bias.data[b] = other.bias.data


class Conv1d(_ConvNd):
  r""" Based on PyTorch(1.6.0)'s Conv1d source code:
  https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d


  Input format: [N, B, Cin, Lin]
  B: Training array size (number of batches/jobs).
  N: Batch size.
  C: Number of channels.
  L: Length of signal sequence.

  Weight format: [B, Cout, Cin/groups, kernel_size]
  Bias format: [B, Cout]

  Output format: [N, B, Cout, Lout]
  """

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride=1,
      padding=0,
      dilation=1,
      groups=1,
      bias=True,
      padding_mode='zeros',  # TODO: refine this type
      B=1,
  ):
    kernel_size = _single(kernel_size)
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)
    super(Conv1d, self).__init__(in_channels, out_channels,
                                 kernel_size, stride, padding, dilation, False,
                                 _single(0), groups, bias, padding_mode, B)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    Lin = input.size(3)
    input = input.view(-1, self.B * self.in_channels, Lin)
    weight = self.weight.view(self.B * self.out_channels,
                              self.in_channels // self.groups,
                              *self.kernel_size)
    bias = (self.bias.view(self.B * self.out_channels)
            if self.bias is not None else self.bias)

    if self.padding_mode != 'zeros':
      y = F.conv1d(
          F.pad(input,
                self._reversed_padding_repeated_twice,
                mode=self.padding_mode), weight, bias, self.stride, _single(0),
          self.dilation, self.groups * self.B)
    else:
      y = F.conv1d(input, weight, bias, self.stride, self.padding,
                   self.dilation, self.groups * self.B)
    Lout = y.size(2)
    return y.view(-1, self.B, self.out_channels, Lout)

  def snatch_parameters(self, other, b):
    assert isinstance(other, nn.Conv1d)
    super(Conv1d, self).snatch_parameters(other, b)


class Conv2d(_ConvNd):
  r""" Based on PyTorch(1.6.0)'s Conv2d source code:
  https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d

  Input format: [N B C H W]
  B: Training array size (number of batches/jobs).
  N: Batch size.
  C: Number of channels.
  H: Height.
  W: Width.

  Weight format: [B, Cout, Cin/groups, H, W]
  Bias format: [B, Cout]
  """

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride=1,
      padding=0,
      dilation=1,
      groups=1,
      bias=True,
      padding_mode='zeros',  # TODO: refine this type
      B=1,
  ):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    super(Conv2d, self).__init__(in_channels, out_channels,
                                 kernel_size, stride, padding, dilation, False,
                                 _pair(0), groups, bias, padding_mode, B)

  def _conv_forward(self, input, weight):
    Hin, Win = input.size(3), input.size(4)
    input = input.view(-1, self.B * self.in_channels, Hin, Win)
    weight = weight.view(self.B * self.out_channels,
                         self.in_channels // self.groups, *self.kernel_size)
    bias = (self.bias.view(self.B * self.out_channels)
            if self.bias is not None else self.bias)

    if self.padding_mode != 'zeros':
      y = F.conv2d(
          F.pad(input,
                self._reversed_padding_repeated_twice,
                mode=self.padding_mode), weight, bias, self.stride, _pair(0),
          self.dilation, self.groups * self.B)
    else:
      y = F.conv2d(input, weight, bias, self.stride, self.padding,
                   self.dilation, self.groups * self.B)
    Hout, Wout = y.size(2), y.size(3)
    return y.view(-1, self.B, self.out_channels, Hout, Wout)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    return self._conv_forward(input, self.weight)

  def snatch_parameters(self, other, b):
    assert isinstance(other, nn.Conv2d)
    super(Conv2d, self).snatch_parameters(other, b)


class _ConvTransposeNd(_ConvNd):

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      bias,
      padding_mode,
      B,
  ):
    if padding_mode != 'zeros':
      raise ValueError('Only "zeros" padding mode is supported for {}'.format(
          self.__class__.__name__))

    super(_ConvTransposeNd, self).__init__(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        B,
    )

  def _output_padding(self, input, output_size, stride, padding, kernel_size):
    # type: (Tensor, Optional[List[int]], List[int], List[int], List[int]) -> List[int]
    # Input format: [N, B, Cin, ...]
    if output_size is None:
      # converting to list if was not already
      ret = _single(self.output_padding)
    else:
      # Given the input format as [N, B, Cin, ...], we need to exclude the first 3 items
      # This is modified from the original code
      k = input.dim() - 3
      if len(output_size) == k + 3:
        output_size = output_size[3:]
      if len(output_size) != k:
        # We are checking to ensure the output_size shows either the dimensions of conv operator (1d, 2d, etc.)
        # or the whole shape of the expected output (with B), e.g. 2d + 3 = 5
        raise ValueError(
            "output_size must have {} or {} elements (got {})".format(
                k, k + 3, len(output_size)))

      min_sizes = torch.jit.annotate(List[int], [])
      max_sizes = torch.jit.annotate(List[int], [])
      for d in range(k):
        # This is modified for HFTA
        dim_size = ((input.size(d + 3) - 1) * stride[d] - 2 * padding[d] +
                    kernel_size[d])
        min_sizes.append(dim_size)
        max_sizes.append(min_sizes[d] + stride[d] - 1)

      for i in range(len(output_size)):
        size = output_size[i]
        min_size = min_sizes[i]
        max_size = max_sizes[i]
        if size < min_size or size > max_size:
          raise ValueError(
              ("requested an output size of {}, but valid sizes range "
               "from {} to {} (for an input of {})").format(
                   output_size, min_sizes, max_sizes,
                   input.size()[3:]))

      res = torch.jit.annotate(List[int], [])
      for d in range(k):
        res.append(output_size[d] - min_sizes[d])

      ret = res
    return ret


class ConvTranspose2d(_ConvTransposeNd):
  r""" Applies a 2D transposed convolution operator over an input image
  composed of several input planes.

  Based on PyTorch(1.6.0)'s ConvTranspose2d source code:
  https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#ConvTranspose2d

  Input format: [N, B, Cin, Hin, Win]
  B: Training array size (number of batches/jobs).
  N: Batch size.
  C: Number of channels.
  H: Height.
  W: Width.

  Output format: [N, B, Cout, Hout, Wout]

  Weight format: [B, Cin, Cout/groups, kernel_size[0], kernel_size[1]]
  Bias format: [B, Cout]
  """

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride=1,
      padding=0,
      output_padding=0,
      groups=1,
      bias=True,
      dilation=1,
      padding_mode='zeros',
      B=1,
  ):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    output_padding = _pair(output_padding)
    super(ConvTranspose2d, self).__init__(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
        bias,
        padding_mode,
        B,
    )

  def forward(self,
              input: torch.Tensor,
              output_size: Optional[List[int]] = None) -> torch.Tensor:

    if self.padding_mode != 'zeros':
      raise ValueError(
          'Only `zeros` padding mode is supported for ConvTranspose2d')

    output_padding = self._output_padding(input, output_size, self.stride,
                                          self.padding, self.kernel_size)

    Hin, Win = input.size(3), input.size(4)
    input = input.view(-1, self.B * self.in_channels, Hin, Win)
    weight = self.weight.view(self.B * self.in_channels,
                              self.out_channels // self.groups,
                              *self.kernel_size)
    bias = (self.bias.view(self.B * self.out_channels)
            if self.bias is not None else self.bias)

    y = F.conv_transpose2d(input, weight, bias, self.stride, self.padding,
                           output_padding, self.groups * self.B, self.dilation)

    Hout, Wout = y.size(2), y.size(3)
    return y.view(-1, self.B, self.out_channels, Hout, Wout)

  def snatch_parameters(self, other, b):
    assert isinstance(other, nn.ConvTranspose2d)
    super(ConvTranspose2d, self).snatch_parameters(other, b)
