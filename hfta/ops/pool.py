import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.common_types import (_size_any_t, _size_any_opt_t, _size_2_t,
                                   _size_2_opt_t)
from typing import Optional


class _MaxPoolNd(Module):
  __constants__ = [
      'kernel_size', 'stride', 'padding', 'dilation', 'return_indices',
      'ceil_mode', 'B'
  ]
  return_indices: bool
  ceil_mode: bool
  B: int

  def __init__(
      self,
      kernel_size: _size_any_t,
      stride: Optional[_size_any_t] = None,
      padding: _size_any_t = 0,
      dilation: _size_any_t = 1,
      return_indices: bool = False,
      ceil_mode: bool = False,
      B: int = 1,
  ) -> None:
    super(_MaxPoolNd, self).__init__()
    self.kernel_size = kernel_size
    self.stride = stride if (stride is not None) else kernel_size
    self.padding = padding
    self.dilation = dilation
    self.return_indices = return_indices
    self.ceil_mode = ceil_mode
    self.B = B

  def extra_repr(self) -> str:
    return ('kernel_size={kernel_size}, stride={stride}, padding={padding} '
            ', dilation={dilation}, ceil_mode={ceil_mode}, B={B}'.format(
                **self.__dict__))


class MaxPool2d(_MaxPoolNd):
  r""" Input format: [N B C H W]
  B: Training array size (number of batches/jobs).
  N: Batch size.
  C: Number of channels.
  H: Height.
  W: Width.
  """

  kernel_size: _size_2_t
  stride: _size_2_t
  padding: _size_2_t
  dilation: _size_2_t

  def forward(self, input: Tensor) -> Tensor:
    N, B, C, H, W = input.size()
    res = F.max_pool2d(input.view(N * B, C, H, W), self.kernel_size,
                       self.stride, self.padding, self.dilation, self.ceil_mode,
                       self.return_indices)
    if self.return_indices:
      y, indices = res
    else:
      y = res
    NxB, C, H, W = y.size()
    y = y.view(N, B, C, H, W)
    return (y, indices.view(N, B, C, H, W)) if self.return_indices else y


class _AdaptiveAvgPoolNd(Module):
  __constants__ = ['output_size', 'B']

  def __init__(self, output_size: _size_any_opt_t, B: int = 1) -> None:
    super(_AdaptiveAvgPoolNd, self).__init__()
    self.output_size = output_size
    self.B = B

  def extra_repr(self) -> str:
    return 'output_size={}, B={}'.format(self.output_size, self.B)


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
  r""" Input format: [N B C H W]
  B: Training array size (number of batches/jobs).
  N: Batch size.
  C: Number of channels.
  H: Height.
  W: Width.
  """

  output_size: _size_2_opt_t

  def forward(self, input: Tensor) -> Tensor:
    N, B, C, H, W = input.size()
    output = F.adaptive_avg_pool2d(input.view(N * B, C, H, W), self.output_size)
    NxB, C, H, W = output.size()
    return output.view(N, B, C, H, W)
