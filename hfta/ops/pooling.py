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

  def __init__(self,
               kernel_size: _size_any_t,
               stride: Optional[_size_any_t] = None,
               padding: _size_any_t = 0,
               dilation: _size_any_t = 1,
               return_indices: bool = False,
               ceil_mode: bool = False,
               B: int = 1) -> None:
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
    res = F.max_pool2d(input.reshape(N * B, C, H, W), self.kernel_size,
                       self.stride, self.padding, self.dilation, self.ceil_mode,
                       self.return_indices)
    if self.return_indices:
      y, indices = res
    else:
      y = res
    NxB, C, H, W = y.size()
    y = y.reshape(N, B, C, H, W)
    return (y, indices.reshape(N, B, C, H, W)) if self.return_indices else y


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
    output = F.adaptive_avg_pool2d(input.reshape(N * B, C, H, W), self.output_size)
    NxB, C, H, W = output.size()
    return output.reshape(N, B, C, H, W)


class _AvgPoolNd(Module):
  __constants__ = [
      'kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'B'
  ]

  def extra_repr(self) -> str:
    return 'kernel_size={}, stride={}, padding={}, B={}'.format(
        self.kernel_size, self.stride, self.padding, self.B)


class AvgPool2d(_AvgPoolNd):
  r"""Applies a 2D average pooling over an input signal composed of several input
    planes.
    The additional argument: "B" is added to the constructor to support fused 
    inputs, Otherwise, the usage is identical to those described here:
        https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

    Shape:
        - Input: :math:`(N, B, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, B, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor
          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor
    Examples::
        >>> # pool of square window of size=3, stride=2, B=2
        >>> m = nn.AvgPool2d(3, stride=2, B=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool2d((3, 2), stride=(2, 1, B=2))
        >>> input = torch.randn(20, 2, 16, 50, 32)
        >>> output = m(input)
    """
  __constants__ = [
      'kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad',
      'divisor_override', 'B'
  ]

  kernel_size: _size_2_t
  stride: _size_2_t
  padding: _size_2_t
  ceil_mode: bool
  count_include_pad: bool

  def __init__(self,
               kernel_size: _size_2_t,
               stride: Optional[_size_2_t] = None,
               padding: _size_2_t = 0,
               ceil_mode: bool = False,
               count_include_pad: bool = True,
               divisor_override: Optional[int] = None,
               B: Optional[int] = 0) -> None:
    super(AvgPool2d, self).__init__()
    self.kernel_size = kernel_size
    self.stride = stride if (stride is not None) else kernel_size
    self.padding = padding
    self.ceil_mode = ceil_mode
    self.count_include_pad = count_include_pad
    self.divisor_override = divisor_override
    self.B = B

  def forward(self, input: Tensor) -> Tensor:
    N, B, C, H, W = input.size()
    input = input.reshape(N * B, C, H, W)
    out = F.avg_pool2d(input, self.kernel_size, self.stride, self.padding,
                       self.ceil_mode, self.count_include_pad,
                       self.divisor_override)
    _, _, H_out, W_out = out.size()
    return out.reshape(N, B, C, H_out, W_out)
