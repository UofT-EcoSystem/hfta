import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaxPool2d(nn.Module):
  """ Input format: [N B C H W]
  B: Training array size (number of batches/jobs).
  N: Batch size.
  C: Number of channels.
  H: Height.
  W: Width.
  """

  def __init__(
      self,
      kernel_size,
      stride=None,
      padding=0,
      dilation=1,
      return_indices=False,
      ceil_mode=False,
      B=1,
  ):
    super(MaxPool2d, self).__init__()
    self.pool = nn.MaxPool2d(
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices,
        ceil_mode=ceil_mode,
    )

  def forward(self, x):
    N, B, C, H, W = x.size()
    res = self.pool(x.view(N * B, C, H, W))
    if self.pool.return_indices:
      y, indices = res
    else:
      y = res
    NxB, C, H, W = y.size()
    y = y.view(N, B, C, H, W)
    return (y, indices.view(N, B, C, H, W)) if self.pool.return_indices else y


class _AdaptiveAvgPoolNd(nn.Module):
  __constants__ = ['output_size']

  def __init__(self, output_size, B=1):
    super(_AdaptiveAvgPoolNd, self).__init__()
    self.output_size = output_size
    self.B = B

  def extra_repr(self) -> str:
    return 'output_size={}, B={}'.format(self.output_size, self.B)


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
  """ Input format: [N B C H W]
  B: Training array size (number of batches/jobs).
  N: Batch size.
  C: Number of channels.
  H: Height.
  W: Width.
  """

  def forward(self, input: Tensor) -> Tensor:
    N, B, C, H, W = input.size()
    output = F.adaptive_avg_pool2d(input.view(N * B, C, H, W), self.output_size)
    NxB, C, H, W = output.size()
    return output.view(N, B, C, H, W)
