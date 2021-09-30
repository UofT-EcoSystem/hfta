import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter, init


class Linear(Module):
  """ Input format: [B * F]
  B: Training array size (number of batches/jobs).
  *: any number of dimensions, such as Batch size.
  F: in_features.
  """

  __constants__ = ['in_features', 'out_features', 'B']
  in_features: int
  out_features: int
  weight: Tensor
  B: int

  def __init__(self,
               in_features: int,
               out_features: int,
               bias: bool = True,
               device=None,
               dtype=None,
               B=1) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.B = B
    self.weight = Parameter(
        torch.empty((B, in_features, out_features), **factory_kwargs))
    if bias:
      self.bias = Parameter(torch.empty((B, 1, out_features), **factory_kwargs))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    for b in range(self.B):
      init.kaiming_uniform_(self.weight[b], a=math.sqrt(5), mode='fan_out')
      if self.bias is not None:
        _, fan_out = init._calculate_fan_in_and_fan_out(self.weight[b])
        bound = 1 / math.sqrt(fan_out) if fan_out > 0 else 0
        init.uniform_(self.bias[b], -bound, bound)

  def forward(self, input: Tensor) -> Tensor:
    old_shape = list(input.size())
    input = input.view(old_shape[0], -1, old_shape[-1])
    if self.bias is None:
      res = torch.bmm(input, self.weight)
    else:
      res = torch.baddbmm(self.bias, input, self.weight)
    old_shape[-1] = self.out_features
    return res.view(old_shape)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}, B={}'.format(
        self.in_features, self.out_features, self.bias is not None, self.B)

  def snatch_parameters(self, other, b):
    assert isinstance(other, nn.Linear)
    assert 0 <= b < self.B
    self.weight.data[b] = other.weight.data.transpose(0, 1)
    if self.bias is not None:
      self.bias.data[b] = other.bias.data.unsqueeze(0)
