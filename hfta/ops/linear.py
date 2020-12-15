import math

import torch
import torch.nn as nn


class Linear(nn.Module):
  """ Input format: [B * F]
  B: Training array size (number of batches/jobs).
  *: any number of dimensions, such as Batch size.
  F: in_features.
  """

  def __init__(self, in_features, out_features, bias=True, B=1):
    super(Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.B = B
    self.weight = nn.Parameter(torch.Tensor(B, in_features, out_features))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(B, 1, out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    for b in range(self.B):
      nn.init.kaiming_uniform_(self.weight[b], a=math.sqrt(5), mode='fan_out')
      if self.bias is not None:
        _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight[b])
        bound = 1 / math.sqrt(fan_out)
        nn.init.uniform_(self.bias[b], -bound, bound)

  def forward(self, x):
    old_shape = list(x.size())
    x = x.view(old_shape[0], -1, old_shape[-1])
    if self.bias is None:
      res = torch.bmm(x, self.weight)
    else:
      res = torch.baddbmm(self.bias, x, self.weight)
    old_shape[-1] = self.out_features
    return res.view(old_shape)

  def snatch_parameters(self, other, b):
    assert isinstance(other, nn.Linear)
    assert 0 <= b < self.B
    self.weight.data[b] = other.weight.data.transpose(0, 1)
    if self.bias is not None:
      self.bias.data[b] = other.bias.data.unsqueeze(0)
