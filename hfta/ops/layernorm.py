import torch
import numbers
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
  """ Input format: [B *  *normalized_shape]
    B: Training array size (number of batches/jobs).
    *: any number of dimension, at least 1 dimension, such as Batch size.
    *normalized_shape: shoud be same as normalized_shape.
  """

  def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, B=1):
    super(LayerNorm, self).__init__()
    if isinstance(normalized_shape, numbers.Integral):
      normalized_shape = (normalized_shape,)
    self.normalized_shape = tuple(normalized_shape)
    self.B = B
    self.eps = eps
    self.elementwise_affine = elementwise_affine
    if self.elementwise_affine:
      self.weight = nn.Parameter(torch.Tensor(B, *normalized_shape))
      self.bias = nn.Parameter(torch.Tensor(B, *normalized_shape))
    else:
      self.register_parameter('weight', None)
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    if self.elementwise_affine:
      nn.init.ones_(self.weight)
      nn.init.zeros_(self.bias)

  def forward(self, input):
    res = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
    if self.elementwise_affine:
      inside_length = input.dim() - len(self.normalized_shape) - 1
      w_shape = [self.B] + ([1] * inside_length) + list(self.normalized_shape)
      bias = self.bias.view(w_shape)
      weight = self.weight.view(w_shape)
      res = torch.addcmul(bias, res, weight)
    return res

  def extra_repr(self):
    return '{normalized_shape}, eps={eps}, B={B}, ' \
        'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

  def snatch_parameters(self, other, b):
    assert isinstance(other, nn.LayerNorm)
    assert 0 <= b < self.B
    if self.elementwise_affine:
      self.weight.data[b] = other.weight.data
      self.bias.data[b] = other.bias.data
