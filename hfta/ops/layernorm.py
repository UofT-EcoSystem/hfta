import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, Size
from torch.nn import Module, Parameter, init
from typing import Union, List, Tuple

_shape_t = Union[int, List[int], Size]


class LayerNorm(Module):
  """ Input format: [B *  *normalized_shape]
    B: Training array size (number of batches/jobs).
    *: any number of dimension, at least 1 dimension, such as Batch size.
    *normalized_shape: shoud be same as normalized_shape.
  """
  __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', 'B']
  normalized_shape: Tuple[int, ...]
  eps: float
  elementwise_affine: bool
  B: int

  def __init__(
      self,
      normalized_shape: _shape_t,
      eps: float = 1e-5,
      elementwise_affine: bool = True,
      device=None,
      dtype=None,
      B=1,
  ) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(LayerNorm, self).__init__()
    if isinstance(normalized_shape, numbers.Integral):
      # mypy error: incompatible types in assignment
      normalized_shape = (normalized_shape,)  # type: ignore[assignment]
    self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
    self.B = B
    self.eps = eps
    self.elementwise_affine = elementwise_affine
    if self.elementwise_affine:
      self.weight = Parameter(
          torch.empty((B, *normalized_shape), **factory_kwargs))
      self.bias = Parameter(
          torch.empty((B, *normalized_shape), **factory_kwargs))
    else:
      self.register_parameter('weight', None)
      self.register_parameter('bias', None)

    self.reset_parameters()

  def reset_parameters(self) -> None:
    if self.elementwise_affine:
      init.ones_(self.weight)
      init.zeros_(self.bias)

  def forward(self, input: Tensor) -> Tensor:
    res = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
    if self.elementwise_affine:
      inside_length = input.dim() - len(self.normalized_shape) - 1
      w_shape = [self.B] + ([1] * inside_length) + list(self.normalized_shape)
      bias = self.bias.view(w_shape)
      weight = self.weight.view(w_shape)
      res = torch.addcmul(bias, res, weight)
    return res

  def extra_repr(self) -> str:
    return '{normalized_shape}, eps={eps}, B={B}, ' \
        'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

  def snatch_parameters(self, other, b):
    assert isinstance(other, nn.LayerNorm)
    assert 0 <= b < self.B
    if self.elementwise_affine:
      self.weight.data[b] = other.weight.data
      self.bias.data[b] = other.bias.data
