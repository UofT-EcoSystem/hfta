import torch.nn as nn

from torch import Tensor


class Dropout2d(nn.Module):
  """ Input format: [N B C H W]
  B: Training array size (number of batches/jobs).
  N: Batch size.
  C: Number of channels.
  H: Height.
  W: Width.
  """
  __constants__ = ['p', 'inplace', 'B']
  p: float
  inplace: bool
  B: int

  def __init__(
      self,
      p: float = 0.5,
      inplace: bool = False,
      B: int = 1,
  ) -> None:
    super(Dropout2d, self).__init__()
    self.B = B  # Not used
    self.dropout = nn.Dropout2d(p, inplace)

  def extra_repr(self) -> str:
    return '{}, B={}'.format(self.dropout.extra_repr(), self.B)

  def forward(self, input: Tensor) -> Tensor:
    shape = list(input.size())
    new_shape = [shape[0] * shape[1]] + shape[2:]
    y = self.dropout(input.view(new_shape))
    return y.view(shape)
