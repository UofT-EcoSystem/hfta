import torch.nn as nn

from torch import Tensor


class Dropout2d(nn.Dropout2d):
  """ Input format: [N B C H W]
  B: Training array size (number of batches/jobs).
  N: Batch size.
  C: Number of channels.
  H: Height.
  W: Width.
  """

  def __init__(
      self,
      p: float = 0.5,
      inplace: bool = False,
      B: int = 1,
  ) -> None:
    super(Dropout2d, self).__init__(p, inplace)
    self.B = B  # Not used

  def extra_repr(self) -> str:
    return '{}, B={}'.format(super(Dropout2d, self).extra_repr(), self.B)

  def forward(self, input: Tensor) -> Tensor:
    shape = list(input.size())
    new_shape = [shape[0] * shape[1]] + shape[2:]
    y = super(Dropout2d, self).forward(input.view(new_shape))
    return y.view(shape)
