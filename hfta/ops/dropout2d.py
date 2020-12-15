import torch.nn as nn


class Dropout2d(nn.Module):
  """ Input format: [N B C H W]
  B: Training array size (number of batches/jobs).
  N: Batch size.
  C: Number of channels.
  H: Height.
  W: Width.
  """

  def __init__(self, p, inplace=False, B=1):
    super(Dropout2d, self).__init__()
    self.dropout = nn.Dropout2d(p, inplace)

  def forward(self, x):
    shape = list(x.size())
    new_shape = [shape[0] * shape[1]] + shape[2:]
    y = self.dropout(x.view(new_shape))
    return y.view(shape)
