import torch.nn as nn
import torch
import math
from hfta.ops import get_hfta_op_for


class GELU(nn.Module):
  """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
  "Implements FFN equation."

  def __init__(self, d_model, d_ff, dropout=0.1, B=1):
    super(PositionwiseFeedForward, self).__init__()
    Linear = get_hfta_op_for(nn.Linear, B)
    self.w_1 = Linear(d_model, d_ff)
    self.w_2 = Linear(d_ff, d_model)
    self.dropout = get_hfta_op_for(nn.Dropout, B)(dropout)
    self.activation = GELU()

  def forward(self, x):
    return self.w_2(self.dropout(self.activation(self.w_1(x))))


class SublayerConnection(nn.Module):
  """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

  def __init__(self, size, dropout, B=1):
    super(SublayerConnection, self).__init__()
    self.norm = get_hfta_op_for(nn.LayerNorm, B)(size)
    self.dropout = get_hfta_op_for(nn.Dropout, B)(dropout)

  def forward(self, x, sublayer):
    "Apply residual connection to any sublayer with the same size."
    _x = self.norm(x)
    _x = sublayer(_x)
    _x = self.dropout(_x)
    return x + _x
