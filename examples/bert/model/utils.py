# Copyright (c) 2020-     UofT-EcoSystem,
# Copyright 2018 - 2019 Junseong Kim, Scatter Lab, respective BERT contributors
# Copyright (c) 2018 Alexander Rush : The Annotated Trasnformer
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch.nn as nn
from hfta.ops import get_hfta_op_for


class PositionwiseFeedForward(nn.Module):
  "Implements FFN equation."

  def __init__(self, d_model, d_ff, dropout=0.1, B=1):
    super(PositionwiseFeedForward, self).__init__()
    Linear = get_hfta_op_for(nn.Linear, B)
    self.w_1 = Linear(d_model, d_ff)
    self.w_2 = Linear(d_ff, d_model)
    self.dropout = get_hfta_op_for(nn.Dropout, B)(dropout)

  def forward(self, x):
    return self.w_2(self.dropout(nn.functional.gelu(self.w_1(x))))


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
