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

from .utils import SublayerConnection, PositionwiseFeedForward
from hfta.ops import get_hfta_op_for, MultiheadAttention


class TransformerBlock(nn.Module):
  """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
  """

  def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, B=1):
    """
      :param hidden: hidden size of transformer
      :param attn_heads: head sizes of multi-head attention
      :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
      :param dropout: dropout rate
    """

    super().__init__()
    self.attention = MultiheadAttention(hidden, attn_heads, B=B)
    self.feed_forward = PositionwiseFeedForward(d_model=hidden,
                                                d_ff=feed_forward_hidden,
                                                dropout=dropout,
                                                B=B)
    self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout, B=B)
    self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout, B=B)
    self.dropout = get_hfta_op_for(nn.Dropout, B)(p=dropout)

  def forward(self, x, mask):
    x = self.input_sublayer(
        x, lambda _x: self.attention.forward(_x, _x, _x, attn_mask=mask)[0])
    x = self.output_sublayer(x, self.feed_forward)
    return self.dropout(x)
