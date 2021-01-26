import torch.nn as nn

from hfta.hfta_ops.MultiheadAttention import MultiheadAttention
from .utils import SublayerConnection, PositionwiseFeedForward
from hfta.ops import get_hfta_op_for


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
