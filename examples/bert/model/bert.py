import torch.nn as nn
import torch
import torch.nn.functional as F
from hfta.ops import get_hfta_op_for

from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):
  """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

  def __init__(self,
               vocab_size,
               hidden=768,
               n_layers=12,
               attn_heads=12,
               dropout=0.1,
               B=1):
    """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

    super().__init__()
    self.hidden = hidden
    self.n_layers = n_layers
    self.attn_heads = attn_heads
    self.B = B

    # paper noted they used 4*hidden_size for ff_network_hidden_size
    self.feed_forward_hidden = hidden * 4

    # embedding for BERT, sum of positional, segment, token embeddings
    self.embedding = BERTEmbedding(vocab_size=vocab_size,
                                   embed_size=hidden,
                                   B=B)

    # multi-layers transformer blocks, deep network
    self.transformer_blocks = nn.ModuleList([
        TransformerBlock(hidden, attn_heads, hidden * 4, dropout, B=B)
        for _ in range(n_layers)
    ])

    self.output = get_hfta_op_for(nn.Linear, B)(hidden, vocab_size)

  def forward(self, x, postion_info, segment_info=None):
    # attention masking for padded token
    # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
    if (segment_info == None):
      segment_info = torch.ones(x.shape)
    if self.B > 0:
      mask = (x <= 0).unsqueeze(2).repeat(1, 1, x.size(2), 1).unsqueeze(2)
    else:
      mask = (x <= 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

    # embedding the indexed sequence to sequence of vectors
    x = self.embedding(x, postion_info, segment_info)

    # running over multiple transformer blocks
    for transformer in self.transformer_blocks:
      x = transformer.forward(x, mask)

    return F.log_softmax(self.output(x), dim=-1)
