import torch.nn as nn
from hfta.ops import get_hfta_op_for


class BERTEmbedding(nn.Module):
  """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

  def __init__(self, vocab_size, embed_size, dropout=0.1, B=1):
    """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
    super().__init__()
    Embedding = get_hfta_op_for(nn.Embedding, B)
    self.token = Embedding(vocab_size, embed_size)
    self.position = Embedding(512, embed_size)
    self.segment = Embedding(3, embed_size)
    self.dropout = get_hfta_op_for(nn.Dropout, B)(p=dropout)
    self.embed_size = embed_size

  def forward(self, sequence, postion_label, segment_label):
    x = self.token(sequence)
    p = self.position(postion_label)
    s = self.segment(segment_label)
    x += p + s
    return self.dropout(x)
