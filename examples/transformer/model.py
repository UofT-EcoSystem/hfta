import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hfta.ops import convert_ops, get_hfta_op_for, MultiheadAttention


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
  """Inject some information about the relative or absolute position of the tokens
      in the sequence. The positional encodings have the same dimension as
      the embeddings, so that the two can be summed. Here, we use sine and cosine
      functions of different frequencies.
  .. math::
      \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
      \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
      \text{where pos is the word position and i is the embed idx)
  Args:
      d_model: the embed dim (required).
      dropout: the dropout value (default=0.1).
      max_len: the max. length of the incoming sequence (default=5000).
  Examples:
      >>> pos_encoder = PositionalEncoding(d_model)
  """

  def __init__(self, d_model, dropout=0.1, max_len=5000, B=1):
    super(PositionalEncoding, self).__init__()
    self.dropout = get_hfta_op_for(nn.Dropout, B)(p=dropout)
    self.B = B
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    """Inputs of forward function
    Args:
        x: the sequence fed to the positional encoder model (required).
    Shape:
        x: [sequence length, batch size, B, embed dim]
        output: [sequence length, batch size, embed dim]
    Examples:
        >>> output = pos_encoder(x)
    """
    if self.B > 0:
      x = x + self.pe[:, :x.size(-2)].unsqueeze(0)
    else:
      x = x + self.pe[:, :x.size(-2)]
    return self.dropout(x)


class TransformerModel(nn.Module):
  """Container module with an encoder, a recurrent or transformer module, and a decoder."""

  def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, B=1):
    super(TransformerModel, self).__init__()
    Embedding, Linear, TransformerEncoder, TransformerEncoderLayer = convert_ops(
      B, nn.Embedding, nn.Linear, nn.TransformerEncoder,
      nn.TransformerEncoderLayer)
    self.model_type = 'Transformer'
    self.src_mask = None
    self.B = B
    self.pos_encoder = PositionalEncoding(ninp, dropout, B=B)
    encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    self.encoder = Embedding(ntoken, ninp)
    self.ninp = ninp
    self.decoder = Linear(ninp, ntoken)
    self.init_weights()

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask

  def init_weights(self):
    initrange = 0.1
    nn.init.uniform_(self.encoder.weight, -initrange, initrange)
    nn.init.zeros_(self.decoder.weight)
    nn.init.uniform_(self.decoder.weight, -initrange, initrange)

  def forward(self, src, has_mask=True):
    #src = [B, N, L]
    #output = [B, N, L, ntoken]
    if has_mask:
      device = src.device
      if self.src_mask is None or self.src_mask.size(0) != src.size(-1):
        mask = self._generate_square_subsequent_mask(src.size(-1)).to(device)
        self.src_mask = mask
    else:
      self.src_mask = None

    src = self.encoder(src) * math.sqrt(self.ninp)
    src = self.pos_encoder(src)
    output = self.transformer_encoder(src, self.src_mask)
    output = self.decoder(output)
    return F.log_softmax(output, dim=-1)
