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
    Embedding, Linear, TransformerEncoder = convert_ops(B, nn.Embedding,
                                                        nn.Linear,
                                                        nn.TransformerEncoder)
    self.model_type = 'Transformer'
    self.src_mask = None
    self.B = B
    self.pos_encoder = PositionalEncoding(ninp, dropout, B=B)
    encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, B=B)
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


def _get_activation_fn(activation):
  if activation == "relu":
    return F.relu
  elif activation == "gelu":
    return F.gelu

  raise RuntimeError(
      "activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(torch.nn.Module):
  """TransformerEncoderLayer is made up of self-attn and feedforward network.
  This standard encoder layer is based on the paper "Attention Is All You Need".
  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
  Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
  Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
  in a different way during application.

  Args:
      d_model: the number of expected features in the input (required).
      nhead: the number of heads in the multiheadattention models (required).
      dim_feedforward: the dimension of the feedforward network model (default=2048).
      dropout: the dropout value (default=0.1).
      activation: the activation function of intermediate layer, relu or gelu (default=relu).

  Examples::
      >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
      >>> src = torch.rand(10, 32, 512)
      >>> out = encoder_layer(src)
  """

  def __init__(self,
               d_model,
               nhead,
               dim_feedforward=2048,
               dropout=0.1,
               activation="relu",
               B=1):
    super(TransformerEncoderLayer, self).__init__()
    self.B = B

    Dropout, Linear, LayerNorm = convert_ops(B, nn.Dropout, nn.Linear,
                                             nn.LayerNorm)
    self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, B=B)

    # Implementation of Feedforward model
    self.linear1 = Linear(d_model, dim_feedforward)
    self.dropout = Dropout(dropout)
    self.linear2 = Linear(dim_feedforward, d_model)

    self.norm1 = LayerNorm(d_model)
    self.norm2 = LayerNorm(d_model)
    self.dropout1 = Dropout(dropout)
    self.dropout2 = Dropout(dropout)

    self.activation = _get_activation_fn(activation)

  def __setstate__(self, state):
    if 'activation' not in state:
      state['activation'] = F.relu
    super(TransformerEncoderLayer, self).__setstate__(state)

  def forward(self, src, src_mask=None, src_key_padding_mask=None):
    """Pass the input through the encoder layer.

    Args:
        src: the sequence to the encoder layer (required).
        src_mask: the mask for the src sequence (optional).
        src_key_padding_mask: the mask for the src keys per batch (optional).

    Shape:
        see the docs in Transformer class.
    """
    src2 = self.self_attn(src,
                          src,
                          src,
                          attn_mask=src_mask,
                          key_padding_mask=src_key_padding_mask)[0]
    src = src + self.dropout1(src2)

    src = self.norm1(src)
    src = self.linear1(src)
    src2 = self.linear2(self.dropout(self.activation(src)))
    src = src + self.dropout2(src2)
    src = self.norm2(src)
    return src
