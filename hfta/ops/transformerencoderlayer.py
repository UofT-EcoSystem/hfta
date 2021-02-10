import torch.nn as nn
import torch
import torch.nn.functional as F
import functools


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
    from .linear import Linear
    from .layernorm import LayerNorm
    from .multiheadattention import MultiheadAttention
    Linear = nn.Linear if B == 0 else functools.partial(Linear, B=B)
    Dropout = nn.Dropout
    LayerNorm = nn.LayerNorm if B == 0 else functools.partial(LayerNorm, B=B)
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


def _get_activation_fn(activation):
  if activation == "relu":
    return F.relu
  elif activation == "gelu":
    return F.gelu

  raise RuntimeError(
      "activation should be relu/gelu, not {}".format(activation))
