import torch
import torch.nn as nn
import torch.nn.functional as F
from hfta.ops import get_hfta_op_for


class MultiheadAttention(nn.Module):

  def __init__(self,
               embed_dim,
               num_heads,
               dropout=0.,
               bias=True,
               activation=F.relu,
               B=1):
    """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
    super(MultiheadAttention, self).__init__()
    if embed_dim % num_heads != 0:
      raise ValueError(
          '`in_features`({}) should be divisible by `head_num`({})'.format(
              embed_dim, num_heads))
    Linear = get_hfta_op_for(nn.Linear, B=B)
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.B = B
    self.head_dim = self.embed_dim // self.num_heads
    assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    self.scaling = float(self.head_dim)**-0.5
    self.dropout = dropout
    self.activation = activation
    self.bias = bias
    self.linear_q = Linear(embed_dim, embed_dim, bias)
    self.linear_k = Linear(embed_dim, embed_dim, bias)
    self.linear_v = Linear(embed_dim, embed_dim, bias)
    self.linear_o = Linear(embed_dim, embed_dim)

  def _reset_parameters(self):
    tmp_weight = torch.zeros((3 * self.embed_dim, self.embed_dim))
    torch.nn.init.xavier_uniform_(tmp_weight)
    tmp_weight = tmp_weight.reshape((3, self.embed_dim, self.embed_dim))
    self.linear_q.weight.data = tmp_weight[0]
    self.linear_k.weight.data = tmp_weight[1]
    self.linear_v.weight.data = tmp_weight[2]
    if self.bias:
      torch.nn.init.constant_(self.linear_q.bias, 0.)
      torch.nn.init.constant_(self.linear_k.bias, 0.)
      torch.nn.init.constant_(self.linear_v.bias, 0.)

  def forward(self,
              query,
              key,
              value,
              key_padding_mask=None,
              need_weights=True,
              attn_mask=None):
    # attn_mask == [B, N, num_heads, L, S] or [L, S]
    # query = [B, N, L, E]
    # key = value = [B, N, S, E]
    # assert S == L
    # output: o = [B, N, L, E], o_weight = [B, N, L, S]
    N = query.shape[0]

    if self.B > 0:
      N = query.size(1)
      query = query.contiguous().view(query.shape[0], -1, query.shape[-1])
      key = key.contiguous().view(key.shape[0], -1, key.shape[-1])
      value = value.contiguous().view(value.shape[0], -1, value.shape[-1])

    query = self.linear_q(query.contiguous())
    key = self.linear_k(key.contiguous())
    value = self.linear_v(value.contiguous())

    if self.B > 0:
      query = query.view(self.B * N, -1, query.shape[-1])
      key = key.view(self.B * N, -1, key.shape[-1])
      value = value.view(self.B * N, -1, value.shape[-1])

    bsz, tgt_len, embed_dim = query.size()
    src_len = key.size(1)

    query *= self.scaling

    query = query.contiguous().view(bsz, tgt_len, self.num_heads,
                                    self.head_dim).transpose(1, 2)
    query = query.contiguous().view(bsz * self.num_heads, tgt_len,
                                    self.head_dim)

    key = key.contiguous().view(bsz, src_len, self.num_heads,
                                self.head_dim).transpose(1, 2)
    key = key.contiguous().view(bsz * self.num_heads, src_len, self.head_dim)

    value = value.contiguous().view(bsz, src_len, self.num_heads,
                                    self.head_dim).transpose(1, 2)
    value = value.contiguous().view(bsz * self.num_heads, src_len,
                                    self.head_dim)

    o_weights = torch.bmm(query, key.transpose(1, 2))

    if attn_mask is not None:
      if attn_mask.dim() > 4 + min(self.B, 1) or attn_mask.dim() < 2:
        raise RuntimeError("attn_mask's dimension {} is not supported".format(
            attn_mask.dim()))

      while attn_mask.dim() < 4 + min(self.B, 1):
        attn_mask = attn_mask.unsqueeze(0)
      if not ((self.B == 0 or (attn_mask.size(0) in [1, self.B])) and
              (attn_mask.size(-4) in [1, N] and
               attn_mask.size(-3) in [1, self.num_heads] and attn_mask.size(-2)
               == query.size(-2) and attn_mask.size(-1) == key.size(-2))):
        raise RuntimeError('The size of the attn_mask is not correct.')
      # attn_mask's dim is 5 now.

      old_shape = o_weights.shape
      if self.B > 0:
        o_weights = o_weights.view((self.B, N, -1, tgt_len, src_len))
      else:
        o_weights = o_weights.view((N, -1, tgt_len, src_len))

      if attn_mask.dtype == torch.bool:
        o_weights.masked_fill_(attn_mask, float('-inf'))
      else:
        o_weights += attn_mask
      o_weights = o_weights.view(old_shape)

    o_weights = F.softmax(o_weights, dim=-1)
    o_weights = F.dropout(o_weights, p=self.dropout, training=self.training)
    o = torch.bmm(o_weights, value)
    if self.B > 0:
      o = o.contiguous().view(self.B, N, self.num_heads, tgt_len, self.head_dim)
      o = o.transpose(2, 3).contiguous().view(self.B, N, tgt_len, embed_dim)
    else:
      o = o.contiguous().view(bsz, self.num_heads, tgt_len, self.head_dim)
      o = o.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
    o = self.linear_o(o)

    if need_weights:
      # average attention weights over heads
      o_weights = o_weights.view(bsz, self.num_heads, tgt_len, src_len)
      o_weights = o_weights.sum(dim=1) / self.num_heads
      if (self.B > 0):
        o_weights = o_weights.view(self.B, N, tgt_len, src_len)
      return o, o_weights
    else:
      return o, None

  @staticmethod
  def gen_history_mask(x):
    """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
    batch_size, seq_len, _ = x.size()
    res = torch.tril(torch.ones(seq_len, seq_len))
    return res.view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

  def extra_repr(self):
    return 'in_features={}, head_num={}, bias={}, activation={}'.format(
        self.embed_dim,
        self.num_heads,
        self.bias,
        self.activation,
    )

  def snatch_parameters(self, other):
    assert isinstance(other, nn.MultiheadAttention)
    assert other._qkv_same_embed_dim
    assert other.bias_k is None and other.bias_v is None
    assert self.embed_dim == other.embed_dim
    assert self.num_heads == other.num_heads

    tmp_weight = other.in_proj_weight.reshape(3, self.embed_dim, self.embed_dim)
    self.linear_q.weight.data = tmp_weight[0].view(
        self.linear_q.weight.data.shape)
    self.linear_k.weight.data = tmp_weight[1].view(
        self.linear_k.weight.data.shape)
    self.linear_v.weight.data = tmp_weight[2].view(
        self.linear_v.weight.data.shape)

    if self.bias:
      tmp_bias = other.in_proj_bias.reshape(3, self.embed_dim)
      self.linear_q.bias.data = tmp_bias[0].view(self.linear_q.bias.data.shape)
      self.linear_k.bias.data = tmp_bias[1].view(self.linear_k.bias.data.shape)
      self.linear_v.bias.data = tmp_bias[2].view(self.linear_v.bias.data.shape)

    self.linear_o.weight.data = other.out_proj.weight.data.view(
        self.linear_o.weight.data.shape)
    self.linear_o.bias.data = other.out_proj.bias.data.view(
        self.linear_o.bias.data.shape)
