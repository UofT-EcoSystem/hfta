from typing import Optional

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn import init


class Embedding(Module):
  """ Input format: [B, N, *] (Integer tensor)
  B: Training array size (number of batches/jobs).
  N: Batch size.
  *: Words.
  """

  def __init__(self,
               num_embeddings,
               embedding_dim,
               padding_idx=None,
               max_norm=None,
               norm_type=2.,
               scale_grad_by_freq=False,
               sparse=False,
               _weight=None,
               B=1) -> None:
    super(Embedding, self).__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.B = B
    if padding_idx is not None:
      if padding_idx > 0:
        assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
      elif padding_idx < 0:
        assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
        padding_idx = self.num_embeddings + padding_idx
    self.padding_idx = padding_idx
    self.max_norm = max_norm
    self.norm_type = norm_type
    self.scale_grad_by_freq = scale_grad_by_freq
    self.addition = torch.arange(B) * self.num_embeddings
    if _weight is None:
      self.weight = Parameter(torch.Tensor(B, num_embeddings, embedding_dim))
      self.reset_parameters()
    else:
      assert list(_weight.shape) == [B, num_embeddings, embedding_dim], \
          'Shape of weight does not match num_embeddings and embedding_dim'
      self.weight = Parameter(_weight)
    self.sparse = sparse

  def reset_parameters(self) -> None:
    init.normal_(self.weight)
    if self.padding_idx is not None:
      with torch.no_grad():
        self.weight[self.padding_idx].fill_(0)

  def forward(self, input: Tensor) -> Tensor:
    #assume input is [B, N, *]
    assert self.B == input.shape[0], \
      "input should have shape [{}, N, *]".format(self.B)
    self.addition = self.addition.to(input.device)
    addition_shape = [self.B] + ([1] * (input.dim() - 1))
    addition = self.addition.view(addition_shape)
    new_input = (input + addition)
    res = F.embedding(new_input, self.weight.view(-1, self.embedding_dim),
                      self.padding_idx, self.max_norm, self.norm_type,
                      self.scale_grad_by_freq, self.sparse)
    return res

  def extra_repr(self) -> str:
    s = '{num_embeddings}, {embedding_dim}, {B}'
    if self.padding_idx is not None:
      s += ', padding_idx={padding_idx}'
    if self.max_norm is not None:
      s += ', max_norm={max_norm}'
    if self.norm_type != 2:
      s += ', norm_type={norm_type}'
    if self.scale_grad_by_freq is not False:
      s += ', scale_grad_by_freq={scale_grad_by_freq}'
    if self.sparse is not False:
      s += ', sparse=True'
    return s.format(**self.__dict__)

  # Did not checked
  @classmethod
  def from_pretrained(cls,
                      embeddings,
                      freeze=True,
                      padding_idx=None,
                      max_norm=None,
                      norm_type=2.,
                      scale_grad_by_freq=False,
                      sparse=False):
    assert embeddings.dim() == 2, \
        'Embeddings parameter is expected to be 2-dimensional'
    rows, cols = embeddings.shape
    embedding = cls(num_embeddings=rows,
                    embedding_dim=cols,
                    _weight=embeddings,
                    padding_idx=padding_idx,
                    max_norm=max_norm,
                    norm_type=norm_type,
                    scale_grad_by_freq=scale_grad_by_freq,
                    sparse=sparse)
    embedding.weight.requires_grad = not freeze
    return embedding

  def snatch_parameters(self, other, b):
    assert 0 <= b < self.B
    self.weight.data[b] = other.weight.data
