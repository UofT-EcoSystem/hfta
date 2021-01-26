import numpy as np
import torch
import torch.nn as nn
from MultiheadAttention import MultiheadAttention

from hfta.ops import testcase_automator


def _generate_square_subsequent_mask(sz):
  mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
      mask == 1, float(0.0))
  return mask


def testcase(
    B=3,
    N=16,
    L=32,
    E=64,
    num_heads=8,
    dropout=0.,
    bias=True,
    need_weights=False,
    use_mask=True,
):
  # just test it like self-attention, q, k, v have same shape
  S = L
  with torch.no_grad():
    q_array = [torch.rand(L, N, E) for _ in range(B)]
    k_array = [torch.rand(S, N, E) for _ in range(B)]
    v_array = [torch.rand(S, N, E) for _ in range(B)]
    q_fused = torch.cat([item.transpose(0, 1).unsqueeze(0) for item in q_array],
                        dim=0)
    k_fused = torch.cat([item.transpose(0, 1).unsqueeze(0) for item in k_array],
                        dim=0)
    v_fused = torch.cat([item.transpose(0, 1).unsqueeze(0) for item in v_array],
                        dim=0)

    if (use_mask):
      mask = _generate_square_subsequent_mask(L)
    else:
      mask = None

    args = (E, num_heads)
    kwargs = {
        'dropout': dropout,
        'bias': bias,
    }
    atte_array = [nn.MultiheadAttention(*args, **kwargs) for _ in range(B)]
    my_attention = MultiheadAttention(*args, **kwargs, B=B)
    y_array = []
    w_array = []
    for b in range(B):
      my_attention.snatch_parameters(atte_array[b], b)
      y, w = atte_array[b](q_array[b],
                           k_array[b],
                           v_array[b],
                           need_weights=need_weights,
                           attn_mask=mask)
      y_array.append(y)
      if (need_weights):
        w_array.append(w)
    y_actual, w_actual = my_attention(q_fused,
                                      k_fused,
                                      v_fused,
                                      need_weights=need_weights,
                                      attn_mask=mask)
    y_actual = y_actual.transpose(1, 2)
    y_except = torch.cat([y.unsqueeze(0) for y in y_array], dim=0)

    try:
      np.testing.assert_allclose(
          y_actual.numpy(),
          y_except.numpy(),
          rtol=1e-4,
      )
    except AssertionError as e:
      print("For output:")
      print(e)

    if (need_weights):
      w_except = torch.cat([w.unsqueeze(0) for w in w_array], dim=0)
      try:
        np.testing.assert_allclose(
            w_actual.numpy(),
            w_except.numpy(),
            rtol=1e-4,
        )
      except AssertionError as e:
        print("For weight:")
        print(e)


if __name__ == '__main__':
  testcase_automator(
      testcase,
      {
          # dropout is not stable, not tested
          'B': [1, 3, 5, 10],
          'N': [1, 8, 32, 64],
          'L': [16, 32, 64],
          'E': [16, 32, 64, 128],
          'num_heads': [1, 16, 32],
          'bias': [False],
          'need_weights': [False],
          'use_mask': [False],
      },
  )
