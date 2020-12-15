import numpy as np
import torch
import torch.nn as nn

from hfta.ops import get_hfta_op_for, testcase_automator


def testcase(B=3,
             N=32,
             input_dim=(20,),
             num_embeddings=200,
             embedding_dim=50,
             padding_idx=None,
             max_norm=None,
             norm_type=2.,
             scale_grad_by_freq=False,
             sparse=False,
             _weight=None):
  with torch.no_grad():
    x_array = [
        torch.randint(num_embeddings, [N] + list(input_dim)) for _ in range(B)
    ]
    x_fused = torch.cat([x.unsqueeze(0) for x in x_array], dim=0)
    args = (num_embeddings, embedding_dim)
    kwargs = {
        'padding_idx': padding_idx,
        'max_norm': max_norm,
        'norm_type': norm_type,
        'scale_grad_by_freq': scale_grad_by_freq,
        'sparse': sparse,
        '_weight': _weight,
    }
    embedding_array = [nn.Embedding(*args, **kwargs) for _ in range(B)]
    embedding_fused = get_hfta_op_for(nn.Embedding, B=B)(*args, **kwargs)
    # Init weights and biases.
    for b in range(B):
      embedding_fused.snatch_parameters(embedding_array[b], b)
    y_array = [embedding_array[b](x_array[b]) for b in range(B)]
    y_fused_actual = embedding_fused(x_fused)
    y_fused_expect = torch.cat([y.unsqueeze(0) for y in y_array], dim=0)
    try:
      np.testing.assert_allclose(
          y_fused_actual.numpy(),
          y_fused_expect.numpy(),
          rtol=1e-4,
      )
    except AssertionError as e:
      print(e)


if __name__ == '__main__':
  testcase_automator(
      testcase,
      {
          'B': [1, 2, 5, 10],
          'N': [1, 8, 64],
          'input_dim': [(32,), (16, 16), (8, 8, 8), (128,), (512,)],
          'num_embeddings': [50, 200, 2000],
          'embedding_dim': [32, 128, 786],
          'padding_idx': [0],
      },
  )
