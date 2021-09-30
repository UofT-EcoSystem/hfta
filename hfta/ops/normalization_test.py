import numpy as np
import torch
import torch.nn as nn

from hfta.ops import (get_hfta_op_for, testcase_automator, assert_allclose,
                      dump_error_msg)


def testcase(
    B=5,
    N=64,
    normalized_shape=(200,),
    elementwise_affine=True,
    device=torch.device('cpu'),
    dtype=torch.float,
):
  with torch.no_grad():
    x_array = [
        torch.rand([N] + list(normalized_shape), device=device, dtype=dtype)
        for _ in range(B)
    ]
    x_fused = torch.cat([x.unsqueeze(0) for x in x_array], dim=0)
    args = (normalized_shape,)
    kwargs = {
        'elementwise_affine': elementwise_affine,
        'device': device,
        'dtype': dtype,
    }
    layernorm_array = [nn.LayerNorm(*args, **kwargs) for _ in range(B)]
    layernorm_fused = get_hfta_op_for(nn.LayerNorm, B=B)(*args, **kwargs)
    # Init weights and biases.
    for b in range(B):
      layernorm_fused.snatch_parameters(layernorm_array[b], b)
    y_array = [layernorm_array[b](x_array[b]) for b in range(B)]
    y_fused_actual = layernorm_fused(x_fused)
    y_fused_expect = torch.cat([y.unsqueeze(0) for y in y_array], dim=0)
    try:
      assert_allclose(
          y_fused_actual.cpu().numpy(),
          y_fused_expect.cpu().numpy(),
          rtol=1e-4,
      )
    except AssertionError as e:
      dump_error_msg(e)


if __name__ == '__main__':
  testcase_automator(
      testcase,
      {
          'B': [1, 2, 5, 10],
          'N': [1, 8, 64],
          'normalized_shape': [(8, 10, 20), (20,), (10, 20)],
          'elementwise_affine': [False],
          'device': [torch.device('cuda:0')],
          'dtype': [torch.double],
      },
  )
