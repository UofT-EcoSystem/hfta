import numpy as np
import torch

from hfta.ops import (get_hfta_op_for, testcase_automator, support_dtype,
                      assert_allclose)


def testcase(
    B=3,
    N=32,
    C=1024,
    HWin=16,
    p=0.4,
    device=torch.device('cpu'),
    dtype=torch.float,
):
  if not support_dtype(device, dtype):
    return
  with torch.no_grad():
    x_array = [
        torch.ones(N, C, HWin, HWin, device=device, dtype=dtype)
        for _ in range(B)
    ]
    x_fused = torch.cat([x.unsqueeze(1) for x in x_array], dim=1)
    dropout2d_fused = get_hfta_op_for(torch.nn.Dropout2d, B=B)(p)
    y_fused = dropout2d_fused(x_fused)
    for b in range(B):
      y = y_fused[:, b, :, :, :]
      assert y.size(0) == N
      assert y.size(1) == C
      for n in range(N):
        zero_channels = 0
        for c in range(C):
          s = y[n, c].sum()
          # Each channel either has all zeros or no zeros.
          try:
            assert_allclose(s.cpu(), HWin**2 / (1 - p), rtol=1e-4)
          except AssertionError as e:
            assert_allclose(s.cpu(), 0, atol=1e-4)
            # s must be zero at this point.
            zero_channels += 1
        assert_allclose(zero_channels / C, p, rtol=2e-1)


if __name__ == '__main__':
  testcase_automator(
      testcase,
      {
          'p': [0.25, 0.99],
          'B': [1, 8],
          'N': [1, 64],
          'C': [2000],
          'HWin': [8, 32],
          'device': [torch.device('cuda:0')],
          'dtype': [torch.half, torch.double],
      },
  )
