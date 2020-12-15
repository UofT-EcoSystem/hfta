import numpy as np
import torch
import sys
sys.path.append('../')
from ops import get_hfta_op_for, testcase_automator


def testcase(B=3, N=32, C=5120, HWin=16, p=0.4):
  with torch.no_grad():
    x_array = [torch.ones(N, C, HWin, HWin) for _ in range(B)]
    x_fused = torch.cat([x.unsqueeze(1) for x in x_array], dim=1)
    dropout2d_fused = get_hfta_op_for(torch.nn.Dropout2d, B=B)(p)
    res_fused = dropout2d_fused(x_fused).transpose(0, 1) * (1 - p)
    res_fused_sum = torch.sum(res_fused.reshape(B * N * C, -1), dim=1).view(
        (B, N, C))

    # check the rate of zero channel is p
    # dropout2d has vary large various, it may fail some test though rtol is 2.5e-2
    y_fused_raw = torch.zeros((B, N, C))
    y_fused_raw[torch.where(res_fused_sum.view((B, N, C)) == 0)] = 1
    y_fused_actual = torch.mean(y_fused_raw, dim=2)
    y_fused_expect = torch.ones((B, N)) * p
    try:
      np.testing.assert_allclose(
          y_fused_actual.numpy(),
          y_fused_expect.numpy(),
          atol=max(2.5e-2, 1 / C),
      )
    except AssertionError as e:
      print(e)
      print("working on B=%d,N=%d, C=%d, HWin=%d, p=%f" % (B, N, C, HWin, p))

    # Check whether each channel either all zeros or no zeros
    res_fused_abs0 = torch.abs(res_fused_sum - HWin * HWin)
    y_fused_raw[torch.where(res_fused_abs0 <= 1e-4)] = 1
    y_fused_actual = y_fused_raw
    y_fused_expect = torch.ones((B, N, C))
    try:
      np.testing.assert_allclose(
          y_fused_actual.numpy(),
          y_fused_expect.numpy(),
          atol=1e-4,
      )
    except AssertionError as e:
      print(e)
      print("do not zero entire channel, "
            "working on B=%d,N=%d, C=%d, HWin=%d, p=%f" % (B, N, C, HWin, p))


if __name__ == '__main__':
  testcase_automator(
      testcase,
      {
          'p': [0.25, 0.5, 1],
          'B': [1, 4, 8],
          'N': [1, 8, 64],
          'C': [2048, 8192],
          'HWin': [8, 16, 32]
      },
  )
