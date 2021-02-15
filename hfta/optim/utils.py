import torch
import torch.nn as nn
import numpy as np
import itertools

from hfta.ops import get_hfta_op_for


def _snatch_grads_unfused(op_list, op, b):
  op_list[b].weight.grad = op.weight.grad
  if op_list[b].bias is not None:
    op_list[b].bias.grad = op.bias.grad


def _snatch_grads_linear(fused_op, op, b, fused=True):
  if fused:
    fused_op.weight.grad[b] = op.weight.grad.transpose(0, 1)
    if fused_op.bias is not None:
      fused_op.bias.grad[b] = op.bias.grad.unsqueeze(0)
  else:
    _snatch_grads_unfused(fused_op, op, b)


def _snatch_grads_conv2d(fused_op, op, b, fused=True):
  if fused:
    fused_op.weight.grad[b] = op.weight.grad
    if fused_op.bias is not None:
      fused_op.bias.grad[b] = op.bias.grad
  else:
    _snatch_grads_unfused(fused_op, op, b)


def _snatch_parameters_unfused(op_list, op, b):
  op_list[b].weight.data = op.weight.data
  if op_list[b].bias is not None:
    op_list[b].bias.data = op.bias.data


def _assert_params_unfused(op_list, op, b):
  np.testing.assert_allclose(
      op_list[b].weight.data.numpy(),
      op.weight.data.numpy(),
      rtol=1e-4,
  )
  if op_list[b].bias is not None:
    np.testing.assert_allclose(
        op_list[b].bias.data.numpy(),
        op.bias.data.numpy(),
        rtol=1e-4,
    )


def _assert_params_linear(fused_op, op, b, fused=True):
  try:
    if fused:
      np.testing.assert_allclose(
          fused_op.weight.data[b].numpy(),
          op.weight.data.transpose(0, 1).numpy(),
          rtol=1e-4,
      )
      if fused_op.bias is not None:
        np.testing.assert_allclose(
            fused_op.bias.data[b].numpy(),
            op.bias.data.unsqueeze(0).numpy(),
            rtol=1e-4,
        )
    else:
      _assert_params_unfused(fused_op, op, b)
  except AssertionError as e:
    print(e)


def _assert_params_conv2d(fused_op, op, b, fused=True):
  try:
    if fused:
      np.testing.assert_allclose(
          fused_op.weight.data[b].numpy(),
          op.weight.data.numpy(),
          rtol=1e-4,
      )
      if fused_op.bias is not None:
        np.testing.assert_allclose(
            fused_op.bias.data[b].numpy(),
            op.bias.data.numpy(),
            rtol=1e-4,
        )
    else:
      _assert_params_unfused(fused_op, op, b)
  except AssertionError as e:
    print(e)


class _TestNet(nn.Module):

  def __init__(self, B=0, partially_fused=False):
    super(_TestNet, self).__init__()
    self.conv1 = get_hfta_op_for(nn.Conv2d, B=B)(3, 16, 3, 3)
    if partially_fused:
      self.conv2 = [nn.Conv2d(64, 32, 5, 5) for _ in range(B)]
    else:
      self.conv2 = get_hfta_op_for(nn.Conv2d, B=B)(64, 32, 5, 5)
    if partially_fused:
      self.linear1 = [nn.Linear(10, 30) for _ in range(B)]
    else:
      self.linear1 = get_hfta_op_for(nn.Linear, B=B)(10, 30)
    self.linear2 = get_hfta_op_for(nn.Linear, B=B)(100, 20)
    self.partially_fused = partially_fused

  def snatch_parameters(self, net, b):
    self.conv1.snatch_parameters(net.conv1, b)
    if self.partially_fused:
      _snatch_parameters_unfused(self.conv2, net.conv2, b)
      _snatch_parameters_unfused(self.linear1, net.linear1, b)
    else:
      self.conv2.snatch_parameters(net.conv2, b)
      self.linear1.snatch_parameters(net.linear1, b)
    self.linear2.snatch_parameters(net.linear2, b)

  def snatch_grads(self, net, b):
    _snatch_grads_conv2d(self.conv1, net.conv1, b)
    _snatch_grads_conv2d(
        self.conv2,
        net.conv2,
        b,
        fused=(not self.partially_fused),
    )
    _snatch_grads_linear(
        self.linear1,
        net.linear1,
        b,
        fused=(not self.partially_fused),
    )
    _snatch_grads_linear(self.linear2, net.linear2, b)

  def unfused_parameters(self):
    if not self.partially_fused:
      return []
    B = len(self.conv2)
    assert B == len(self.linear1)
    return [
        itertools.chain(
            self.conv2[b].parameters(),
            self.linear1[b].parameters(),
        ) for b in range(B)
    ]

  def assert_params(self, net, b):
    _assert_params_conv2d(self.conv1, net.conv1, b)
    _assert_params_conv2d(
        self.conv2,
        net.conv2,
        b,
        fused=(not self.partially_fused),
    )
    _assert_params_linear(
        self.linear1,
        net.linear1,
        b,
        fused=(not self.partially_fused),
    )
    _assert_params_linear(self.linear2, net.linear2, b)


def _init_test_nets_with_grads(net_fused, net_array):
  B = len(net_array)
  # Sync. init. parameters.
  for b in range(B):
    net_fused.snatch_parameters(net_array[b], b)
  # Init. grads for net_array.
  for b in range(B):
    for p in net_array[b].parameters():
      p.grad = torch.rand_like(p)
  # Init. grads for net_fused to zeros.
  for p in net_fused.parameters():
    p.grad = torch.zeros_like(p)
  for b_params in net_fused.unfused_parameters():
    for p in b_params:
      p.grad = torch.zeros_like(p)
  # Assign net_fused with the same grads from net_array.
  for b in range(B):
    net_fused.snatch_grads(net_array[b], b)


def _take_step_on_test_optimizers(optimizer_fused, optimizer_array):
  for optimizer in optimizer_array:
    optimizer.step()
  optimizer_fused.step()


def _verify_test_nets_params(net_fused, net_array):
  B = len(net_array)
  for b in range(B):
    net_fused.assert_params(net_array[b], b)


def _optim_testing_procedure(net_fused, net_array, optimizer_fused,
                             optimizer_array):
  # Init net_array and net_fused with gradients.
  _init_test_nets_with_grads(net_fused, net_array)
  # Call step().
  _take_step_on_test_optimizers(optimizer_fused, optimizer_array)
  # Verify parameter values.
  _verify_test_nets_params(net_fused, net_array)


def _validate_range_for_element(name, e, lb, ub):
  if e < lb or e > ub:
    raise ValueError("Invalid {}: {}".format(name, val))


def _validate_range(name, val, lb, ub):
  if isinstance(val, (float, int)):
    _validate_range_for_element(name, val, lb, ub)
  elif isinstance(val, (list, tuple, torch.Tensor, np.ndarray)):
    for e in val:
      _validate_range_for_element(name, e, lb, ub)
  else:
    raise ValueError("Unsupported type({}): {}".format(val, type(val)))


def _to_tensor(coeff, B, dtype=torch.float):
  if isinstance(coeff, (float, int)):
    res = coeff
  elif isinstance(coeff, (list, tuple, np.ndarray)):
    if isinstance(coeff, (list, tuple)):
      assert len(coeff) == B
    else:
      assert len(coeff.shape) == 1 and coeff.shape[0] == B
    res = torch.as_tensor(coeff, dtype=dtype)
  elif isinstance(coeff, torch.Tensor):
    assert coeff.dim() == 1 and coeff.size(0) == B
    res = coeff
  else:
    raise ValueError("Unsupported type({}): {}".format(coeff, type(coeff)))
  return res


def _get_coeff_like_params_map(coeff, params, B):
  res = {}
  for p in params:
    assert p.size(0) == B
    res[p] = coeff.view(B, *([1] * (p.dim() - 1)))
  return res


def _broadcastablize(optimizer, name, B, is_tuple=False):
  for group in optimizer.param_groups:
    if is_tuple:
      coeffs = group[name]
      new_coeffs = []
      for coeff in coeffs:
        coeff = _to_tensor(coeff, B)
        if isinstance(coeff, (float, int)):
          new_coeffs.append(coeff)
        else:
          assert isinstance(coeff, torch.Tensor)
          new_coeffs.append(
              _get_coeff_like_params_map(coeff, group['params'], B))
      group[name] = tuple(new_coeffs)
    else:
      coeff = group[name]
      coeff = _to_tensor(coeff, B)
      if isinstance(coeff, (float, int)):
        continue
      assert isinstance(coeff, torch.Tensor)
      group[name] = _get_coeff_like_params_map(coeff, group['params'], B)


def _move_coeff_to_same_device(group, name, p, is_tuple=False):
  if is_tuple:
    for coeff in group[name]:
      if isinstance(coeff, dict):
        coeff[p] = coeff[p].to(p.device)
  else:
    if isinstance(group[name], dict):
      group[name][p] = group[name][p].to(p.device)


def index_array_or_return_scalar(array_or_scalar, b):
  scalar_types = (int, float)
  if isinstance(array_or_scalar, scalar_types):
    return array_or_scalar
  elif isinstance(array_or_scalar, (list, tuple)):
    return array_or_scalar[b]
  elif isinstance(array_or_scalar, (torch.Tensor, np.ndarray)):
    return array_or_scalar[b].item()
  else:
    raise ValueError('Unsupported type({}) = {}!'.format(
        array_or_scalar, type(array_or_scalar)))


def _reduce_array_if_possible(arr):
  if isinstance(arr, (list, tuple, np.ndarray, torch.Tensor)):
    first = arr[0]
    for e in arr[1:]:
      if e != first:
        return arr
    if isinstance(arr, (torch.Tensor, np.ndarray)):
      return first.item()
    else:
      return first
  else:
    return arr


def _reduce_array_if_possible_for(*coeffs):
  return (_reduce_array_if_possible(coeff) for coeff in coeffs)


def consolidate_hyperparams_and_determine_B(args, hp_names):
  B = 1
  for hp_name in hp_names:
    hp = getattr(args, hp_name)
    assert isinstance(hp, list)
    B = max(B, len(hp)) if B == 1 else B
    if len(hp) == 1:
      setattr(args, hp_name, hp[0])
    else:
      assert len(hp) == B
  return B


def _zero_grad_if_cuda(optimizer):
  if optimizer.param_groups[0]['params'][0].is_cuda:
    for group in optimizer.param_groups:
      for p in group['params']:
        p.grad = None
    return True
  else:
    return False
