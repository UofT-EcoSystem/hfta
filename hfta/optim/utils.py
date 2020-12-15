import torch
import torch.nn as nn
import numpy as np

from hfta.ops import get_hfta_op_for


class _TestNet(nn.Module):

  def __init__(self, B=0):
    super(_TestNet, self).__init__()
    self.conv1 = get_hfta_op_for(nn.Conv2d, B=B)(3, 16, 3, 3)
    self.conv2 = get_hfta_op_for(nn.Conv2d, B=B)(64, 32, 5, 5)
    self.linear1 = get_hfta_op_for(nn.Linear, B=B)(10, 30)
    self.linear2 = get_hfta_op_for(nn.Linear, B=B)(100, 20)

  def snatch_parameters(self, net, b):
    self.conv1.snatch_parameters(net.conv1, b)
    self.conv2.snatch_parameters(net.conv2, b)
    self.linear1.snatch_parameters(net.linear1, b)
    self.linear2.snatch_parameters(net.linear2, b)


def _init_test_nets_with_grads(net_fused, net_array):
  B = len(net_array)
  for b in range(B):
    net_fused.snatch_parameters(net_array[b], b)
  grads_list = [[] for _ in net_fused.parameters()]
  for b in range(B):
    for i, p in enumerate(net_array[b].parameters()):
      p.grad = torch.rand_like(p)
      grads_list[i].append(p.grad.unsqueeze(0))
  for i, p in enumerate(net_fused.parameters()):
    cat_grad = torch.cat(grads_list[i])
    if p.dim() == 3:  # Linear layer.
      if cat_grad.dim() == 2:  # bias
        cat_grad = cat_grad.unsqueeze(1)
      else:  # weight
        cat_grad = cat_grad.transpose(1, 2)
    p.grad = cat_grad


def _take_step_on_test_optimizers(optimizer_fused, optimizer_array):
  for optimizer in optimizer_array:
    optimizer.step()
  optimizer_fused.step()


def _verify_test_nets_params(net_fused, net_array):
  B = len(net_array)
  params_list = [[] for _ in net_fused.parameters()]
  for b in range(B):
    for i, p in enumerate(net_array[b].parameters()):
      params_list[i].append(p.data.unsqueeze(0))
  for i, p_fused in enumerate(net_fused.parameters()):
    cat_param = torch.cat(params_list[i])
    if p_fused.dim() == 3:  # Linear layer.
      if cat_param.dim() == 2:  # bias
        cat_param = cat_param.unsqueeze(1)
      else:  # weight
        cat_param = cat_param.transpose(1, 2)
    try:
      np.testing.assert_allclose(
          p_fused.data.numpy(),
          cat_param.numpy(),
          rtol=1e-4,
      )
    except AssertionError as e:
      print(e)


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
