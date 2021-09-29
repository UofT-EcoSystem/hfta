from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init


class _NormBase(Module):
  """Common base of _InstanceNorm and _BatchNorm"""

  _version = 2
  __constants__ = [
      'track_running_stats', 'momentum', 'eps', 'num_features', 'affine', 'B'
  ]
  num_features: int
  eps: float
  momentum: float
  affine: bool
  track_running_stats: bool
  B: int

  # WARNING: weight and bias purposely not defined here.
  # See https://github.com/pytorch/pytorch/issues/39670

  def __init__(self,
               num_features: int,
               eps: float = 1e-5,
               momentum: float = 0.1,
               affine: bool = True,
               track_running_stats: bool = True,
               device=None,
               dtype=None,
               B: int = 1) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(_NormBase, self).__init__()
    self.B = B
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.track_running_stats = track_running_stats
    if self.affine:
      self.weight = Parameter(torch.empty(B, num_features, **factory_kwargs))
      self.bias = Parameter(torch.empty(B, num_features, **factory_kwargs))
    else:
      self.register_parameter("weight", None)
      self.register_parameter("bias", None)
    if self.track_running_stats:
      self.register_buffer('running_mean',
                           torch.zeros(B, num_features, **factory_kwargs))
      self.register_buffer('running_var',
                           torch.ones(B, num_features, **factory_kwargs))
      self.running_mean: Optional[Tensor]
      self.running_var: Optional[Tensor]
      self.register_buffer(
          'num_batches_tracked',
          torch.tensor(
              0,
              dtype=torch.long,
              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
      self.num_batches_tracked: Optional[Tensor]
    else:
      self.register_buffer("running_mean", None)
      self.register_buffer("running_var", None)
      self.register_buffer("num_batches_tracked", None)
    self.reset_parameters()

  def reset_running_stats(self) -> None:
    if self.track_running_stats:
      # running_mean/running_var/num_batches... are registered at runtime depending
      # if self.track_running_stats is on
      self.running_mean.zero_()  # type: ignore[union-attr]
      self.running_var.fill_(1)  # type: ignore[union-attr]
      self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

  def reset_parameters(self) -> None:
    self.reset_running_stats()
    if self.affine:
      init.ones_(self.weight)
      init.zeros_(self.bias)

  def _check_input_dim(self, input):
    raise NotImplementedError

  def extra_repr(self):
    return ("{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}, B={B}".format(
                **self.__dict__))

  def _load_from_state_dict(
      self,
      state_dict,
      prefix,
      local_metadata,
      strict,
      missing_keys,
      unexpected_keys,
      error_msgs,
  ):
    version = local_metadata.get("version", None)

    if (version is None or version < 2) and self.track_running_stats:
      # at version 2: added num_batches_tracked buffer
      #               this should have a default value of 0
      num_batches_tracked_key = prefix + "num_batches_tracked"
      if num_batches_tracked_key not in state_dict:
        state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

    super(_NormBase, self)._load_from_state_dict(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    )

  def snatch_parameters(self, other, b):
    assert 0 <= b < self.B
    if self.affine:
      self.weight.data[b] = other.weight.data
      self.bias.data[b] = other.bias.data

    if self.track_running_stats:
      self.running_mean[b] = other.running_mean
      self.running_var[b] = other.running_var
      if self.num_batches_tracked == 0:
        self.num_batches_tracked = other.num_batches_tracked
      elif self.num_batches_tracked != other.num_batches_tracked:
        raise ValueError(
            "Got different \"num_batches_tracked\", {} != {} for b={}".format(
                self.num_batches_tracked, other.num_batches_tracked, b))


class _BatchNorm(_NormBase):

  def __init__(self,
               num_features,
               eps=1e-5,
               momentum=0.1,
               affine=True,
               track_running_stats=True,
               device=None,
               dtype=None,
               B=1):
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(_BatchNorm, self).__init__(num_features,
                                     eps,
                                     momentum,
                                     affine,
                                     track_running_stats,
                                     **factory_kwargs,
                                     B=B)

  def forward(self, input: Tensor) -> Tensor:
    # check dim and reshape the input
    self._check_input_dim(input)
    if (len(input.size()) == 3):
      input = input.transpose(0, 1)
    shape = list(input.size())
    new_shape = [shape[0], shape[1] * shape[2]] + shape[3:]
    input = input.reshape(new_shape)

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
      exponential_average_factor = 0.0
    else:
      exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:
      # TODO: if statement only here to tell the jit to skip emitting this when it is None
      if self.num_batches_tracked is not None:  # type: ignore[has-type]
        self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
        if self.momentum is None:  # use cumulative moving average
          exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:  # use exponential moving average
          exponential_average_factor = self.momentum
    r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
    if self.training:
      bn_training = True
    else:
      bn_training = (self.running_mean is None) and (self.running_var is None)
    r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
    self_weight, self_bias = (
        self.weight.view(self.B * self.num_features),
        self.bias.view(self.B * self.num_features),
    ) if self.affine else (self.weight, self.bias)

    self_running_mean, self_running_var = (
        self.running_mean.view(self.B * self.num_features),
        self.running_var.view(self.B * self.num_features),
    ) if self.track_running_stats else (self.running_mean, self.running_var)

    res = F.batch_norm(
        input,
        # If buffers are not to be tracked, ensure that they won't be updated
        self_running_mean
        if not self.training or self.track_running_stats else None,
        self_running_var
        if not self.training or self.track_running_stats else None,
        self_weight,
        self_bias,
        bn_training,
        exponential_average_factor,
        self.eps,
    ).view(shape)

    if (len(shape) == 3):
      res = res.transpose(0, 1)

    return res


class BatchNorm1d(_BatchNorm):
  r"""Based on PyTorch(1.9.0)'s BatchNorm1d source code:
  https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm1d

  Input format: [B N C] or [N B C L]
  B: Training array size (number of batches/jobs).
  N: Batch size.
  C: Number of features.
  L: feature length.

  """

  def _check_input_dim(self, input):
    if input.dim() != 3 and input.dim() != 4:
      raise ValueError("expected 3D or 4D input (got {}D input)".format(
          input.dim()))


class BatchNorm2d(_BatchNorm):
  r"""Based on PyTorch(1.9.0)'s BatchNorm2d source code:
  https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d

  Input format: [N B C H W]
  B: Training array size (number of batches/jobs).
  N: Batch size.
  C: Number of features.
  H: Height.
  W: Width.
  """

  def _check_input_dim(self, input):
    if input.dim() != 5:
      raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
