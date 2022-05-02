import functools
import torch.nn

from .conv import Conv1d, Conv2d, ConvTranspose2d
from .linear import Linear
from .pooling import MaxPool2d, AdaptiveAvgPool2d, AvgPool2d
from .dropout2d import Dropout2d
from .batchnorm import BatchNorm1d, BatchNorm2d
from .utils import testcase_automator, assert_allclose, dump_error_msg
from .sparse import Embedding
from .normalization import LayerNorm
from .multiheadattention import MultiheadAttention
from .transformerencoderlayer import TransformerEncoderLayer

_OPS_MAP = {
    torch.nn.Conv1d: Conv1d,
    torch.nn.Conv2d: Conv2d,
    torch.nn.ConvTranspose2d: ConvTranspose2d,
    torch.nn.Linear: Linear,
    torch.nn.MaxPool2d: MaxPool2d,
    torch.nn.AdaptiveAvgPool2d: AdaptiveAvgPool2d,
    torch.nn.AvgPool2d: AvgPool2d,
    torch.nn.Dropout2d: Dropout2d,
    torch.nn.BatchNorm1d: BatchNorm1d,
    torch.nn.BatchNorm2d: BatchNorm2d,
    torch.nn.LayerNorm: LayerNorm,
    torch.nn.Embedding: Embedding,
}

_TORCH_NN_ELEWISE = {
    torch.nn.ELU,
    torch.nn.Hardshrink,
    torch.nn.Hardsigmoid,
    torch.nn.Hardtanh,
    torch.nn.Hardswish,
    torch.nn.LeakyReLU,
    torch.nn.LogSigmoid,
    torch.nn.PReLU,
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.RReLU,
    torch.nn.SELU,
    torch.nn.CELU,
    torch.nn.GELU,
    torch.nn.Sigmoid,
    torch.nn.SiLU,
    torch.nn.Mish,
    torch.nn.Softplus,
    torch.nn.Softshrink,
    torch.nn.Softsign,
    torch.nn.Tanh,
    torch.nn.Tanhshrink,
    torch.nn.Threshold,
}

_HFTA_TORCH_IDENTICAL_OPS = {
    torch.nn.Identity,
    torch.nn.Dropout,
    torch.nn.TransformerEncoder,
}

_HFTA_TORCH_IDENTICAL_OPS.update(_TORCH_NN_ELEWISE)


def get_hfta_op_for(torch_op_class, B=1):
  if B > 0:
    if torch_op_class in _HFTA_TORCH_IDENTICAL_OPS:
      return torch_op_class
    else:
      return functools.partial(_OPS_MAP[torch_op_class], B=B)
  else:
    return torch_op_class


def convert_ops(B, *torch_op_classes):
  return (get_hfta_op_for(op_class, B=B) for op_class in torch_op_classes)
