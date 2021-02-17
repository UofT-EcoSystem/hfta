import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from hfta.ops import get_hfta_op_for


def str_to_class(classname):
  return getattr(sys.modules[__name__], classname)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, B=1):
  """3x3 convolution with padding"""
  return get_hfta_op_for(nn.Conv2d, B)(
      in_planes,
      out_planes,
      kernel_size=3,
      stride=stride,
      padding=dilation,
      groups=groups,
      bias=False,
      dilation=dilation,
  )


def conv1x1(in_planes, out_planes, stride=1, B=1):
  """1x1 convolution"""
  return get_hfta_op_for(nn.Conv2d, B)(
      in_planes,
      out_planes,
      kernel_size=1,
      stride=stride,
      bias=False,
  )


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               downsample=None,
               norm_layer=None,
               B=1):
    super(BasicBlock, self).__init__()
    if norm_layer is None:
      norm_layer = get_hfta_op_for(nn.BatchNorm2d, B)

    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride, B=B)
    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes, B=B)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

  def snatch_parameters(self, other, b):
    self.conv1.snatch_parameters(other.conv1, b)
    self.bn1.snatch_parameters(other.bn1, b)
    self.conv2.snatch_parameters(other.conv2, b)
    self.bn2.snatch_parameters(other.bn2, b)
    if self.downsample is not None:
      sequence_snatch_parameters(self.downsample, other.downsample, b)



class Bottleneck(nn.Module):
  # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
  # while original implementation places the stride at the first 1x1 convolution(self.conv1)
  # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
  # This variant is also known as ResNet V1.5 and improves accuracy according to
  # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

  expansion: int = 4

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               downsample=None,
               norm_layer=None,
               B=1):
    super(Bottleneck, self).__init__()
    if norm_layer is None:
      norm_layer = get_hfta_op_for(nn.BatchNorm2d, B)
    # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv1x1(inplanes, planes, B=B)
    self.bn1 = norm_layer(planes)
    self.conv2 = conv3x3(planes, planes, stride, B=B)
    self.bn2 = norm_layer(planes)
    self.conv3 = conv1x1(planes, planes * self.expansion, B=B)
    self.bn3 = norm_layer(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)
    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


  def snatch_parameters(self, other, b):
    self.conv1.snatch_parameters(other.conv1, b)
    self.bn1.snatch_parameters(other.bn1, b)
    self.conv2.snatch_parameters(other.conv2, b)
    self.bn2.snatch_parameters(other.bn2, b)
    self.conv3.snatch_parameters(other.conv2, b)
    self.bn3.snatch_parameters(other.bn2, b)
    if self.downsample is not None:
      sequence_snatch_parameters(self.downsample, other.downsample, b)

class SerialBasicBlock(nn.Module):
  expansion = 1

  def __init__(self,
               inplanes,
               serial_widths,
               outplanes,
               stride=1,
               norm_layer=None,
               B=1):
    super(SerialBasicBlock, self).__init__()
    self.hfta = (B > 0)
    self.B = max(1, B)
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d

    assert len(serial_widths) == self.B

    self.conv1 = [
        conv3x3(inplanes, width, stride, B=0) for width in serial_widths
    ]
    self.bn1 = [norm_layer(width) for width in serial_widths]
    self.relu = [nn.ReLU(inplace=True) for _ in serial_widths]
    self.conv2 = [conv3x3(width, outplanes, B=0) for width in serial_widths]
    self.bn2 = [nn.BatchNorm2d(outplanes) for _ in serial_widths]

    # TODO it makes optim of HFTA get an error, must use torch optim instead
    for i in range(self.B):
      self.add_module("conv1_%d" % i, self.conv1[i])
      self.add_module("conv2_%d" % i, self.conv2[i])
      self.add_module("bn1_%d" % i, self.bn1[i])
      self.add_module("bn2_%d" % i, self.bn2[i])

    self.stride = stride

  def forward(self, x):
    if self.hfta:
      x = x.transpose(0, 1)
    else:
      x = [x,]
    identity = x
    out = [self.conv1[i](x[i]) for i in range(self.B)]

    out = [self.bn1[i](out[i]) for i in range(self.B)]
    out = [self.relu[i](out[i]) for i in range(self.B)]

    out = [self.conv2[i](out[i]) for i in range(self.B)]
    out = [self.bn2[i](out[i]) for i in range(self.B)]

    for i in range(self.B):
      out[i] += identity[i]
    out = [self.relu[i](out[i]) for i in range(self.B)]

    if self.hfta:
      out = [out[i].unsqueeze(1) for i in range(self.B)]
      out = torch.cat(out, 1)
    else:
      out = out[0]

    return out

  # def snatch_parameters(self, other, b):
  #   self.conv1.snatch_parameters(other.conv1, b)
  #   self.bn1.snatch_parameters(other.bn1, b)
  #   self.conv2.snatch_parameters(other.conv2, b)
  #   self.bn2.snatch_parameters(other.bn2, b)
  #   if self.downsample is not None:
  #     sequence_snatch_parameters(self.downsample, other.downsample, b)



class ResNet(nn.Module):

  def __init__(self,
               block,
               layers,
               num_classes=10,
               zero_init_residual=False,
               B=1):
    super(ResNet, self).__init__()
    self.B = B
    norm_layer = get_hfta_op_for(nn.BatchNorm2d, B)
    self._conv_layer = get_hfta_op_for(nn.Conv2d,
                                       B).func if B > 0 else nn.Conv2d
    self._norm_layer = get_hfta_op_for(nn.BatchNorm2d,
                                       B).func if B > 0 else nn.BatchNorm2d
    self._linear_layer = get_hfta_op_for(nn.Linear,
                                         B).func if B > 0 else nn.Linear

    self.inplanes = 64

    self.conv1 = get_hfta_op_for(nn.Conv2d, B=B)(3,
                                                 self.inplanes,
                                                 kernel_size=7,
                                                 stride=2,
                                                 padding=3,
                                                 bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = get_hfta_op_for(nn.MaxPool2d, B=B)(kernel_size=3,
                                                      stride=2,
                                                      padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0], B=B)
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, B=B)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, B=B)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2, B=B)
    self.fc = get_hfta_op_for(nn.Linear, B)(512 * block.expansion, num_classes)
    for m in self.modules():
      if isinstance(m, self._conv_layer):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, self._norm_layer):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self, block, planes, blocks, stride=1, B=1):
    downsample = None
    norm_layer = get_hfta_op_for(nn.BatchNorm2d, B)

    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes * block.expansion, stride, B=B),
          norm_layer(planes * block.expansion),
      )

    layers = []
    layers.append(
        block(self.inplanes,
              planes,
              stride,
              downsample,
              norm_layer,
              B=B))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(self.inplanes,
                planes,
                norm_layer=norm_layer,
                B=B))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    if self.B > 0:
      x = torch.flatten(x, 2)
      x = x.transpose(0, 1)
    else:
      x = torch.flatten(x, 1)
    x = self.fc(x)
    if self.B > 0:
      output = F.log_softmax(x, dim=2)
    else:
      output = F.log_softmax(x, dim=1)
    return output

  def snatch_parameters(self, others, b):
    self.conv1.snatch_parameters(others.conv1, b)
    self.bn1.snatch_parameters(others.bn1, b)
    sequence_snatch_parameters(self.layer1, others.layer1, b)
    sequence_snatch_parameters(self.layer2, others.layer2, b)
    sequence_snatch_parameters(self.layer3, others.layer3, b)
    sequence_snatch_parameters(self.layer4, others.layer4, b)
    self.fc.snatch_parameters(others.fc, b)

  def init_load(self, file_names):
    if self.B == 0:
      self.load_state_dict(torch.load(file_names[0]).state_dict())
    else:
      assert self.B == len(file_names)
      for i, file_name in enumerate(file_names):
        others = torch.load(file_name)
        self.snatch_parameters(others, i)


class ResNetEnsemble(nn.Module):

  def __init__(self,
               block,
               config,
               num_classes=10,
               zero_init_residual=False,
               B=1):
    super(ResNetEnsemble, self).__init__()
    layers = config["layers"]
    serial_layers = config["append_serial"]
    self.B = B
    norm_layer = get_hfta_op_for(nn.BatchNorm2d, B)
    self._conv_layer = get_hfta_op_for(nn.Conv2d,
                                       B).func if B > 0 else nn.Conv2d
    self._norm_layer = get_hfta_op_for(nn.BatchNorm2d,
                                       B).func if B > 0 else nn.BatchNorm2d
    self._linear_layer = get_hfta_op_for(nn.Linear,
                                         B).func if B > 0 else nn.Linear

    self.inplanes = 64

    self.conv1 = get_hfta_op_for(nn.Conv2d, B=B)(3,
                                                 self.inplanes,
                                                 kernel_size=7,
                                                 stride=2,
                                                 padding=3,
                                                 bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = get_hfta_op_for(nn.MaxPool2d, B=B)(kernel_size=3,
                                                      stride=2,
                                                      padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0], serial_layers[0], B=B)
    self.layer2 = self._make_layer(block, 128, layers[1], serial_layers[1], stride=2, B=B)
    self.layer3 = self._make_layer(block,  256, layers[2], serial_layers[2], stride=2, B=B)
    self.layer4 = self._make_layer(block, 512, layers[3], serial_layers[3], stride=2, B=B)
    self.fc = get_hfta_op_for(nn.Linear, B)(512 * block.expansion, num_classes)
    for m in self.modules():
      if isinstance(m, self._conv_layer):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, self._norm_layer):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self, block, planes, blocks, append_serial, stride=1, B=1):
    downsample = None
    norm_layer = get_hfta_op_for(nn.BatchNorm2d, B)

    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes * block.expansion, stride, B=B),
          norm_layer(planes * block.expansion),
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, norm_layer, B=B))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, norm_layer=norm_layer, B=B))
    if append_serial is not None:
      print(append_serial)
      layers.append(SerialBasicBlock(self.inplanes, append_serial, planes * block.expansion, norm_layer=norm_layer, B=B))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    if self.B > 0:
      x = torch.flatten(x, 2)
      x = x.transpose(0, 1)
    else:
      x = torch.flatten(x, 1)
    x = self.fc(x)
    if self.B > 0:
      output = F.log_softmax(x, dim=2)
    else:
      output = F.log_softmax(x, dim=1)
    return output

  def snatch_parameters(self, others, b):
    self.conv1.snatch_parameters(others.conv1, b)
    self.bn1.snatch_parameters(others.bn1, b)
    sequence_snatch_parameters(self.layer1, others.layer1, b)
    sequence_snatch_parameters(self.layer2, others.layer2, b)
    sequence_snatch_parameters(self.layer3, others.layer3, b)
    sequence_snatch_parameters(self.layer4, others.layer4, b)
    self.fc.snatch_parameters(others.fc, b)

  def init_load(self, file_names):
    if self.B == 0:
      self.load_state_dict(torch.load(file_names[0]).state_dict())
    else:
      assert self.B == len(file_names)
      for i, file_name in enumerate(file_names):
        others = torch.load(file_name)
        self.snatch_parameters(others, i)


def sequence_snatch_parameters(seq: nn.Sequential, others: nn.Sequential, b):
  others_dict = {}
  for name, layer in others.named_children():
    others_dict[name] = layer
  for name, layer in seq.named_children():
    others_layer = others_dict[name]
    if isinstance(layer, nn.Sequential):
      sequence_snatch_parameters(layer, others_layer, b)
    else:
      layer.snatch_parameters(others_layer, b)


def ResnetEnsembleModel(config, block, **kwargs):
  return ResNetEnsemble(block, config, **kwargs)

def Resnet18(**kwargs):
  return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)