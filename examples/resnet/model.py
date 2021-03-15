import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
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
               track_running_stats=True,
               B=1):
    super(BasicBlock, self).__init__()
    if norm_layer is None:
      norm_layer = get_hfta_op_for(nn.BatchNorm2d, B)

    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride, B=B)
    self.bn1 = norm_layer(planes, track_running_stats=track_running_stats)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes, B=B)
    self.bn2 = norm_layer(planes, track_running_stats=track_running_stats)
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
               track_running_stats=True,
               B=1):
    super(Bottleneck, self).__init__()
    if norm_layer is None:
      norm_layer = get_hfta_op_for(nn.BatchNorm2d, B)
    # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv1x1(inplanes, planes, B=B)
    self.bn1 = norm_layer(planes, track_running_stats=track_running_stats)
    self.conv2 = conv3x3(planes, planes, stride, B=B)
    self.bn2 = norm_layer(planes, track_running_stats=track_running_stats)
    self.conv3 = conv1x1(planes, planes * self.expansion, B=B)
    self.bn3 = norm_layer(planes * self.expansion, track_running_stats=track_running_stats)
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
               planes,
               stride=1,
               downsample=None,
               norm_layer=None,
               track_running_stats=True,
               B=1):
    super(SerialBasicBlock, self).__init__()
    self.hfta = (B > 0)
    self.B = max(1, B)
    self.downsample = None
    self.unfused_parameters = []
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d

    self.conv1 = [
        conv3x3(inplanes, planes, stride, B=0) for _ in range(B)
    ]
    self.bn1 = [norm_layer(planes, track_running_stats=track_running_stats) for _ in range(B)]
    self.relu = [nn.ReLU(inplace=True) for _ in range(B)]
    self.conv2 = [conv3x3(planes, planes, B=0) for _ in range(B)]
    self.bn2 = [nn.BatchNorm2d(planes) for _ in range(B)]
    if downsample is not None:
      self.downsample = [copy.copy(downsample) for _ in range(B)]

    for i in range(self.B):
      param = []
      param.extend(list(self.conv1[i].parameters()))
      param.extend(list(self.conv2[i].parameters()))
      param.extend(list(self.bn1[i].parameters()))
      param.extend(list(self.bn2[i].parameters()))
      if self.downsample is not None:
        param.extend(list(self.downsample[i].parameters()))
      self.unfused_parameters.append(param)

    self.stride = stride

  def to(self, *args, **kwargs):
    for i in range(self.B):
      self.conv1[i].to(*args, **kwargs)
      self.conv2[i].to(*args, **kwargs)
      self.bn1[i].to(*args, **kwargs)
      self.bn2[i].to(*args, **kwargs)
      if self.downsample is not None:
        self.downsample[i].to(*args, **kwargs)

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
      out[i] += identity[i] if self.downsample is None else self.downsample[i](x[i])
    out = [self.relu[i](out[i]) for i in range(self.B)]

    if self.hfta:
      out = [out[i].unsqueeze(1) for i in range(self.B)]
      out = torch.cat(out, 1)
    else:
      out = out[0]

    return out


class ResNet(nn.Module):

  def __init__(self,
               block,
               layers,
               num_classes=10,
               zero_init_residual=False,
               track_running_stats=True,
               B=1):
    super(ResNet, self).__init__()
    self.B = B
    self.track_running_stats=track_running_stats
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
    self.bn1 = norm_layer(self.inplanes, track_running_stats=track_running_stats)
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
          norm_layer(planes * block.expansion, track_running_stats=self.track_running_stats),
      )

    layers = []
    layers.append(
        block(self.inplanes,
              planes,
              stride,
              downsample,
              norm_layer,
              track_running_stats=self.track_running_stats,
              B=B))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(self.inplanes,
                planes,
                norm_layer=norm_layer,
                track_running_stats=self.track_running_stats,
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


class SpecialFC(nn.Module):

  def __init__(self,
               C_in,
               C_out,
               B=1):
    super(SpecialFC, self).__init__()
    self.hfta = (B > 0)
    self.B = max(1, B)
    self.fc = [nn.Linear(C_in, C_out) for _ in range(B)]
    self.unfused_parameters = [list(self.fc[i].parameters()) for i in range(B)]

  def to(self, *args, **kwargs):
    for i in range(self.B):
      self.fc[i].to(*args, **kwargs)

  def forward(self, x):
    if not self.hfta:
      x = [x,]

    out = [self.fc[i](x[i]) for i in range(self.B)]

    if self.hfta:
      out = [out[i].unsqueeze(0) for i in range(self.B)]
      out = torch.cat(out, 0)
    else:
      out = out[0]

    return out

class SpecialConvBlock(nn.Module):

  def __init__(self, B, in_C, out_C):
    super(SpecialConvBlock, self).__init__()
    self.hfta = (B > 0)
    self.B = max(1, B)
    self.unfused_parameters = []
    self.conv = [nn.Conv2d(in_C, out_C, kernel_size=7, stride=2, padding=3, bias=False) for _ in range(B)]
    self.bn1 = [nn.BatchNorm2d(out_C) for _ in range(B)]
    self.relu = [nn.ReLU(inplace=True) for _ in range(B)]
    self.maxpool = [nn.MaxPool2d(kernel_size=3,stride=2,padding=1) for _ in range(B)]
    for i in range(B):
      self.unfused_parameters.append(list(self.conv[i].parameters()) + list(self.bn1[i].parameters()))

  def to(self, *args, **kwargs):
    for i in range(self.B):
      self.conv[i].to(*args, **kwargs)
      self.bn1[i].to(*args, **kwargs)
      self.relu[i].to(*args, **kwargs)
      self.maxpool[i].to(*args, **kwargs)

  def forward(self, x):
    if self.hfta:
      x = x.transpose(0, 1)
    else:
      x = [x,]

    out = [self.conv[i](x[i]) for i in range(self.B)]
    out = [self.bn1[i](out[i]) for i in range(self.B)]
    out = [self.relu[i](out[i]) for i in range(self.B)]
    out = [self.maxpool[i](out[i]) for i in range(self.B)]

    if self.hfta:
      out = [out[i].unsqueeze(1) for i in range(self.B)]
      out = torch.cat(out, 1)
    else:
      out = out[0]

    return out



class ResNetEnsemble(nn.Module):
  unfused_layers = []

  def __init__(self,
               config,
               block,
               serial_block,
               num_classes=10,
               zero_init_residual=False,
               track_running_stats=True,
               B=1):
    super(ResNetEnsemble, self).__init__()
    layers = config["layers"]
    run_in_serial = config["run_in_serial"]
    self.B = B
    self.track_running_stats = track_running_stats
    norm_layer = get_hfta_op_for(nn.BatchNorm2d, B)
    self._conv_layer = get_hfta_op_for(nn.Conv2d,
                                       B).func if B > 0 else nn.Conv2d
    self._norm_layer = get_hfta_op_for(nn.BatchNorm2d,
                                       B).func if B > 0 else nn.BatchNorm2d
    self._linear_layer = get_hfta_op_for(nn.Linear,
                                         B).func if B > 0 else nn.Linear

    self.inplanes = 64
    if run_in_serial[4][1]:
      self.convBlock = SpecialConvBlock(B, 3, self.inplanes)
      self.unfused_layers.append(self.convBlock)
    else:
      self.convBlock = nn.Sequential(
        get_hfta_op_for(nn.Conv2d, B=B)(3, self.inplanes, kernel_size=7,  stride=2, padding=3, bias=False),
        norm_layer(self.inplanes, track_running_stats=track_running_stats),
        nn.ReLU(inplace=True),
        get_hfta_op_for(nn.MaxPool2d, B=B)(kernel_size=3, stride=2, padding=1)
      )

    self.layer1 = self._make_layer(block, serial_block, 64, layers[0], run_in_serial[0], B=B)
    self.layer2 = self._make_layer(block, serial_block, 128, layers[1], run_in_serial[1], stride=2, B=B)
    self.layer3 = self._make_layer(block, serial_block, 256, layers[2], run_in_serial[2], stride=2, B=B)
    self.layer4 = self._make_layer(block, serial_block, 512, layers[3], run_in_serial[3], stride=2, B=B)
    if run_in_serial[4][0]:
      self.fc = SpecialFC(512 * block.expansion, num_classes, B=B)
      self.unfused_layers.append(self.fc)
    else:
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

  def _make_layer(self, block, serial_block, planes, blocks, run_in_serial, stride=1, B=1):
    downsample = None
    norm_layer = get_hfta_op_for(nn.BatchNorm2d, B)
    assert block.expansion == serial_block.expansion

    if stride != 1 or self.inplanes != planes * block.expansion:
      if self.B > 0 and run_in_serial[0]:
        downsample = nn.Sequential(
          conv1x1(self.inplanes, planes * block.expansion, stride, B=0),
          nn.BatchNorm2d(planes * block.expansion),
        )
      else:
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes * block.expansion, stride, B=B),
            norm_layer(planes * block.expansion, track_running_stats=self.track_running_stats),
        )

    layers = []
    if self.B > 0 and run_in_serial[0]:
      current_block = serial_block(self.inplanes, planes, stride, downsample,
                nn.BatchNorm2d, B=B, track_running_stats=self.track_running_stats)
      self.unfused_layers.append(current_block)
    else:
      current_block = block(self.inplanes, planes, stride, downsample,
                norm_layer, B=B, track_running_stats=self.track_running_stats)
    layers.append(current_block)

    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      if self.B > 0 and run_in_serial[i]:
        current_block = serial_block(self.inplanes, planes, norm_layer=nn.BatchNorm2d,
                B=B, track_running_stats=self.track_running_stats)
        self.unfused_layers.append(current_block)
      else:
        current_block = block(self.inplanes, planes, norm_layer=norm_layer,
                B=B, track_running_stats=self.track_running_stats)
      layers.append(current_block)

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.convBlock(x)

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

  def get_unfused_parameters(self):
    params = [[] for _ in range(self.B)]
    for layer in self.unfused_layers:
        params = [params[i] + layer.unfused_parameters[i] for i in range (self.B)]
    return params

  def unfused_to(self, *args, **kwargs):
      for layer in self.unfused_layers:
          layer.to(*args, **kwargs)

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

def Resnet18(**kwargs):
  return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
