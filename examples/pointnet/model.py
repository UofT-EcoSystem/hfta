from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from hfta.ops import get_hfta_op_for


def convert_ops(B, *torch_op_classes):
  return (get_hfta_op_for(op_class, B=B) for op_class in torch_op_classes)


class STN3d(nn.Module):

  def __init__(self, B=0, track_running_stats=True):
    super(STN3d, self).__init__()
    self.B = B
    Linear, ReLU, BatchNorm1d = convert_ops(B, nn.Linear, nn.ReLU,
                                            nn.BatchNorm1d)
    self.conv1 = Linear(3, 64)
    self.conv2 = Linear(64, 128)
    self.conv3 = Linear(128, 1024)
    self.fc1 = Linear(1024, 512)
    self.fc2 = Linear(512, 256)
    self.fc3 = Linear(256, 9)
    self.relu = ReLU()

    self.bn1 = BatchNorm1d(64, track_running_stats=track_running_stats)
    self.bn2 = BatchNorm1d(128, track_running_stats=track_running_stats)
    self.bn3 = BatchNorm1d(1024, track_running_stats=track_running_stats)
    self.bn4 = BatchNorm1d(512, track_running_stats=track_running_stats)
    self.bn5 = BatchNorm1d(256, track_running_stats=track_running_stats)

  def forward(self, x):
    """
    Input: [N, num_points, 3] or [B, N, num_points, 3]
    Output: [N, 3, 3] or [B, N, 3, 3]
    """
    B, batchsize, n_pts = self.B, x.size(-3), x.size(-2)
    x = x.view((B, batchsize * n_pts, -1) if B > 0 else (batchsize * n_pts, -1))
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.relu(self.bn3(self.conv3(x)))
    x = x.view((B, batchsize, n_pts, -1) if B > 0 else (batchsize, n_pts, -1))
    x, _ = torch.max(x, -2)

    x = self.relu(self.bn4(self.fc1(x)))
    x = self.relu(self.bn5(self.fc2(x)))
    x = self.fc3(x)
    x = x.view((B, batchsize, 3, 3) if B > 0 else (batchsize, 3, 3))

    x = x + torch.eye(
        3,
        dtype=x.dtype,
        device=x.device,
    ).view((1, 1, 3, 3) if self.B > 0 else (1, 3, 3))

    return x


class STNkd(nn.Module):

  def __init__(self, k=64, B=0, track_running_stats=True):
    super(STNkd, self).__init__()
    self.B = B
    Linear, ReLU, BatchNorm1d = convert_ops(B, nn.Linear, nn.ReLU,
                                            nn.BatchNorm1d)
    self.conv1 = Linear(k, 64)
    self.conv2 = Linear(64, 128)
    self.conv3 = Linear(128, 1024)
    self.fc1 = Linear(1024, 512)
    self.fc2 = Linear(512, 256)
    self.fc3 = Linear(256, k * k)
    self.relu = ReLU()

    self.bn1 = BatchNorm1d(64, track_running_stats=track_running_stats)
    self.bn2 = BatchNorm1d(128, track_running_stats=track_running_stats)
    self.bn3 = BatchNorm1d(1024, track_running_stats=track_running_stats)
    self.bn4 = BatchNorm1d(512, track_running_stats=track_running_stats)
    self.bn5 = BatchNorm1d(256, track_running_stats=track_running_stats)

    self.k = k

  def forward(self, x):
    """
    Input: [N, num_points, self.k] or [B, N, num_points, self.k]
    Output: [N, self.k, self.k] or [B, N, self.k, self.k]
    """
    B, batchsize, n_pts, k = self.B, x.size(-3), x.size(-2), self.k
    x = x.view((B, batchsize * n_pts, -1) if B > 0 else (batchsize * n_pts, -1))
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.relu(self.bn3(self.conv3(x)))
    x = x.view((B, batchsize, n_pts, -1) if B > 0 else (batchsize, n_pts, -1))
    x, _ = torch.max(x, -2)

    x = F.relu(self.bn4(self.fc1(x)))
    x = F.relu(self.bn5(self.fc2(x)))
    x = self.fc3(x)
    x = x.view((self.B, batchsize, k, k) if self.B > 0 else (batchsize, k, k))

    x = x + torch.eye(
        k,
        dtype=x.dtype,
        device=x.device,
    ).view((1, 1, k, k) if self.B > 0 else (1, k, k))

    return x


class PointNetfeat(nn.Module):

  def __init__(
      self,
      global_feat=True,
      feature_transform=False,
      B=0,
      track_running_stats=True,
  ):
    super(PointNetfeat, self).__init__()
    self.B = B
    Linear, ReLU, BatchNorm1d = convert_ops(B, nn.Linear, nn.ReLU,
                                            nn.BatchNorm1d)
    self.stn = STN3d(B=B, track_running_stats=track_running_stats)
    self.conv1 = Linear(3, 64)
    self.conv2 = Linear(64, 128)
    self.conv3 = Linear(128, 1024)
    self.bn1 = BatchNorm1d(64, track_running_stats=track_running_stats)
    self.bn2 = BatchNorm1d(128, track_running_stats=track_running_stats)
    self.bn3 = BatchNorm1d(1024, track_running_stats=track_running_stats)
    self.relu = ReLU()
    self.global_feat = global_feat
    self.feature_transform = feature_transform
    if self.feature_transform:
      self.fstn = STNkd(k=64, B=B, track_running_stats=track_running_stats)

  def forward(self, x):
    """
    Input: [N, num_points, 3] or [B, N, num_points, 3]
    Output:
      x: if self.global_feat: [N, 1024] or [B, N, 1024]
         else: [N, num_points, 1088] or [B, N, num_points, 1088]
      trans: [N, 3, 3] or [B, N, 3, 3]
      trans_feat: [N, 64, 64] or [B, N, 64, 64]
    """
    B, batchsize, n_pts = self.B, x.size(-3), x.size(-2)
    trans = self.stn(x)  # [N, 3, 3] or [B, N, 3, 3]
    x = torch.matmul(x, trans)
    x = x.view((B, batchsize * n_pts, -1) if B > 0 else (batchsize * n_pts, -1))
    x = self.relu(self.bn1(self.conv1(x)))
    x = x.view((B, batchsize, n_pts, -1) if B > 0 else (batchsize, n_pts, -1))

    if self.feature_transform:
      trans_feat = self.fstn(x)  # [N, 64, 64] or [B, N, 64, 64]
      x = torch.matmul(x, trans_feat)
    else:
      trans_feat = None

    pointfeat = x
    x = x.view((B, batchsize * n_pts, -1) if B > 0 else (batchsize * n_pts, -1))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.bn3(self.conv3(x))
    x = x.view((B, batchsize, n_pts, -1) if B > 0 else (batchsize, n_pts, -1))
    x, _ = torch.max(x, -2)
    if self.global_feat:
      return x, trans, trans_feat
    else:
      x = x.unsqueeze(-2).repeat((1, 1, n_pts, 1) if B > 0 else (1, n_pts, 1))
      return torch.cat([x, pointfeat], -1), trans, trans_feat


class PointNetCls(nn.Module):

  def __init__(
      self,
      k=2,
      feature_transform=False,
      B=0,
      track_running_stats=True,
  ):
    super(PointNetCls, self).__init__()
    self.B = B
    Linear, ReLU, Dropout, BatchNorm1d = convert_ops(B, nn.Linear, nn.ReLU,
                                                     nn.Dropout, nn.BatchNorm1d)
    self.feature_transform = feature_transform
    self.feat = PointNetfeat(
        global_feat=True,
        feature_transform=feature_transform,
        B=B,
        track_running_stats=track_running_stats,
    )
    self.fc1 = Linear(1024, 512)
    self.fc2 = Linear(512, 256)
    self.fc3 = Linear(256, k)
    self.dropout = Dropout(p=0.3)
    self.bn1 = BatchNorm1d(512, track_running_stats=track_running_stats)
    self.bn2 = BatchNorm1d(256, track_running_stats=track_running_stats)
    self.relu = ReLU()

  def forward(self, x):
    """
    Input: [N, num_points, 3] or [B, N, num_points, 3] if B > 0
    Output:
      scores: [N, num_classes] or [B, N, num_classes] (i.e., k == num_classes)
      trans: [B, N, 3, 3]
      trans_feat: [B, N, 64, 64]
    """
    x, trans, trans_feat = self.feat(x)
    x = self.relu(self.bn1(self.fc1(x)))
    x = self.relu(self.bn2(self.dropout(self.fc2(x))))
    x = self.fc3(x)
    return F.log_softmax(x, dim=-1), trans, trans_feat


class PointNetDenseCls(nn.Module):

  def __init__(
      self,
      k=2,
      feature_transform=False,
      B=0,
      track_running_stats=True,
  ):
    super(PointNetDenseCls, self).__init__()
    self.B = B
    Linear, ReLU, BatchNorm1d = convert_ops(B, nn.Linear, nn.ReLU,
                                            nn.BatchNorm1d)
    self.k = k
    self.feature_transform = feature_transform
    self.feat = PointNetfeat(
        global_feat=False,
        feature_transform=feature_transform,
        B=B,
        track_running_stats=track_running_stats,
    )
    self.conv1 = Linear(1088, 512)
    self.conv2 = Linear(512, 256)
    self.conv3 = Linear(256, 128)
    self.conv4 = Linear(128, self.k)
    self.bn1 = BatchNorm1d(512, track_running_stats=track_running_stats)
    self.bn2 = BatchNorm1d(256, track_running_stats=track_running_stats)
    self.bn3 = BatchNorm1d(128, track_running_stats=track_running_stats)
    self.relu = ReLU()

  def forward(self, x):
    """
    Input: [N, num_points, 3] or [B, N, num_points, 3]
    Output: (self.k == num_classes)
      scores: [N, num_points, num_classes] or [B, N, num_points, num_classes]
      trans: [N, 3, 3] or [B, N, 3, 3]
      trans_feat: [N, 64, 64] or [B, N, 64, 64]
    """
    B, batchsize, n_pts = self.B, x.size(-3), x.size(-2)
    x, trans, trans_feat = self.feat(x)
    x = x.view((B, batchsize * n_pts, -1) if B > 0 else (batchsize * n_pts, -1))
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.conv4(x)
    x = x.view((B, batchsize, n_pts, -1) if B > 0 else (batchsize, n_pts, -1))
    x = F.log_softmax(x, dim=-1)

    return x, trans, trans_feat


def feature_transform_regularizer(trans):
  """
  Input: [N, d, d] or [B, N, d, d]; d is either 3 or 64
  """
  batchsize, d = trans.size(-3), trans.size(-2)
  B = trans.size(0) if trans.dim() > 3 else 0
  I = torch.eye(d, dtype=trans.dtype, device=trans.device).unsqueeze(0)
  if B > 0:
    I = I.unsqueeze(0)
  loss = torch.mean(
      torch.norm(
          torch.matmul(trans, trans.transpose(-2, -1)) - I,
          dim=(-1, -2),
      ))
  if B > 0:
    loss *= B
  return loss


if __name__ == '__main__':

  # Change the value of B to try different setup
  B = 5

  sim_data = torch.rand(32, 2500, 3) if B == 0 else torch.rand(B, 32, 2500, 3)
  trans = STN3d(B=B)
  out = trans(sim_data)
  print('stn', out.size())
  print('loss', feature_transform_regularizer(out))

  sim_data_64d = torch.rand(32, 2500, 64) if B == 0 else torch.rand(
      B, 32, 2500, 64)
  trans = STNkd(k=64, B=B)
  out = trans(sim_data_64d)
  print('stn64d', out.size())
  print('loss', feature_transform_regularizer(out))

  pointfeat = PointNetfeat(global_feat=True, B=B)
  out, _, _ = pointfeat(sim_data)
  print('global feat', out.size())

  pointfeat = PointNetfeat(global_feat=False, B=B)
  out, _, _ = pointfeat(sim_data)
  print('point feat', out.size())

  cls = PointNetCls(k=5, B=B)
  out, _, _ = cls(sim_data)
  print('class', out.size())

  seg = PointNetDenseCls(k=3, B=B)
  out, _, _ = seg(sim_data)
  print('seg', out.size())
