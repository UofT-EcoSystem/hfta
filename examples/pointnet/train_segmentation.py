import argparse
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from dataset import ShapeNetDataset
from model import PointNetDenseCls, feature_transform_regularizer

try:
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.debug.metrics as met
except ImportError:
  pass

from hfta.ops import get_hfta_op_for
from hfta.optim import (get_hfta_optim_for, get_hfta_lr_scheduler_for,
                        consolidate_hyperparams_and_determine_B)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize',
    type=int,
    default=32,
    help='input batch size',
)
parser.add_argument(
    '--workers',
    type=int,
    help='number of data loading workers',
    default=4,
)
parser.add_argument(
    '--epochs',
    type=int,
    default=25,
    help='number of epochs to train for',
)
parser.add_argument(
    '--iters-per-epoch',
    type=int,
    default=float('inf'),
    help='number of epochs to train for',
)
parser.add_argument('--outf', type=str, default=None, help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument(
    '--class_choice',
    type=str,
    default='Chair',
    help="class_choice",
)
parser.add_argument(
    '--feature_transform',
    action='store_true',
    help="use feature transform",
)
parser.add_argument(
    '--device',
    type=str,
    default='cuda',
    choices=['cpu', 'cuda', 'xla'],
    help="the device where this test is running",
)
parser.add_argument(
    '--hfta',
    default=False,
    action='store_true',
    help='use HFTA',
)
parser.add_argument(
    '--lr',
    type=float,
    default=[0.001],
    nargs='*',
    help='learning rate, default=0.001',
)
parser.add_argument(
    '--beta1',
    type=float,
    default=[0.9],
    nargs='*',
    help='beta1 coefficient (default: 0.9)',
)
parser.add_argument(
    '--beta2',
    type=float,
    default=[0.999],
    nargs='*',
    help='beta2 coefficient (default: 0.999)',
)
parser.add_argument(
    '--weight_decay',
    type=float,
    default=[0],
    nargs='*',
    help='weight_decay',
)
parser.add_argument(
    '--gamma',
    type=float,
    default=[0.5],
    nargs='*',
    help='learning rate decay, default=0.5',
)
parser.add_argument(
    '--step_size',
    type=int,
    default=[20],
    nargs='*',
    help='Period of learning rate decay',
)
parser.add_argument(
    '--amp',
    default=False,
    action='store_true',
    help='Enable AMP; only used when --device is cuda',
)

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# TODO: make it better
if opt.hfta:
  B = consolidate_hyperparams_and_determine_B(
      opt,
      ['lr', 'beta1', 'beta2', 'weight_decay', 'gamma', 'step_size'],
  )
else:
  B = 0
  (opt.lr, opt.beta1, opt.beta2, opt.weight_decay, opt.gamma,
   opt.step_size) = (opt.lr[0], opt.beta1[0], opt.beta2[0], opt.weight_decay[0],
                     opt.gamma[0], opt.step_size[0])

if opt.device == 'cuda':
  assert torch.cuda.is_available()
  torch.backends.cudnn.benchmark = True
  print('Enable cuDNN heuristics!')
device = xm.xla_device() if opt.device == 'xla' else torch.device(opt.device)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
)

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False,
)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
)

print('len(dataset)={}'.format(len(dataset)),
      'len(test_dataset)={}'.format(len(test_dataset)))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
if opt.outf is not None:
  try:
    os.makedirs(opt.outf)
  except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(
    k=num_classes,
    feature_transform=opt.feature_transform,
    B=B,
    track_running_stats=(opt.device != 'xla'),
)

if opt.model != '':
  classifier.load_state_dict(torch.load(opt.model))

optimizer = get_hfta_optim_for(optim.Adam, B=B)(
    classifier.parameters(),
    lr=opt.lr,
    betas=(opt.beta1, opt.beta2),
    weight_decay=opt.weight_decay,
)
scheduler = get_hfta_lr_scheduler_for(optim.lr_scheduler.StepLR, B=B)(
    optimizer,
    step_size=opt.step_size,
    gamma=opt.gamma,
)

if opt.device == 'cuda' and opt.amp:
  scaler = amp.GradScaler()

classifier.to(device)

num_batch = len(dataloader)


def loss_fn(output, label, batch_size, trans_feat, num_classes):
  loss = F.nll_loss(output.view(-1, num_classes), label.flatten())
  if B > 0:
    loss *= B
  if opt.feature_transform:
    loss += feature_transform_regularizer(trans_feat) * 0.001
  return loss


classifier = classifier.train()
timing = {'epoch': [], 'epoch_start': [], 'epoch_stop': []}
for epoch in range(opt.epochs):
  timing['epoch'].append(epoch)
  timing['epoch_start'].append(time.time())
  for i, data in enumerate(dataloader, 0):
    if i > opt.iters_per_epoch:
      break
    points, target = data
    points, target = points.to(device), target.to(device) - 1
    N = points.size(0)
    if B > 0:
      points = points.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
      target = target.unsqueeze(0).expand(B, -1, -1)
    optimizer.zero_grad()
    if opt.device == 'cuda':
      with amp.autocast(enabled=opt.amp):
        pred, trans, trans_feat = classifier(points)
        loss = loss_fn(pred, target, N, trans_feat, num_classes)
      if opt.amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
      else:
        loss.backward()
        optimizer.step()
    else:
      pred, trans, trans_feat = classifier(points)
      loss = loss_fn(pred, target, N, trans_feat, num_classes)
      loss.backward()
      if opt.device == 'xla':
        xm.optimizer_step(optimizer, barrier=True)
      else:
        optimizer.step()

    print('[{}: {}/{}] train loss: {}'.format(epoch, i, num_batch, loss.item()))

    if opt.device == 'cuda' and opt.amp:
      scaler.update()
  scheduler.step()
  timing['epoch_stop'].append(time.time())
  print('Epoch {} took {} s!'.format(
      epoch,
      timing['epoch_stop'][-1] - timing['epoch_start'][-1],
  ))
  #torch.save(classifier.state_dict(),
  #           '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))
if opt.device == 'xla':
  print(met.metrics_report())
pd.DataFrame(timing).to_csv('{}/timing.csv'.format(opt.outf))
