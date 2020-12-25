import argparse
import os
import pandas as pd
import random
import time
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from dataset import ShapeNetDataset, ModelNetDataset
from model import PointNetCls, feature_transform_regularizer

try:
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.debug.metrics as met
except ImportError:
  pass

from hfta.ops import get_hfta_op_for
from hfta.optim import (get_hfta_optim_for, get_hfta_lr_scheduler_for,
                        consolidate_hyperparams_and_determine_B)
from hfta.workflow import EpochTimer


def seeding(seed):
  print("Random Seed: ", seed)
  random.seed(seed)
  torch.manual_seed(seed)


def build_dataset(args):
  if args.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=args.dataset,
        classification=True,
        npoints=args.num_points,
    )
    test_dataset = ShapeNetDataset(
        root=args.dataset,
        classification=True,
        split='test',
        npoints=args.num_points,
        data_augmentation=False,
    )
  elif args.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=args.dataset,
        npoints=args.num_points,
        split='trainval',
    )
    test_dataset = ModelNetDataset(
        root=args.dataset,
        split='test',
        npoints=args.num_points,
        data_augmentation=False,
    )
  else:
    exit('wrong dataset type')
  return dataset, test_dataset


def build_dataloader(args, dataset, test_dataset):
  dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=int(args.workers),
      drop_last=True,
  )
  testdataloader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=int(args.workers),
  )
  return dataloader, testdataloader


def main(args):
  blue = lambda x: '\033[94m' + x + '\033[0m'

  seeding(args.seed)

  if args.hfta:
    B = consolidate_hyperparams_and_determine_B(
        args,
        ['lr', 'beta1', 'beta2', 'weight_decay', 'gamma', 'step_size'],
    )
  else:
    B = 0
    (args.lr, args.beta1, args.beta2, args.weight_decay, args.gamma,
     args.step_size) = (args.lr[0], args.beta1[0], args.beta2[0],
                        args.weight_decay[0], args.gamma[0], args.step_size[0])

  if args.device == 'cuda':
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    print('Enable cuDNN heuristics!')
  device = (xm.xla_device()
            if args.device == 'xla' else torch.device(args.device))

  dataset, test_dataset = build_dataset(args)
  dataloader, testdataloader = build_dataloader(args, dataset, test_dataset)

  print('len(dataset)={}'.format(len(dataset)),
        'len(test_dataset)={}'.format(len(test_dataset)))
  num_classes = len(dataset.classes)
  print('classes', num_classes)

  if args.outf is not None:
    try:
      os.makedirs(args.outf)
    except OSError:
      pass

  classifier = PointNetCls(
      k=num_classes,
      feature_transform=args.feature_transform,
      B=B,
      track_running_stats=(args.device != 'xla'),
  )

  if args.model != '':
    classifier.load_state_dict(torch.load(args.model))

  optimizer = get_hfta_optim_for(optim.Adam, B=B)(
      classifier.parameters(),
      lr=args.lr,
      betas=(args.beta1, args.beta2),
      weight_decay=args.weight_decay,
  )
  scheduler = get_hfta_lr_scheduler_for(optim.lr_scheduler.StepLR, B=B)(
      optimizer,
      step_size=args.step_size,
      gamma=args.gamma,
  )

  scaler = amp.GradScaler(enabled=(args.device == 'cuda' and args.amp))

  classifier.to(device)

  num_batch = len(dataloader)

  def loss_fn(output, label, batch_size, trans_feat):
    if B > 0:
      loss = B * F.nll_loss(output.view(B * batch_size, -1), label)
    else:
      loss = F.nll_loss(output, label)
    if args.feature_transform:
      loss += feature_transform_regularizer(trans_feat) * 0.001
    return loss

  classifier = classifier.train()
  epoch_timer = EpochTimer()

  # Training loop
  for epoch in range(args.epochs):
    num_samples_per_epoch = 0
    epoch_timer.epoch_start(epoch)
    for i, data in enumerate(dataloader, 0):
      if i > args.iters_per_epoch:
        break
      if args.warmup_data_loading:
        continue

      points, target = data
      target = target[:, 0]
      points, target = points.to(device), target.to(device)
      N = points.size(0)
      if B > 0:
        points = points.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        target = target.repeat(B)
      optimizer.zero_grad()
      if args.device == 'cuda':
        with amp.autocast(enabled=args.amp):
          pred, trans, trans_feat = classifier(points)
          loss = loss_fn(pred, target, N, trans_feat)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
      else:
        pred, trans, trans_feat = classifier(points)
        loss = loss_fn(pred, target, N, trans_feat)
        loss.backward()
        if args.device == 'xla':
          xm.optimizer_step(optimizer, barrier=True)
        else:
          optimizer.step()

      print('[{}: {}/{}] train loss: {}'.format(epoch, i, num_batch,
                                                loss.item()))
      num_samples_per_epoch += N * max(B, 1)
      scaler.update()
    scheduler.step()
    epoch_timer.epoch_stop(num_samples_per_epoch)
    print('Epoch {} took {} s!'.format(epoch, epoch_timer.epoch_latency(epoch)))

  if args.device == 'xla' and not args.eval:
    print(met.metrics_report())
  if args.outf is not None:
    epoch_timer.to_csv(args.outf)

  if args.eval:
    # Run validation loop.
    print("Running validation loop ...")
    classifier = classifier.eval()
    with torch.no_grad():
      total_correct = torch.zeros(max(B, 1), device=device)
      total_testset = 0
      for data in testdataloader:
        if args.warmup_data_loading:
          continue
        points, target = data
        target = target[:, 0]
        points, target = points.to(device), target.to(device)
        N = points.size(0)
        if B > 0:
          points = points.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
          target = target.repeat(B)
        pred, _, _ = classifier(points)
        pred_choice = pred.argmax(-1)

        correct = pred_choice.eq(target.view(B, N) if B > 0 else target).sum(-1)

        total_correct.add_(correct)
        total_testset += N

      final_accuracy = total_correct / total_testset
      final_accuracy = final_accuracy.cpu().tolist()
      if args.outf is not None:
        pd.DataFrame({
            'acc': final_accuracy
        }).to_csv(os.path.join(args.outf, 'eval.csv'))

      # Return test_accuracy
      return final_accuracy


def attach_config_args(parser=argparse.ArgumentParser()):
  parser.add_argument(
      '--workers',
      type=int,
      help='number of data loading workers',
      default=4,
  )
  parser.add_argument(
      '--epochs',
      type=int,
      default=250,
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
      '--dataset_type',
      type=str,
      default='shapenet',
      help="dataset type shapenet|modelnet40",
  )
  parser.add_argument(
      '--num_points',
      type=int,
      default=2500,
      help='num of points for dataset',
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
      '--amp',
      default=False,
      action='store_true',
      help='Enable AMP; only used when --device is cuda',
  )
  parser.add_argument(
      '--eval',
      default=False,
      action='store_true',
      help='run the evaluation loop',
  )
  parser.add_argument(
      '--seed',
      type=int,
      help='Seed',
      default=1117,
  )
  parser.add_argument(
      '--warmup-data-loading',
      default=False,
      action='store_true',
      help='go over the training and validation loops without performing '
      'forward and backward passes',
  )
  return parser


def attach_fusible_args(parser=argparse.ArgumentParser()):
  # Adam settings:
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
  # StepLR settings:
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
  return parser


def attach_nonfusible_args(parser=argparse.ArgumentParser()):
  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='input batch size',
  )
  parser.add_argument(
      '--feature_transform',
      action='store_true',
      help="use feature transform",
  )
  return parser


def attach_args(parser=argparse.ArgumentParser()):
  attach_config_args(parser)
  attach_fusible_args(parser)
  attach_nonfusible_args(parser)
  return parser


if __name__ == "__main__":
  parser = attach_args()
  main(parser.parse_args())
