import argparse
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from v2 import mobilenet_v2
from v3 import mobilenetv3_small, mobilenetv3_large

try:
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.debug.metrics as met
except ImportError:
  pass

import sys
from hfta.optim import (get_hfta_optim_for, get_hfta_lr_scheduler_for,
                        consolidate_hyperparams_and_determine_B)

from hfta.workflow import EpochTimer


def attach_args(parser=argparse.ArgumentParser(description='MobileNet V2 and V3 Example')):
  # Training settings
  parser.add_argument(
      '--version',
      type=str,
      default='v2',
      choices=['v2', 'v3s', 'v3l'],
      help='version of the MobileNet (default: v2)',
  )
  parser.add_argument(
      '--batch-size',
      type=int,
      default=256,
      help='input batch size for training (default: 64)',
  )
  parser.add_argument(
      '--epochs',
      type=int,
      default=14,
      help='number of epochs to train (default: 14)',
  )
  parser.add_argument(
      '--iters-per-epoch',
      type=int,
      default=float('inf'),
      help='number of epochs to train for',
  )
  parser.add_argument(
      '--lr',
      type=float,
      default=[0.01],
      nargs='*',
      help='learning rate (default: 0.01)',
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
      '--device',
      type=str,
      default='cuda',
      choices=['cpu', 'cuda', 'xla'],
      help='using cpu, cuda or tpu. (default: cuda)',
  )
  parser.add_argument(
      '--seed',
      type=int,
      default=13452,
      help='random seed (default: 1)',
  )
  parser.add_argument(
      '--outf',
      default=None,
      type=str,
      help='folder to dump output',
  )
  parser.add_argument(
      '--dataroot',
      type=str,
      default='../../datasets/cifar10',
      help='folder that stores input dataset',
  )
  parser.add_argument(
      '--dataset',
      type=str,
      default='cifar10',
      choices=['cifar10', 'imagenet'],
      help='dataset type',
  )
  parser.add_argument(
      '--num-workers',
      type=int,
      default=2,
      help='number of DataLoader workers (default: 2)',
  )
  parser.add_argument(
      '--hfta',
      action="store_true",
      default=False,
      help='using HFTA to run',
  )
  parser.add_argument(
      '--amp',
      action="store_true",
      default=False,
      help='For Saving the current Model',
  )
  parser.add_argument(
      '--eval',
      default=False,
      action='store_true',
      help='run the evaluation loop',
  )
  parser.add_argument(
      '--warmup-data-loading',
      default=False,
      action='store_true',
      help='go over the training and validation loops without performing '
      'forward and backward passes',
  )
  return parser


def _seeding(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)


def _mkdir_outf(args):
  if args.outf is None:
    return
  try:
    os.makedirs(args.outf)
  except OSError:
    pass


def _create_device_handle(args):
  if args.device == 'cuda':
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    print('Enable cuDNN heuristics!')
  return (torch.device(args.device)
          if args.device in {'cpu', 'cuda'} else xm.xla_device())


def _create_scaler(args):
  if args.device == 'cuda' and args.amp:
    return amp.GradScaler()
  else:
    return None


def _create_dataloaders(args):
  if args.dataset == 'cifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.CIFAR10(
        args.dataroot,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        args.dataroot,
        train=False,
        download=True,
        transform=transform,
    )
    num_classes = 10
  elif args.dataset == 'imagenet':
    # Data loading code
    traindir = os.path.join(args.dataroot, 'train')
    valdir = os.path.join(args.dataroot, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    )
    test_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
    )
    num_classes = 1000
  else:
    raise ValueError('Invalid dataset: {} !'.format(args.dataset))
  kwargs = {
      'batch_size': args.batch_size,
      'num_workers': args.num_workers,
      'pin_memory': True,
      'shuffle': True,
  }
  train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
  test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
  return train_loader, test_loader, num_classes


def _loss_fn(criterion, outputs, labels, B, batch_size):
  if B > 0:
    loss = B * criterion(outputs.view(B * batch_size, -1), labels)
  else:
    loss = criterion(outputs, labels)
  return loss


def train(args, model, criterion, optimizer, scaler, device, train_loader,
          epoch, B):
  model.train()
  num_samples_done = 0
  B_real = max(B, 1)
  for batch_idx, (inputs, labels) in enumerate(train_loader):
    if batch_idx >= args.iters_per_epoch:
      break
    if args.warmup_data_loading:
      continue

    optimizer.zero_grad()

    inputs, labels = inputs.to(device), labels.to(device)
    batch_size = inputs.size(0)
    if B > 0:
      inputs = inputs.unsqueeze(1).expand(-1, B, -1, -1, -1).contiguous()
      labels = labels.repeat(B)

    if args.device == 'cuda':
      with amp.autocast(enabled=args.amp):
        outputs = model(inputs)
        loss = _loss_fn(criterion, outputs, labels, B, batch_size)
    else:
      outputs = model(inputs)
      loss = _loss_fn(criterion, outputs, labels, B, batch_size)

    if scaler is not None:
      scaler.scale(loss).backward()
    else:
      loss.backward()

    if args.device == 'xla':
      xm.optimizer_step(optimizer, barrier=True)
    else:
      if scaler is not None:
        scaler.step(optimizer)
      else:
        optimizer.step()

    if scaler is not None:
      scaler.update()

    num_samples_done += batch_size * B_real

  return num_samples_done


def test(args, model, device, test_loader, B):
  print('Running validation loop ...')
  model.eval()
  with torch.no_grad():
    total_correct_1 = torch.zeros(max(B, 1), device=device)
    total_correct_5 = torch.zeros(max(B, 1), device=device)
    total_samples = 0
    for inputs, labels in test_loader:
      if args.warmup_data_loading:
        continue
      inputs, labels = inputs.to(device), labels.to(device)
      batch_size = inputs.size(0)
      if B > 0:
        inputs = inputs.unsqueeze(1).expand(-1, B, -1, -1, -1).contiguous()
        labels = labels.repeat(B).view(B, batch_size)
      outputs = model(inputs)
      # pred: [N, maxk] or [B, N, maxk]
      _, pred = outputs.topk(5, dim=-1, largest=True, sorted=True)
      pred = pred.transpose(-2, -1)  # pred: [maxk, N] or [B, maxk, N]
      labels = labels.unsqueeze(-2)  # labels: [1, N] or [B, 1, N]
      correct = pred.eq(labels)  # correct: [maxk, N] or [B, maxk, N]
      # top-1 accuracy:
      correct_1 = (correct[:, :1] if B > 0 else correct[:1]).sum((-2, -1))
      correct_5 = (correct[:, :5] if B > 0 else correct[:5]).sum((-2, -1))
      total_correct_1.add_(correct_1)
      total_correct_5.add_(correct_5)
      total_samples += batch_size
    final_accuracy_1 = total_correct_1 / total_samples
    final_accuracy_5 = total_correct_5 / total_samples
    return final_accuracy_1.cpu().tolist(), final_accuracy_5.cpu().tolist()


def _get_model_constructor(args):
  if args.version == 'v2':
    return mobilenet_v2
  elif args.version == 'v3s':
    return mobilenetv3_small
  elif args.version == 'v3l':
    return mobilenetv3_large
  else:
    raise ValueError('Invalid model version: {} !'.format(args.version))


def main(args):
  _seeding(args)

  _mkdir_outf(args)

  device = _create_device_handle(args)

  scaler = _create_scaler(args)

  train_loader, test_loader, num_classes = _create_dataloaders(args)

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

  model = _get_model_constructor(args)(num_classes=num_classes, B=B).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = get_hfta_optim_for(optim.Adam, B=B)(
      model.parameters(),
      lr=args.lr,
      betas=(args.beta1, args.beta2),
      weight_decay=args.weight_decay,
  )
  scheduler = get_hfta_lr_scheduler_for(optim.lr_scheduler.StepLR, B=B)(
      optimizer,
      step_size=args.step_size,
      gamma=args.gamma,
  )

  epoch_timer = EpochTimer()

  for epoch in range(args.epochs):
    epoch_timer.epoch_start(epoch)
    num_samples_done = train(args, model, criterion, optimizer, scaler,
                             device, train_loader, epoch, B)
    scheduler.step()

    epoch_timer.epoch_stop(num_samples_done)
    print('Epoch {} took {} s!'.format(epoch, epoch_timer.epoch_latency(epoch)))

  if args.device == 'xla':
    print(met.metrics_report())

  if args.outf is not None:
    epoch_timer.to_csv(args.outf)

  if args.eval:
    acc_top1, acc_top5 = test(args, model, device, test_loader, B)
    if args.outf is not None:
      pd.DataFrame({
          'acc:top1': acc_top1,
          'acc:top5': acc_top5,
      }).to_csv(os.path.join(args.outf, 'eval.csv'))
    return acc_top1, acc_top5


if __name__ == '__main__':
  main(attach_args().parse_args())
