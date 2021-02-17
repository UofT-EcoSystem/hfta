import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
from torchvision import datasets, transforms
import argparse

try:
  import torch_xla.core.xla_model as xm
except ImportError:
  pass


def train(args, model, device, train_loader, optimizer, epoch, B, save_loss=False, scaler=None):
  model.train()
  avg_loss = torch.zeros(max(1, B)).to(device)
  all_loss = []
  num_samples_per_epoch = 0

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    N = target.size(0)
    num_samples_per_epoch += N * max(B, 0)
    if B > 0:
      data = data.unsqueeze(1).expand(-1, B, -1, -1, -1)
      target = target.repeat(B)
    optimizer.zero_grad()

    if args.device == "cuda":
      with amp.autocast(enabled=args.amp):
        output = model(data.contiguous())
        if B > 0:
          loss = B * F.nll_loss(output.view(B * N, -1), target)
        else:
          loss = F.nll_loss(output, target)
    else:
      output = model(data.contiguous())
      if B > 0:
        loss = B * F.nll_loss(output.view(B * N, -1), target)
      else:
        loss = F.nll_loss(output, target)

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

    if save_loss:
      with torch.no_grad():
        if B > 0:
          output = output.view(B * N, -1)
        losses = F.nll_loss(output, target.contiguous(),
                            reduction='none').view(-1, N).mean(dim=1)
        all_loss.append(torch.unsqueeze(losses, 0))
        avg_loss += losses

    if batch_idx % args.log_interval == 0:
      with torch.no_grad():
        if B > 0:
          output = output.view(B * N, -1)
        avg_loss = F.nll_loss(output, target.contiguous(),
                              reduction='none').view(-1, N).mean(dim=1)
      loss_str = ["%.5f" % e for e in avg_loss]
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
        epoch,
        batch_idx * len(data),
        len(train_loader.dataset),
        100. * batch_idx / len(train_loader),
        loss_str,
      ))
  if save_loss:
    return num_samples_per_epoch, torch.cat(all_loss)
  else:
    return num_samples_per_epoch, None


def test(model, device, test_loader, B):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      N = target.size(0)
      if B > 0:
        data = data.unsqueeze(1).expand(-1, B, -1, -1, -1)
        target = target.repeat(B)
      output = model(data.contiguous())
      # sum up batch loss
      if B > 0:
        output = output.view(B * N, -1)
      test_loss += F.nll_loss(output, target.contiguous(),
                              reduction='none').view(-1, N).sum(dim=1)
      # get the index of the max log-probability
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).view(-1, N).sum(dim=1)

  length = len(test_loader.dataset)
  test_loss /= length
  loss_str = ["%.4f" % e for e in test_loss]
  correct_str = [
      "%d/%d(%.2lf%%)" % (e, length, 100. * e / length) for e in correct
  ]
  print('Test set: \tAverage loss: {}, \n \t\t\tAccuracy: {}\n'.format(
      loss_str, correct_str))



def init_dataloader(args, shuffle=False):
  kwargs = {'batch_size': args.batch_size}
  kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': shuffle}, )

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
  ])
  dataset1 = datasets.CIFAR10(args.dataset,
                              train=True,
                              transform=transform)
  dataset2 = datasets.CIFAR10(args.dataset,
                              train=False,
                              transform=transform)
  train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
  return train_loader, test_loader


def attach_default_args(parser=argparse.ArgumentParser()):
  parser.add_argument('--epochs',
                      type=int,
                      default=25,
                      help='number of epochs to train for')
  parser.add_argument('--iters-per-epoch',
                      type=int,
                      default=float('inf'),
                      help='number of iterations per epoch to train for')
  parser.add_argument('--batch-size',
                      type=int,
                      default=1000,
                      help='input batch size for training (default: 64)')
  parser.add_argument('--outf', type=str, default=None, help='output folder')
  parser.add_argument('--dataset', type=str, required=True, help="dataset path")
  parser.add_argument('--device',
                      type=str,
                      default='cuda',
                      choices=['cpu', 'cuda', 'xla'],
                      help='the device where this test is running')
  parser.add_argument('--hfta',
                      default=False,
                      action='store_true',
                      help='use HFTA')
  parser.add_argument('--eval',
                      default=False,
                      action='store_true',
                      help='run the evaluation loop')
  parser.add_argument('--amp',
                      default=False,
                      action='store_true',
                      help='Enable AMP; only used when --device is cuda')
  parser.add_argument('--log-interval',
                      type=int,
                      default=50,
                      metavar='N',
                      help='report interval')
  parser.add_argument('--seed', type=int, default=1117, help='Seed')
  parser.add_argument('--warmup-data-loading',
                      default=False,
                      action='store_true',
                      help='go over the training and validation loops without performing '
                      'forward and backward passes')
  return attach_fusible_args(parser)


def attach_fusible_args(parser=argparse.ArgumentParser()):
  # Adadelta settings:
  parser.add_argument('--lr',
                      type=float,
                      default=[0.001],
                      nargs='*',
                      help='learning rate, default=0.0002')
  parser.add_argument('--gamma',
                      type=float,
                      default=[0.7],
                      nargs='*',
                      help='beta1 for adam. default=0.5')
  return parser

