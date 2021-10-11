from __future__ import print_function
import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

try:
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.debug.metrics as met
except ImportError:
  pass

from hfta.ops import get_hfta_op_for
from hfta.optim import get_hfta_optim_for, consolidate_hyperparams_and_determine_B
from hfta.workflow import EpochTimer


def attach_config_args(parser=argparse.ArgumentParser()):
  parser.add_argument('--ngpu',
                      type=int,
                      default=1,
                      help='number of GPUs to use')
  parser.add_argument(
      '--netG',
      default='',
      help="path to netG (to continue training)",
  )
  parser.add_argument(
      '--netD',
      default='',
      help="path to netD (to continue training)",
  )
  parser.add_argument(
      '--workers',
      type=int,
      help='number of data loading workers',
      default=2,
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
      help='number of iterations per epoch to train for',
  )
  parser.add_argument(
      '--classes',
      default='bedroom',
      help='comma separated list of classes for the lsun data set',
  )
  parser.add_argument('--outf', type=str, default=None, help='output folder')
  parser.add_argument('--dataset', type=str, required=True, help="dataset path")
  parser.add_argument(
      '--dataset_type',
      required=True,
      help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake',
  )

  parser.add_argument(
      '--device',
      type=str,
      default='cuda',
      choices=['cpu', 'cuda', 'xla'],
      help='the device where this test is running',
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
  parser.add_argument('--seed', type=int, default=1117, help='Seed')
  parser.add_argument(
      '--warmup-data-loading',
      default=False,
      action='store_true',
      help='go over the training and validation loops without performing '
      'forward and backward passes',
  )
  parser.add_argument(
      '--dry-run',
      action='store_true',
      help='check a single training cycle works',
  )
  return parser


def attach_fusible_args(parser=argparse.ArgumentParser()):
  # Adam settings:
  parser.add_argument(
      '--lr',
      type=float,
      default=[0.0002],
      nargs='*',
      help='learning rate, default=0.0002',
  )
  parser.add_argument(
      '--beta1',
      type=float,
      default=[0.5],
      nargs='*',
      help='beta1 for adam. default=0.5',
  )
  return parser


def attach_nonfusible_args(parser=argparse.ArgumentParser()):
  parser.add_argument(
      '--batchSize',
      type=int,
      default=128,
      help='input batch size',
  )
  parser.add_argument(
      '--imageSize',
      type=int,
      default=64,
      help='the height / width of the input image to network',
  )
  parser.add_argument(
      '--nz',
      type=int,
      default=100,
      help='size of the latent z vector',
  )
  parser.add_argument('--ngf', type=int, default=64)
  parser.add_argument('--ndf', type=int, default=64)
  return parser


def attach_args(parser=argparse.ArgumentParser()):
  attach_config_args(parser)
  attach_fusible_args(parser)
  attach_nonfusible_args(parser)
  return parser


args = attach_args().parse_args()
print(args)

try:
  if args.outf is not None:
    os.makedirs(args.outf)
except OSError:
  pass

print("Random Seed: ", args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.device == 'cuda':
  cudnn.benchmark = True

if args.dataset is None and str(args.dataset_type).lower() != 'fake':
  raise ValueError("`dataset_type` parameter is required for dataset \"%s\"" %
                   args.dataset)

if args.dataset_type in ['imagenet', 'folder', 'lfw']:
  # folder dataset
  dataset = dset.ImageFolder(
      root=args.dataset,
      transform=transforms.Compose([
          transforms.Resize(args.imageSize),
          transforms.CenterCrop(args.imageSize),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ]),
  )
  nc = 3
elif args.dataset_type == 'lsun':
  classes = [c + '_train' for c in args.classes.split(',')]
  dataset = dset.LSUN(
      root=args.dataset,
      classes=classes,
      transform=transforms.Compose([
          transforms.Resize(args.imageSize),
          transforms.CenterCrop(args.imageSize),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ]),
  )
  nc = 3
elif args.dataset_type == 'cifar10':
  dataset = dset.CIFAR10(
      root=args.dataset,
      download=True,
      transform=transforms.Compose([
          transforms.Resize(args.imageSize),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ]),
  )
  nc = 3

elif args.dataset_type == 'mnist':
  dataset = dset.MNIST(
      root=args.dataset,
      download=True,
      transform=transforms.Compose([
          transforms.Resize(args.imageSize),
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,)),
      ]),
  )
  nc = 1

elif args.dataset_type == 'fake':
  dataset = dset.FakeData(
      image_size=(3, args.imageSize, args.imageSize),
      transform=transforms.ToTensor(),
  )
  nc = 3

assert dataset
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batchSize,
    shuffle=True,
    num_workers=int(args.workers),
)

if args.hfta:
  B = consolidate_hyperparams_and_determine_B(args, ['lr', 'beta1'])
else:
  B = 0
  args.lr, args.beta1 = args.lr[0], args.beta1[0]

if args.device == 'cuda':
  assert torch.cuda.is_available()
device = xm.xla_device() if args.device == 'xla' else torch.device(args.device)

ngpu = int(args.ngpu)
nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)


# custom weights initialization called on netG and netD
# NOTE(wangsh46): This is okay for HFTA, because torch.nn.init.normal_ (or
# torch.Tensor.normal_ to be specific) is element-wise; so is
# torch.nn.init.zeros_
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    torch.nn.init.normal_(m.weight, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    torch.nn.init.normal_(m.weight, 1.0, 0.02)
    torch.nn.init.zeros_(m.bias)


ConvTranspose2d = get_hfta_op_for(nn.ConvTranspose2d, B=B)
BatchNorm2d = get_hfta_op_for(nn.BatchNorm2d, B=B)
ReLU = get_hfta_op_for(nn.ReLU, B=B)
Tanh = get_hfta_op_for(nn.Tanh, B=B)


class Generator(nn.Module):

  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
        # input is Z, going into a convolution
        ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        BatchNorm2d(ngf * 8, track_running_stats=(args.device != 'xla')),
        ReLU(True),
        # state size. (ngf*8) x 4 x 4
        ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        BatchNorm2d(ngf * 4, track_running_stats=(args.device != 'xla')),
        ReLU(True),
        # state size. (ngf*4) x 8 x 8
        ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        BatchNorm2d(ngf * 2, track_running_stats=(args.device != 'xla')),
        ReLU(True),
        # state size. (ngf*2) x 16 x 16
        ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        BatchNorm2d(ngf, track_running_stats=(args.device != 'xla')),
        ReLU(True),
        # state size. (ngf) x 32 x 32
        ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        Tanh()
        # state size. (nc) x 64 x 64
    )

  def forward(self, input):
    if input.is_cuda and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output = self.main(input)
    return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if args.netG != '':
  netG.load_state_dict(torch.load(args.netG))
print(netG)

Conv2d = get_hfta_op_for(nn.Conv2d, B=B)
LeakyReLU = get_hfta_op_for(nn.LeakyReLU, B=B)


class Discriminator(nn.Module):

  def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
        # input is (nc) x 64 x 64
        Conv2d(nc, ndf, 4, 2, 1, bias=False),
        LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 32 x 32
        Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        BatchNorm2d(ndf * 2, track_running_stats=(args.device != 'xla')),
        LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        BatchNorm2d(ndf * 4, track_running_stats=(args.device != 'xla')),
        LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        BatchNorm2d(ndf * 8, track_running_stats=(args.device != 'xla')),
        LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 4 x 4
        Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
    )

  def forward(self, input):
    if input.is_cuda and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output = self.main(input)

    if B > 0:
      output = output.view(-1, B, 1).squeeze(2)
    else:
      output = output.view(-1, 1).squeeze(1)
    return output


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if args.netD != '':
  netD.load_state_dict(torch.load(args.netD))
print(netD)

criterion = nn.BCEWithLogitsLoss()

if B > 0:
  fixed_noise = torch.randn(args.batchSize, B, nz, 1, 1, device=device)
else:
  fixed_noise = torch.randn(args.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
Adam = get_hfta_optim_for(optim.Adam, B=B)
optimizerD = Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

if args.device == 'cuda' and args.amp:
  scaler = amp.GradScaler()

if args.dry_run:
  args.epochs = 1


def loss_fn(output, label, batch_size):
  if B > 0:
    return B * criterion(output.view(batch_size * B), label)
  else:
    return criterion(output, label)


print("NVIDIA_TF32_OVERRIDE: {}".format(os.environ.get('NVIDIA_TF32_OVERRIDE')))

epoch_timer = EpochTimer()
for epoch in range(args.epochs):
  num_samples_per_epoch = 0
  epoch_timer.epoch_start(epoch)
  for i, data in enumerate(dataloader, 0):
    if i >= args.iters_per_epoch:
      break
    if args.warmup_data_loading:
      continue
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    netD.zero_grad(set_to_none=True)
    real_cpu = data[0].to(device)
    batch_size = real_cpu.size(0)
    if B > 0:
      real_cpu = real_cpu.unsqueeze(1).expand(-1, B, -1, -1, -1).contiguous()
      label_size = (batch_size * B,)
    else:
      label_size = (batch_size,)
    label = torch.full(
        label_size,
        real_label,
        dtype=real_cpu.dtype,
        device=device,
    )

    if args.device == 'cuda':
      with amp.autocast(enabled=args.amp):
        output = netD(real_cpu)
        errD_real = loss_fn(output, label, batch_size)
      if args.amp:
        scaler.scale(errD_real).backward()
      else:
        errD_real.backward()
    else:
      output = netD(real_cpu)
      errD_real = loss_fn(output, label, batch_size)
      errD_real.backward()
    D_x = output.mean().item()

    # train with fake
    if B > 0:
      noise = torch.randn(batch_size, B, nz, 1, 1, device=device)
    else:
      noise = torch.randn(batch_size, nz, 1, 1, device=device)
    label.fill_(fake_label)
    if args.device == 'cuda':
      with amp.autocast(enabled=args.amp):
        fake = netG(noise)
        output = netD(fake.detach())
        errD_fake = loss_fn(output, label, batch_size)
      if args.amp:
        scaler.scale(errD_fake).backward()
      else:
        errD_fake.backward()
    else:
      fake = netG(noise)
      output = netD(fake.detach())
      errD_fake = loss_fn(output, label, batch_size)
      errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    if args.device == 'cuda':
      if args.amp:
        scaler.step(optimizerD)
      else:
        optimizerD.step()
    elif args.device == 'cpu':
      optimizerD.step()
    else:
      xm.optimizer_step(optimizerD, barrier=True)

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad(set_to_none=True)
    label.fill_(real_label)  # fake labels are real for generator cost
    if args.device == 'cuda':
      with amp.autocast(enabled=args.amp):
        output = netD(fake)
        errG = loss_fn(output, label, batch_size)
      if args.amp:
        scaler.scale(errG).backward()
      else:
        errG.backward()
    else:
      output = netD(fake)
      errG = loss_fn(output, label, batch_size)
      errG.backward()
    D_G_z2 = output.mean().item()
    if args.device == 'cuda':
      if args.amp:
        scaler.step(optimizerG)
      else:
        optimizerG.step()
    elif args.device == 'cpu':
      optimizerG.step()
    else:
      xm.optimizer_step(optimizerG, barrier=True)

    if args.device == 'cuda' and args.amp:
      scaler.update()

    num_samples_per_epoch += batch_size * max(B, 1)

    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): '
          '%.4f / %.4f' % (
              epoch,
              args.epochs,
              i,
              len(dataloader),
              errD.item(),
              errG.item(),
              D_x,
              D_G_z1,
              D_G_z2,
          ))
    if args.dry_run:
      break

  epoch_timer.epoch_stop(num_samples_per_epoch)
  print('Epoch {} took {} s!'.format(epoch, epoch_timer.epoch_latency(epoch)))

if args.device == 'xla':
  print(met.metrics_report())
if args.outf is not None:
  epoch_timer.to_csv(args.outf)
