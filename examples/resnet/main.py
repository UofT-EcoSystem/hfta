import argparse
import random
import torch
import numpy as np
import torch.optim as optim
import torch.cuda.amp as amp

from utils import train, test, init_dataloader, attach_default_args

try:
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.debug.metrics as met
except ImportError:
  pass

from hfta.optim import get_hfta_optim_for
from hfta.workflow import EpochTimer
from model import Resnet18


def attach_args(parser=argparse.ArgumentParser(
    description='Resnet Model Example')):
  parser = attach_default_args(parser)
  parser.add_argument('--convergence_test',
                      action='store_true',
                      default=False,
                      help='Run convergence test')
  parser.add_argument('--save_init_model',
                      action='store_true',
                      default=False,
                      help='For Saving the current Model')
  parser.add_argument('--load_init_model',
                      action='store_true',
                      default=False,
                      help='For Saving the current Model')
  parser.add_argument('--model_dir', type=str, default=None, help='model file path')
  return parser

def main(args):
  print(args)
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  track_running_stats=(args.device != 'xla')
  if args.device == 'cuda':
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    print('Enable cuDNN heuristics!')

  device = (torch.device(args.device)
            if args.device in {'cpu', 'cuda'} else xm.xla_device())
  if args.device == 'cuda' and args.amp:
    scaler = amp.GradScaler()
  else:
    scaler = None

  train_loader, test_loader = init_dataloader(args)

  B = len(args.lr) if args.hfta else 0

  model = Resnet18(num_classes=10, B=B, track_running_stats=track_running_stats).to(device)
  if not args.convergence_test:
    if B == 0 and args.save_init_model:
      torch.save(model, args.model_dir)
    if args.load_init_model:
      model.init_load([args.model_dir] * max(1, B))
  print('B={} lr={}'.format(B, args.lr))
  
  optimizer = get_hfta_optim_for(optim.Adadelta, B=B)(
      model.parameters(),
      lr=args.lr if B > 0 else args.lr[0],
  )

  all_losses = []
  epoch_timer = EpochTimer()
  for epoch in range(args.epochs):
    epoch_timer.epoch_start(epoch)
    num_samples_per_epoch, epoch_losses = train(args, model, device, train_loader, optimizer, epoch,  B, save_loss=args.convergence_test, scaler=scaler)
    epoch_timer.epoch_stop(num_samples_per_epoch)
    if args.convergence_test:
      all_losses.append(epoch_losses)
    print('Epoch {} took {} s!'.format(epoch, epoch_timer.epoch_latency(epoch)))

  if args.convergence_test:
    all_losses = torch.cat(all_losses, 0).cpu().numpy()
    np.savetxt(args.outf, all_losses, delimiter=",")
  else:
    if args.device == 'xla':
      print(met.metrics_report())
    if args.outf is not None:
      epoch_timer.to_csv(args.outf)

  if args.eval:
    test(model, device, test_loader, B)
  print('All jobs Finished!')


if __name__ == '__main__':
  main(attach_args().parse_args())
