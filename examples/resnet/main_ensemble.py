import argparse
import random
import torch
import numpy as np
import torch.optim as optim
import torch.cuda.amp as amp
import json

from model import ResnetEnsembleModel, str_to_class
from utils import train, test, init_dataloader, attach_default_args


try:
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.debug.metrics as met
except ImportError:
  pass

from hfta.optim import get_hfta_optim_for
from hfta.workflow import EpochTimer

def attach_args(parser=argparse.ArgumentParser(
    description='Resnet Ensemble Model Example')):
  parser = attach_default_args(parser)
  parser.add_argument(
      '--config_file',
      type=str,
      default='resnet_ensemble.conf',
      help='folder to load the structure of resnet ensemble model')
  return parser

def main(args):
  print(args)
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
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

  model_def = json.load(open(args.config_file, 'r'))
  print("Loaded model:", model_def["name"])
  print("Model config:", model_def)
  B = model_def["B"]

  block = str_to_class(model_def["block"])
  model = ResnetEnsembleModel(model_def["arch"], block,
                              num_classes=10, B=B).to(device)

  optimizer = get_hfta_optim_for(optim.Adadelta, B=B)(
      model.parameters(),
      lr=args.lr if B > 0 else args.lr[0],
  )

  epoch_timer = EpochTimer()
  for epoch in range(args.epochs):
    epoch_timer.epoch_start(epoch)
    num_samples_per_epoch, _ = train(args, model, device, train_loader, optimizer, epoch, B, scaler=scaler)
    epoch_timer.epoch_stop(num_samples_per_epoch)
    print('Epoch {} took {} s!'.format(epoch, epoch_timer.epoch_latency(epoch)))

  if args.device == 'xla':
    print(met.metrics_report())
  if args.outf is not None:
    epoch_timer.to_csv(args.outf)

  if args.eval:
    test(model, device, test_loader, B)
  print('All jobs Finished!')


if __name__ == '__main__':
  main(attach_args().parse_args())
