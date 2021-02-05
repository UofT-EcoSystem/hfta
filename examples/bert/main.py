# Copyright (c) 2020-     UofT-EcoSystem,
# Copyright 2018 - 2019 Junseong Kim, Scatter Lab, respective BERT contributors
# Copyright (c) 2018 Alexander Rush : The Annotated Trasnformer
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# coding: utf-8
import argparse
import time
import os
import torch
import torch.nn.functional as F
import torch.onnx
import random
import numpy as np
from torch import optim
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from hfta.optim import get_hfta_optim_for, get_hfta_lr_scheduler_for
from hfta.workflow import EpochTimer
from datasets import Corpus

try:
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.debug.metrics as met
except ImportError:
  pass

import model as Model


def attach_config_args(parser=argparse.ArgumentParser()):
  parser.add_argument('--epochs',
                      type=int,
                      default=250,
                      help='number of epochs to train for')
  parser.add_argument('--iters-per-epoch',
                      type=int,
                      default=float('inf'),
                      help='number of epochs to train for')
  parser.add_argument('--batch_size',
                      type=int,
                      default=32,
                      help='input batch size')
  parser.add_argument('--emsize',
                      type=int,
                      default=128,
                      help='size of word embeddings')
  parser.add_argument('--nhid',
                      type=int,
                      default=512,
                      help='number of hidden units per layer')
  parser.add_argument('--nlayers', type=int, default=8, help='number of layers')
  parser.add_argument('--bptt', type=int, default=32, help='sequence length')
  parser.add_argument('--nhead',
                      type=int,
                      default=8,
                      help='the number of heads of the transformer model')
  parser.add_argument('--max-token',
                      type=int,
                      default=1000000,
                      help='Max number of tokens to predict')
  parser.add_argument('--dropout',
                      type=float,
                      default=0.1,
                      help='dropout applied to layers (0 = no dropout)')
  parser.add_argument('--outf', type=str, default=None, help='output folder')
  parser.add_argument('--model', type=str, default='', help='model path')
  parser.add_argument('--dataset', type=str, required=True, help="dataset path")
  parser.add_argument('--device',
                      type=str,
                      default='cuda',
                      choices=['cpu', 'cuda', 'xla'],
                      help="the device where this test is running")
  parser.add_argument('--hfta',
                      default=False,
                      action='store_true',
                      help='use HFTA')
  parser.add_argument('--amp',
                      default=False,
                      action='store_true',
                      help='Enable AMP; only used when --device is cuda')
  parser.add_argument('--eval',
                      default=False,
                      action='store_true',
                      help='run the evaluation loop')
  parser.add_argument('--seed', type=int, help='Seed', default=1117)
  parser.add_argument('--log-interval',
                      type=int,
                      default=50,
                      metavar='N',
                      help='report interval')
  parser.add_argument(
      '--warmup-data-loading',
      default=False,
      action='store_true',
      help='go over the training and validation loops without performing '
      'forward and backward passes')
  return parser


def attach_fusible_args(parser=argparse.ArgumentParser()):
  # Adam settings:
  parser.add_argument('--lr',
                      type=float,
                      default=[20],
                      nargs='*',
                      help='learning rate (default: 1.0)')
  parser.add_argument('--gamma',
                      type=float,
                      default=[0.25],
                      nargs='*',
                      help='learning rate decay, default=0.5')
  parser.add_argument('--step_size',
                      type=int,
                      default=[5],
                      nargs='*',
                      help='Period of learning rate decay')
  return parser


def attach_args(
    parser=argparse.ArgumentParser(description='PyTorch Bert Language Model')):
  attach_config_args(parser)
  attach_fusible_args(parser)
  return parser


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
def batchify(data, bsz):
  # Work out how cleanly we can divide the dataset into bsz parts.
  nbatch = data[0].size(0) // bsz
  # Trim off any extra elements that wouldn't cleanly fit (remainders).
  seq = data[0].narrow(0, 0, nbatch * bsz).view(bsz, -1)
  pos = data[1].narrow(0, 0, nbatch * bsz).view(bsz, -1)
  # Evenly divide the data across the bsz batches.
  seq = seq.t().contiguous().to(device)
  pos = pos.t().contiguous().to(device)
  return seq, pos


###############################################################################
# Training code
###############################################################################


def repackage_hidden(h):
  """Wraps hidden states in new Tensors, to detach them from their history."""

  if isinstance(h, torch.Tensor):
    return h.detach()
  else:
    return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.
def get_batch(source, i):
  seq_len = min(args.bptt, len(source[0]) - 1 - i)
  data = source[0][i:i + seq_len].t().contiguous()
  pos = source[1][i:i + seq_len].t().contiguous()
  seg = torch.zeros(pos.shape, dtype=torch.long)
  for idx, p in enumerate(pos):
    if (p[-1] - p[0] + 1 == len(p)):
      continue
    id = 0
    pre = -1
    for idy, p_now in enumerate(p):
      if pre > p_now:
        id = idy
        break
      pre = p_now
    seg[idx][id:] = 1
  target = source[0][i + 1:i + 1 + seq_len].view(-1)
  return data, pos, seg.to(device), target


def evaluate(args, model, eval_data, B=0):
  # Turn on evaluation mode which disables dropout.
  model.eval()
  total_loss = 0.
  ntokens = len(corpus.dictionary)
  length = 0
  with torch.no_grad():
    for i in range(0, eval_data[0].size(0) - 1, args.bptt):
      data, pos, seg, targets = get_batch(eval_data, i)
      NL = targets.size(0)
      if B > 0:
        data = data.unsqueeze(0).expand(B, -1, -1)
        pos = pos.unsqueeze(0).expand(B, -1, -1)
        seg = seg.unsqueeze(0).expand(B, -1, -1)
        targets = targets.repeat(B)
      output = model(data, pos, seg)
      output = output.view(-1, ntokens)
      total_loss += F.nll_loss(output, targets.contiguous(),
                               reduction='none').view(-1, NL).mean(dim=1)
      length += 1

  test_loss = total_loss / length
  loss_str = ["%.4f" % e for e in test_loss]
  print('Test set: \tAverage loss: {} \n'.format(loss_str))
  return test_loss


def train(args, model, train_data, optimizer, epoch, B, scaler=None):
  # Turn on training mode which enables dropout.
  model.train()
  start_time = time.time()
  ntokens = len(corpus.dictionary)

  num_samples_per_epoch = 0
  for batch_idx, i in enumerate(range(0, train_data[0].size(0) - 1, args.bptt)):
    if batch_idx >= args.iters_per_epoch:
      break
    if args.warmup_data_loading:
      continue

    data, pos, seg, targets = get_batch(train_data, i)

    NL = targets.size(0)
    if B > 0:
      data = data.unsqueeze(0).expand(B, -1, -1)
      pos = pos.unsqueeze(0).expand(B, -1, -1)
      seg = seg.unsqueeze(0).expand(B, -1, -1)
      targets = targets.repeat(B)

    # Starting each batch, we detach the hidden state from how it was previously produced.
    # If we didn't, the model would try backpropagating all the way to start of the dataset.
    optimizer.zero_grad()

    if args.device == "cuda":
      with amp.autocast(enabled=args.amp):
        output = model(data, pos, seg)
        output = output.view(-1, ntokens)
        if args.amp:
          assert scaler is not None
        loss = max(B, 1) * F.nll_loss(output, targets)
    else:
      output = model(data, pos)
      output = output.view(-1, ntokens)
      loss = max(B, 1) * F.nll_loss(output, targets)

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

    num_samples_per_epoch += args.batch_size * max(B, 1)
    if batch_idx % args.log_interval == 0 and batch_idx > 0:
      with torch.no_grad():
        cur_loss = F.nll_loss(output, targets.contiguous(),
                              reduction='none').view(-1, NL).mean(dim=1)
      loss_str = ["%.7f" % e for e in cur_loss]
      elapsed = time.time() - start_time
      print('| epoch {:3d} | {:5d}/{:5d} batches |  ms/batch {:5.2f} | '
            'loss {}'.format(epoch, batch_idx,
                             len(train_data[0]) // args.bptt,
                             elapsed * 1000 / args.log_interval, loss_str))
      start_time = time.time()

  return num_samples_per_epoch


args = attach_args().parse_args()
print(args)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.device == 'cuda':
  assert torch.cuda.is_available()
  cudnn.benchmark = True

device = xm.xla_device() if args.device == 'xla' else torch.device(args.device)

B = len(args.lr) if args.hfta else 0
###############################################################################
# Load data
###############################################################################
corpus = Corpus(args.dataset, args.max_token, True)
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, args.batch_size)
test_data = batchify(corpus.test, args.batch_size)

###############################################################################
# Build the model
###############################################################################
ntokens = len(corpus.dictionary)
model = Model.BERT(ntokens,
                   args.nhid,
                   args.nlayers,
                   args.nhead,
                   args.dropout,
                   B=B).to(device)

if args.device == 'cuda' and args.amp:
  scaler = amp.GradScaler()
else:
  scaler = None

optimizer = get_hfta_optim_for(optim.Adadelta, B=B)(
    model.parameters(),
    lr=args.lr if B > 0 else args.lr[0],
)

scheduler = get_hfta_lr_scheduler_for(optim.lr_scheduler.StepLR, B=B)(
    optimizer,
    step_size=3,
    gamma=args.gamma if B > 0 else args.gamma[0],
)

print("NVIDIA_TF32_OVERRIDE: {}".format(os.environ.get('NVIDIA_TF32_OVERRIDE')))

epoch_timer = EpochTimer()
print("start training!")
# At any point you can hit Ctrl + C to break out of training early.
try:
  for epoch in range(args.epochs):
    epoch_timer.epoch_start(epoch)
    num_samples_per_epoch = train(args,
                                  model,
                                  train_data,
                                  optimizer,
                                  epoch,
                                  B,
                                  scaler=scaler)
    scheduler.step()
    epoch_timer.epoch_stop(num_samples_per_epoch)
    print('Epoch {} took {} s!'.format(epoch, epoch_timer.epoch_latency(epoch)))
  if args.eval:
    val_loss = evaluate(args, model, val_data, B=B)

  if args.device == 'xla':
    print(met.metrics_report())
  if args.outf is not None:
    epoch_timer.to_csv(args.outf)
except KeyboardInterrupt:
  print('Exiting from training early')
