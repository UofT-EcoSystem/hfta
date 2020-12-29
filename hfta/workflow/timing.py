import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.interpolate
import time

from .utils import _init_modes, _init_precs


class EpochTimer:

  def __init__(self):
    self.reset()

  def reset(self):
    self._timing = {
        'epoch': [],
        'epoch_start': [],
        'epoch_stop': [],
        'num_samples': [],
    }
    self._epoch_to_idx = {}
    self._epoch_start_called = False

  def epoch_start(self, epoch):
    self._epoch_to_idx[epoch] = len(self._timing['epoch'])
    self._timing['epoch'].append(epoch)
    self._timing['epoch_start'].append(time.time())
    self._epoch_start_called = True

  def epoch_stop(self, num_samples):
    assert self._epoch_start_called
    self._timing['epoch_stop'].append(time.time())
    self._timing['num_samples'].append(num_samples)
    self._epoch_start_called = False

  def epoch_latency(self, epoch):
    idx = self._epoch_to_idx[epoch]
    return self._timing['epoch_stop'][idx] - self._timing['epoch_start'][idx]

  def to_csv(self, outdir, filename='timing.csv'):
    pd.DataFrame(self._timing).to_csv(os.path.join(outdir, filename))


def _attach_args(
    parser=argparse.ArgumentParser(description='Timing Results Parser')):
  parser.add_argument(
      '--outdirs',
      type=str,
      required=True,
      default=[],
      nargs='+',
      help='path(s) to the workflow outdir_prefix.',
  )
  parser.add_argument(
      '--device',
      type=str,
      required=True,
      choices=['cpu', 'cuda', 'xla'],
      help='cpu, cuda or xla',
  )
  parser.add_argument(
      '--device-model',
      type=str,
      required=True,
      help='The model of the device (e.g, v100, a100, rtx6000 or TPU v3)',
  )
  parser.add_argument(
      '--precs',
      type=str,
      default=None,
      choices=['fp32', 'amp', 'bf16'],
      nargs='*',
      help='training precision(s)',
  )
  parser.add_argument(
      '--modes',
      type=str,
      default=None,
      choices=['serial', 'concurrent', 'mps', 'mig', 'hfta'],
      nargs='*',
      help='hardware sharing mode(s)',
  )
  parser.add_argument(
      '--filename',
      type=str,
      default='timing.csv',
      help='timing filename',
  )
  parser.add_argument(
      '--save',
      type=str,
      required=True,
      help='the file path to save the result summary (and the plot if --plot '
      'is enabled)',
  )
  parser.add_argument(
      '--plot',
      default=False,
      action='store_true',
      help='plot figure using matplotlib',
  )
  return parser


def _parse_args(parser):
  args = parser.parse_args()
  args.device_model = args.device_model.lower()
  args.outdirs = [
      os.path.abspath(os.path.expanduser(outdir)) for outdir in args.outdirs
  ]
  if args.precs is None:
    args.precs = _init_precs(args.device, args.device_model)
  if args.modes is None:
    args.modes = _init_modes(args.device, args.device_model)
  if args.plot:
    assert 'serial' in args.modes
  return args


def _calculate_throughputs(timing_dfs, device):
  timestamp_start = float('inf')
  timestamp_stop = float('-inf')
  total_samples = 0
  if device in {'cpu', 'cuda'}:
    warmup_offset = 1
  else:
    assert device == 'xla'
    warmup_offset = 2
  for timing_df in timing_dfs:
    assert len(timing_df) > warmup_offset
    timestamp_start = min(timing_df['epoch_start'].iloc[warmup_offset],
                          timestamp_start)
    timestamp_stop = max(timing_df['epoch_stop'].iloc[-1], timestamp_stop)
    total_samples += timing_df['num_samples'].iloc[warmup_offset:].sum()
  return total_samples / (timestamp_stop - timestamp_start)


def _aggregate_along_rows(df, mode, prec):
  s_avg, s_min, s_max = df.mean(axis=1), df.min(axis=1), df.max(axis=1)
  df['{}:{}:avg'.format(mode, prec)] = s_avg
  df['{}:{}:min'.format(mode, prec)] = s_min
  df['{}:{}:max'.format(mode, prec)] = s_max


def _get_serial_throughputs(
    outdirs,
    device,
    device_model,
    precs,
    filename,
):
  """ The result is in the format of
  {
    'amp': pd.DataFrame,  # df only contains 1 row for B=1
    'fp32': pd.DataFrame, # df only contains 1 row for B=1
  }
  df format: (`B` is the index)
  B  serial:{prec}:0 serial:{prec}:1 ... serial:{prec}:avg serial:{prec}:min serial:{prec}:max
  1       float           float      ...        float             float             float
  """
  throughputs = {}
  for prec in precs:
    throughputs[prec] = {
        'B': [1],
    }

    for outdir_idx, outdir in enumerate(outdirs):
      end_outdir_path = os.path.join(outdir, device, device_model, prec,
                                     'serial')
      timing_df = pd.read_csv(os.path.join(end_outdir_path, filename))
      throughputs[prec]['serial:{}:{}'.format(prec, outdir_idx)] = [
          _calculate_throughputs([timing_df], device),
      ]

    throughputs[prec] = pd.DataFrame(throughputs[prec]).set_index('B')
    _aggregate_along_rows(throughputs[prec], 'serial', prec)

  return throughputs


def _get_hardware_sharing_throughputs(
    outdirs,
    device,
    device_model,
    precs,
    filename,
    mode,
):
  """ The result is in the format of
  {
    'amp': pd.DataFrame,  # df contains max_B rows
    'fp32': pd.DataFrame, # df contains max_B rows
  }
  df format: (`B` is the index)
  B     {mode}:{prec}:0 {mode}:{prec}:1 ... {mode}:{prec}:avg {mode}:{prec}:min {mode}:{prec}:max
  1          float           float      ...       float             float             float
  2          float           float      ...       float             float             float
  3          float           float      ...       float             float             float
  ...
  max_B      float           float      ...       float             float             float
  """
  throughputs = {}
  for prec in precs:
    throughputs[prec] = {'B': []}
    for outdir_idx, outdir in enumerate(outdirs):
      Bs = []
      throughputs_of_Bs = []
      mode_outdir_path = os.path.join(outdir, device, device_model, prec, mode)
      for B_exp in os.listdir(mode_outdir_path):
        B = int(B_exp[1:])
        Bs.append(B)
        B_outdir_path = os.path.join(mode_outdir_path, B_exp)
        timing_dfs = None
        if mode == 'hfta':
          timing_dfs = [pd.read_csv(os.path.join(B_outdir_path, filename))]
        else:
          timing_dfs = [
              pd.read_csv(
                  os.path.join(B_outdir_path, 'idx{}'.format(idx), filename))
              for idx in range(B)
          ]
        throughputs_of_Bs.append(_calculate_throughputs(timing_dfs, device))
      max_B = max(Bs)
      linear_interpolator = scipy.interpolate.interp1d(Bs, throughputs_of_Bs)
      throughputs[prec]['{}:{}:{}'.format(mode, prec, outdir_idx)] = [
          linear_interpolator(B) for B in range(1, max_B + 1)
      ]
      throughputs[prec]['B'] = range(1, max_B + 1)
    throughputs[prec] = pd.DataFrame(throughputs[prec]).set_index('B')
    _aggregate_along_rows(throughputs[prec], mode, prec)
  return throughputs


_LINESTYLES = {
    'fp32': '--',
    'amp': '-',
    'bf16': '--',
}

_COLORS = {
    'serial': 'r',
    'concurrent': 'g',
    'mps': 'blue',
    'mig': 'orange',
    'hfta': 'purple',
}


def _plot_summary(summary, savepath, device):
  assert 'serial' in summary
  if device in {'cpu', 'cuda'}:
    assert 'fp32' in summary['serial']
    baseline = summary['serial']['fp32']['serial:fp32:avg'].loc[1]
  else:
    assert 'bf16' in summary['serial']
    baseline = summary['serial']['bf16']['serial:bf16:avg'].loc[1]
  plt.clf()
  for mode, throughputs in summary.items():
    for prec, df in throughputs.items():
      if mode == 'serial':
        plt.axhline(
            y=df['serial:{}:avg'.format(prec)].loc[1] / baseline,
            label='serial:{}'.format(prec),
            color=_COLORS[mode],
            linestyle=_LINESTYLES[prec],
        )
        plt.axhspan(
            df['serial:{}:min'.format(prec)].loc[1] / baseline,
            df['serial:{}:max'.format(prec)].loc[1] / baseline,
            facecolor=_COLORS[mode],
            alpha=0.3,
        )
      else:
        plt.plot(
            df.index.values,
            df['{}:{}:avg'.format(mode, prec)] / baseline,
            label='{}:{}'.format(mode, prec),
            color=_COLORS[mode],
            linestyle=_LINESTYLES[prec],
        )
        plt.fill_between(
            df.index.values,
            df['{}:{}:min'.format(mode, prec)] / baseline,
            df['{}:{}:max'.format(mode, prec)] / baseline,
            facecolor=_COLORS[mode],
            alpha=0.3,
        )
  lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.xlabel("B")
  plt.ylabel("Normalized Throughput")
  plt.rcParams['savefig.dpi'] = 300
  plt.savefig('{}.png'.format(savepath), bbox_inches='tight')


def timing_parser_main():
  args = _parse_args(_attach_args())
  summary = {}
  for mode in args.modes:
    if mode == 'serial':
      summary[mode] = _get_serial_throughputs(
          args.outdirs,
          args.device,
          args.device_model,
          args.precs,
          args.filename,
      )
    else:
      summary[mode] = _get_hardware_sharing_throughputs(
          args.outdirs,
          args.device,
          args.device_model,
          args.precs,
          args.filename,
          mode,
      )
  pd.concat(
      [summary[mode][prec] for mode in args.modes for prec in args.precs],
      axis=1,
  ).to_csv('{}.csv'.format(args.save))
  if args.plot:
    _plot_summary(summary, args.save, args.device)
