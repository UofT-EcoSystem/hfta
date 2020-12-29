import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import pathlib
import psutil
import scipy.interpolate
import subprocess
import threading
import time

from .utils import run_command, _init_modes, _init_precs
from .timing import _aggregate_along_rows, _LINESTYLES, _COLORS

DCGM_FIELD_IDS = {
    'DCGM_FI_PROF_GR_ENGINE_ACTIVE': 1001,
    'DCGM_FI_PROF_DRAM_ACTIVE': 1005,
    'DCGM_FI_DEV_GPU_UTIL': 203,
    'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE': 1004,
    'DCGM_FI_PROF_PIPE_FP16_ACTIVE': 1008,
    'DCGM_FI_PROF_PIPE_FP32_ACTIVE': 1007,
    'DCGM_FI_PROF_PIPE_FP64_ACTIVE': 1006,
    'DCGM_FI_PROF_SM_OCCUPANCY': 1003,
    'DCGM_FI_PROF_SM_ACTIVE': 1002,
    'DCGM_FI_DEV_FB_TOTAL': 250,
    'DCGM_FI_DEV_FB_FREE': 251,
    'DCGM_FI_DEV_FB_USED': 252,
    'DCGM_FI_PROF_PCIE_TX_BYTES': 1009,
    'DCGM_FI_PROF_PCIE_RX_BYTES': 1010,
    'DCGM_FI_DEV_MEM_COPY_UTIL': 204,
}


def _get_dcgm_fields_enabled_for(device_model):
  if device_model in {'v100', 'a100'}:
    return [
        'DCGM_FI_DEV_GPU_UTIL',
        'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE',
        'DCGM_FI_PROF_PIPE_FP16_ACTIVE',
        'DCGM_FI_PROF_PIPE_FP32_ACTIVE',
        'DCGM_FI_PROF_PIPE_FP64_ACTIVE',
        'DCGM_FI_PROF_SM_OCCUPANCY',
        'DCGM_FI_PROF_SM_ACTIVE',
        'DCGM_FI_DEV_FB_USED',
        'DCGM_FI_PROF_PCIE_RX_BYTES',
        'DCGM_FI_PROF_PCIE_TX_BYTES',
        'DCGM_FI_DEV_MEM_COPY_UTIL',
        'DCGM_FI_PROF_GR_ENGINE_ACTIVE',
        'DCGM_FI_PROF_DRAM_ACTIVE',
    ]
  else:
    return [
        "DCGM_FI_DEV_GPU_UTIL",
        "DCGM_FI_DEV_FB_USED",
        "DCGM_FI_DEV_MEM_COPY_UTIL",
    ]


class DcgmMonitor:

  def __init__(self, device_model):
    self.fields = _get_dcgm_fields_enabled_for(device_model)
    self.field_ids = [DCGM_FIELD_IDS[f] for f in self.fields]
    self.field_ids_str = ','.join(map(str, self.field_ids))
    self.reset()

  def reset(self):
    self.metrics = {f: [] for f in self.fields}
    self.metrics.update({
        'timestamp': [],
        'cpu_percent': [],
        'host_mem_total': [],
        'host_mem_available': [],
    })
    self.to_shutdown = False

  def sample_metrics(self, interval=1.0):
    cpu = psutil.cpu_percent(interval=interval, percpu=False)
    self.metrics['cpu_percent'].append(cpu)
    mem = psutil.virtual_memory()
    self.metrics['host_mem_total'].append(mem.total)
    self.metrics['host_mem_available'].append(mem.available)
    self.metrics['timestamp'].append(time.time())
    dcgmi_out = run_command('dcgmi dmon -e {} -c 5'.format(self.field_ids_str))
    dcgmi_samples = {f: [] for f in self.fields}
    for line in dcgmi_out.split('\n')[-4:-1]:
      # THIS ASSUMES THAT THE OUTPUT OF DCGM MONITOR HAS THE FORMAT GPU X METRIC1 METRIC2 ...
      for idx, val in enumerate(line.split()[2:]):
        if val == 'N/A':
          continue
        dcgmi_samples[self.fields[idx]].append(float(val))
    for f, vals in dcgmi_samples.items():
      if len(vals) > 0:
        self.metrics[f].append(sum(vals) / len(vals))
      else:
        self.metrics[f].append(float('nan'))

  def save(self, output_dir, filename='dcgm_metrics.csv'):
    csv_path = os.path.join(output_dir, filename)
    pd.DataFrame(self.metrics).to_csv(csv_path)
    print('Saving metrics to {} !'.format(csv_path))


def dcgm_monitor_thread(monitor, outdir):
  while not monitor.to_shutdown:
    monitor.sample_metrics()
    time.sleep(9)
  monitor.save(outdir)


def dcgm_monitor_start(monitor, outdir):
  run_command('nv-hostengine')
  t = threading.Thread(
      target=dcgm_monitor_thread,
      name='DCGM Monitor Thread',
      args=(monitor, outdir),
  )
  t.start()
  return t


def dcgm_monitor_stop(monitor, thread):
  monitor.to_shutdown = True
  thread.join()
  monitor.reset()
  run_command('nv-hostengine -t')


def _attach_args(
    parser=argparse.ArgumentParser(description='DCGM Metric Parser')):
  parser.add_argument(
      '--outdirs',
      type=str,
      required=True,
      default=[],
      nargs='+',
      help='path(s) to the workflow outdir_prefix.',
  )
  parser.add_argument(
      '--device-model',
      type=str,
      required=True,
      help='The model of the device (e.g, v100, a100 or rtx6000)',
  )
  parser.add_argument(
      '--precs',
      type=str,
      default=None,
      choices=['fp32', 'amp'],
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
      default='dcgm_metrics.csv',
      help='DCGM metric filename',
  )
  parser.add_argument(
      '--savedir',
      type=str,
      required=True,
      help='the path of dir to save the result summary (and the plot if --plot '
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
  args.device = 'cuda'
  if args.precs is None:
    args.precs = _init_precs(args.device, args.device_model)
  if args.modes is None:
    args.modes = _init_modes(args.device, args.device_model)
  if args.plot:
    assert 'serial' in args.modes
  return args


_AGGREGATE_FUNCS = {
    'DCGM_FI_PROF_GR_ENGINE_ACTIVE': 'mean',
    'DCGM_FI_PROF_DRAM_ACTIVE': 'mean',
    'DCGM_FI_DEV_GPU_UTIL': 'mean',
    'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE': 'mean',
    'DCGM_FI_PROF_PIPE_FP16_ACTIVE': 'mean',
    'DCGM_FI_PROF_PIPE_FP32_ACTIVE': 'mean',
    'DCGM_FI_PROF_PIPE_FP64_ACTIVE': 'mean',
    'DCGM_FI_PROF_SM_OCCUPANCY': 'mean',
    'DCGM_FI_PROF_SM_ACTIVE': 'mean',
    'DCGM_FI_DEV_FB_USED': 'max',
    'DCGM_FI_PROF_PCIE_TX_BYTES': 'sum',
    'DCGM_FI_PROF_PCIE_RX_BYTES': 'sum',
    'DCGM_FI_DEV_MEM_COPY_UTIL': 'mean',
    'cpu_percent': 'mean',
    'host_mem_total': 'max',
    'host_mem_available': 'min',
}


def _aggregate_metric(metrics_df, field):
  lb, ub = round(0.1 * len(metrics_df)), round(0.9 * len(metrics_df))
  if field == 'host_mem_usage':
    return getattr(
        metrics_df['host_mem_total'].iloc[lb:ub],
        _AGGREGATE_FUNCS['host_mem_total'],
    )() - getattr(
        metrics_df['host_mem_available'].iloc[lb:ub],
        _AGGREGATE_FUNCS['host_mem_available'],
    )()
  else:
    return getattr(metrics_df[field].iloc[lb:ub], _AGGREGATE_FUNCS[field])()


def _get_serial_metrics(
    outdirs,
    device_model,
    precs,
    filename,
    field,
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
  metrics = {}
  for prec in precs:
    metrics[prec] = {
        'B': [1],
    }

    for outdir_idx, outdir in enumerate(outdirs):
      end_outdir_path = os.path.join(outdir, 'cuda', device_model, prec,
                                     'serial')
      metrics_df = pd.read_csv(os.path.join(end_outdir_path, filename))
      metrics[prec]['serial:{}:{}'.format(prec, outdir_idx)] = [
          _aggregate_metric(metrics_df, field),
      ]

    metrics[prec] = pd.DataFrame(metrics[prec]).set_index('B')
    _aggregate_along_rows(metrics[prec], 'serial', prec)

  return metrics


def _get_hardware_sharing_metrics(
    outdirs,
    device_model,
    precs,
    filename,
    mode,
    field,
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
  metrics = {}
  for prec in precs:
    metrics[prec] = {'B': []}
    for outdir_idx, outdir in enumerate(outdirs):
      Bs = []
      metrics_of_Bs = []
      mode_outdir_path = os.path.join(outdir, 'cuda', device_model, prec, mode)
      for B_exp in os.listdir(mode_outdir_path):
        B = int(B_exp[1:])
        Bs.append(B)
        B_outdir_path = os.path.join(mode_outdir_path, B_exp)
        metrics_df = pd.read_csv(os.path.join(B_outdir_path, filename))
        metrics_of_Bs.append(_aggregate_metric(metrics_df, field))
      max_B = max(Bs)
      linear_interpolator = scipy.interpolate.interp1d(Bs, metrics_of_Bs)
      metrics[prec]['{}:{}:{}'.format(mode, prec, outdir_idx)] = [
          linear_interpolator(B) for B in range(1, max_B + 1)
      ]
      metrics[prec]['B'] = range(1, max_B + 1)
    metrics[prec] = pd.DataFrame(metrics[prec]).set_index('B')
    _aggregate_along_rows(metrics[prec], mode, prec)
  return metrics


_PLOT_LABELS = {
    'DCGM_FI_PROF_GR_ENGINE_ACTIVE': 'Graphics engine active (ratio of time)',
    'DCGM_FI_PROF_DRAM_ACTIVE': 'Device memory interface active (ratio of '
                                'cycles)',
    'DCGM_FI_DEV_GPU_UTIL': 'GPU Utilization',
    'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE': 'Tensor (HMMA) pipe active (ratio of '
                                       'cycles)',
    'DCGM_FI_PROF_PIPE_FP16_ACTIVE': 'FP16 pipe active (ratio of cycles; '
                                     'exclude HMMA)',
    'DCGM_FI_PROF_PIPE_FP32_ACTIVE': 'FP32 pipe active (ratio of cycles)',
    'DCGM_FI_PROF_PIPE_FP64_ACTIVE': 'FP64 pipe active (ratio of cycles)',
    'DCGM_FI_PROF_SM_OCCUPANCY': 'Warps resident on an SM (ratio over the '
                                 'theoretical maximum number of warps per '
                                 'elapsed cycle)',
    'DCGM_FI_PROF_SM_ACTIVE': 'An SM has at least 1 warp assigned (ratio of '
                              'cycles)',
    'DCGM_FI_DEV_FB_USED': 'Used frame buffer (global memory; in GB)',
    'DCGM_FI_PROF_PCIE_TX_BYTES': 'GB of active PCIe transmit data',
    'DCGM_FI_PROF_PCIE_RX_BYTES': 'GB of active PCIe read data',
    'DCGM_FI_DEV_MEM_COPY_UTIL': 'Memory Utilization',
    'cpu_percent': 'CPU Utilization',
    'host_mem_usage': 'Host Memory Usage (GB)',
}

_PLOT_UNIT_CONVERSION = {
    'DCGM_FI_DEV_FB_USED': lambda x: x / 1024,
    'DCGM_FI_PROF_PCIE_TX_BYTES': lambda x: x / (1024**3),
    'DCGM_FI_PROF_PCIE_RX_BYTES': lambda x: x / (1024**3),
    'host_mem_usage': lambda x: x / (1024**3),
}


def _plot_summary(summary, savedir, field):
  plt.clf()

  def _convert_unit(val):
    if field in _PLOT_UNIT_CONVERSION:
      return _PLOT_UNIT_CONVERSION[field](val)
    else:
      return val

  for mode, metrics in summary.items():
    for prec, df in metrics.items():
      if mode == 'serial':
        plt.axhline(
            y=_convert_unit(df['serial:{}:avg'.format(prec)].loc[1]),
            label='serial:{}'.format(prec),
            color=_COLORS[mode],
            linestyle=_LINESTYLES[prec],
        )
        plt.axhspan(
            _convert_unit(df['serial:{}:min'.format(prec)].loc[1]),
            _convert_unit(df['serial:{}:max'.format(prec)].loc[1]),
            facecolor=_COLORS[mode],
            alpha=0.3,
        )
      else:
        plt.plot(
            df.index.values,
            _convert_unit(df['{}:{}:avg'.format(mode, prec)]),
            label='{}:{}'.format(mode, prec),
            color=_COLORS[mode],
            linestyle=_LINESTYLES[prec],
        )
        plt.fill_between(
            df.index.values,
            _convert_unit(df['{}:{}:min'.format(mode, prec)]),
            _convert_unit(df['{}:{}:max'.format(mode, prec)]),
            facecolor=_COLORS[mode],
            alpha=0.3,
        )
  lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.xlabel("B")
  plt.ylabel(_PLOT_LABELS[field])
  plt.rcParams['savefig.dpi'] = 300
  plt.savefig(
      os.path.join(savedir, '{}.png'.format(field)),
      bbox_inches='tight',
  )


def dcgm_parser_main():
  args = _parse_args(_attach_args())
  pathlib.Path(args.savedir).mkdir(parents=True, exist_ok=True)
  summary = {}
  fields = (_get_dcgm_fields_enabled_for(args.device_model) +
            ['cpu_percent', 'host_mem_usage'])
  for field in fields:
    summary[field] = {}
    for mode in args.modes:
      if mode == 'serial':
        summary[field][mode] = _get_serial_metrics(
            args.outdirs,
            args.device_model,
            args.precs,
            args.filename,
            field,
        )
      else:
        summary[field][mode] = _get_hardware_sharing_metrics(
            args.outdirs,
            args.device_model,
            args.precs,
            args.filename,
            mode,
            field,
        )
    pd.concat(
        [
            summary[field][mode][prec]
            for mode in args.modes
            for prec in args.precs
        ],
        axis=1,
    ).to_csv(os.path.join(args.savedir, '{}.csv'.format(field)))
    if args.plot:
      _plot_summary(summary[field], args.savedir, field)
