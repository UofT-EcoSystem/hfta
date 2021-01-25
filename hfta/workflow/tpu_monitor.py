import logging
import os
import threading
import time
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import scipy.interpolate
import threading
import json

from .utils import run_command, _init_modes, _init_precs
from .timing import _aggregate_along_rows, _LINESTYLES, _COLORS

try:
  from tensorflow.python.profiler import profiler_client
  from tensorflow.errors import UnavailableError
except ImportError:
  pass


class TpuMonitor:
  """
  TPU performance monitor & result parser
  """

  def logging_format(self, msg):
    return '{}: {}'.format("TpuMonitor", msg)

  def debug(self, msg):
    logging.debug(self.logging_format(msg))

  def info(self, msg):
    logging.info(self.logging_format(msg))

  def warning(self, msg):
    logging.warning(self.logging_format(msg))

  def error(self, msg):
    logging.error(self.logging_format(msg))

  def __init__(self, wait_time, duration, outdir):
    self.info(
        "Start TPU Monitor and it will monitor for {} seconds after waiting for {} seconds."
        .format(duration / 1000, wait_time))
    self.args = self.get_profiler_args(duration, outdir)
    self.wait_time = wait_time

  def get_profiler_args(self, duration, outdir):
    self.debug(
        "Initialize TPU profiler arguments with outdir: {}".format(outdir))
    dir_list = outdir.split("/")
    idx = 0
    for i in range(len(dir_list)):
      if dir_list[i] == "benchmarks":
        idx = i + 1
        break

    logdir = "/".join(dir_list[idx:])
    ret = {
        "service_addr":
            "{}:{}".format(os.environ.get("TPU_IP_ADDRESS"), "8466"),
        "logdir":
            "{}/{}".format(os.environ.get("STORAGE_BUCKET"), logdir),
        "duration_ms":
            duration,
        "worker_list":
            '',
        "num_tracing_attempts":
            10,
        "options":
            None
    }
    logging.debug(ret)
    return ret

  def start_monitoring(self):
    success = False
    sleep_time = 2

    # Sleep for wait_time seconds to avoid the training warmup
    time.sleep(self.wait_time)

    while not success:
      try:
        profiler_client.trace(**self.args)
      except UnavailableError as e:
        self.warning(
            "Failed to capture TPU profile, retry in {} seconds".format(
                sleep_time))
        time.sleep(sleep_time)
      else:
        success = True
        self.info("Successfully captured TPU profile")


def tpu_monitor_thread(monitor):

  # Check tensorflow installation
  try:
    from tensorflow.python.profiler import profiler_client
    from tensorflow.errors import UnavailableError
  except ImportError:
    logging.error(
        "Failed to start TPU monitor thread because tensorflow packages cannot be imported. Please install tensorflow first."
    )
    logging.info("Continue the TPU experiment without running TPU profiler.")
    return

  # Check necessary env vars
  for env_var in ["TPU_NAME", "TPU_IP_ADDRESS", "STORAGE_BUCKET"]:
    if os.environ.get(env_var) is None:
      logging.error(
          "Failed to start TPU monitor thread because {} was not defined.".
          format(env_var))
      logging.info("Continue the TPU experiment without running TPU profiler.")
      return
    else:
      logging.debug("{} is {}".format(env_var, os.environ.get(env_var)))

  monitor.start_monitoring()


def tpu_monitor_start(monitor):
  logging.debug("Start TPU monitoring thread")
  t = threading.Thread(
      target=tpu_monitor_thread,
      name='TPU Monitor Thread',
      args=(monitor,),
  )
  t.start()
  return t


def tpu_monitor_stop(monitor, thread):
  logging.debug("Stop TPU monitoring thread")
  thread.join()


# gsutil cp -r ${STORAGE_BUCKET}/cls/**.overview_page.json ./

# Namespace(
#     device='cuda',
#     device_model='v3',
#     filename='dcgm_metrics.csv',
#     modes=['serial', 'concurrent', 'mps', 'hfta'],
#     outdirs=[
#         '/home/yuxuan950427/repo/HFTA-internal/MLSys21/benchmarks/pointnet/run1/cls'
#         '/home/yuxuan950427/repo/HFTA-internal/MLSys21/benchmarks/pointnet/run2/cls'
#         '/home/yuxuan950427/repo/HFTA-internal/MLSys21/benchmarks/pointnet/run3/cls'
#     ],
#     plot=True,
#     precs=['fp32', 'amp'],
#     savedir='../MLSys21/benchmarks/pointnet/dcgm-cls-xla-v3/')

# gs://fusion-profiling/pointnet/run1/cls/xla/v3/bf16/hfta/B1/ gs://fusion-profiling/pointnet/run1/cls/xla/v3/bf16/hfta/B10/ gs://fusion-profiling/pointnet/run1/cls/xla/v3/bf16/hfta/B11/ gs://fusion-profiling/pointnet/run1/cls/xla/v3/bf16/hfta/B2/ gs://fusion-profiling/pointnet/run1/cls/xla/v3/bf16/hfta/B3/ gs://fusion-profiling/pointnet/run1/cls/xla/v3/bf16/hfta/B4/ gs://fusion-profiling/pointnet/run1/cls/xla/v3/bf16/hfta/B5/ gs://fusion-profiling/pointnet/run1/cls/xla/v3/bf16/hfta/B6/ gs://fusion-profiling/pointnet/run1/cls/xla/v3/bf16/hfta/B7/


def _attach_args(
    parser=argparse.ArgumentParser(description='TPU Metric Parser')):
  parser.add_argument(
      '--outdirs',
      type=str,
      required=True,
      help=
      'a string containing paths (space-delimited) to the profile location for multiple runs',
  )
  parser.add_argument(
      '--device-model',
      type=str,
      required=False,
      default='v3',
      help='The model of the device (e.g, v3)',
  )
  parser.add_argument(
      '--precs',
      type=str,
      default=None,
      choices=['bf16'],
      nargs='*',
      help='training precision(s)',
  )
  parser.add_argument(
      '--modes',
      type=str,
      default=None,
      choices=['serial', 'hfta'],
      nargs='*',
      help='hardware sharing mode(s)',
  )
  parser.add_argument(
      '--savedir',
      type=str,
      required=True,
      help='the path of dir to save the result summary (and the plots if --plot '
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
  # Parse the outdirs string to a list
  args.outdirs = [outdir for outdir in args.outdirs.split()]
  args.device = 'xla'
  if args.precs is None:
    args.precs = _init_precs(args.device, args.device_model)
  if args.modes is None:
    args.modes = _init_modes(args.device, args.device_model)
  if args.plot:
    assert 'serial' in args.modes
  return args


def _percentage_str_to_float(string):
  return float(string.strip("%")) / 100.0


def _get_serial_metrics(
    outdirs,
    device_model,
    precs,
    field,
):
  """ The result is in the format of
    {
      'bf16': pd.DataFrame,  # df only contains 1 row for B=1
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
      end_outdir_path = os.path.join(outdir, 'xla', device_model, prec,
                                     'serial')
      profile_file = run_command(
          "gsutil ls {}/**.overview_page.json".format(end_outdir_path))
      data = json.loads(run_command("gsutil cat {}".format(profile_file)))

      found_field = False
      for item in data:
        if item.get("p"):
          if item["p"].get(field):
            # Successfully found the field data
            result = item["p"][field]  # Ex. "83.0%"
            metrics[prec]['serial:{}:{}'.format(
                prec, outdir_idx)] = [_percentage_str_to_float(result)]
            found_field = True

      if not found_field:
        logging.error("Cannot find field {} from profile file {}".format(
            field, profile_file))

    metrics[prec] = pd.DataFrame(metrics[prec]).set_index('B')
    _aggregate_along_rows(metrics[prec], 'serial', prec)

  return metrics


def _get_hardware_sharing_metrics(
    outdirs,
    device_model,
    precs,
    mode,
    field,
):
  """ The result is in the format of
    {
      'bf16': pd.DataFrame,  # df contains max_B rows
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
      mode_outdir_path = os.path.join(outdir, 'xla', device_model, prec, mode)

      B_subdir = [
          path.split("/")[-2] for path in run_command(
              "gsutil ls -d {}/B*".format(mode_outdir_path)).split()
      ]

      for B_exp in B_subdir:
        B = int(B_exp[1:])
        Bs.append(B)
        B_outdir_path = os.path.join(mode_outdir_path, B_exp)

        profile_file = run_command(
            "gsutil ls {}/**.overview_page.json".format(B_outdir_path))
        data = json.loads(run_command("gsutil cat {}".format(profile_file)))

        found_field = False
        for item in data:
          if item.get("p"):
            if item["p"].get(field):
              # Successfully found the field data
              result = item["p"][field]  # Ex. "83.0%"
              metrics_of_Bs.append(_percentage_str_to_float(result))
              found_field = True

        if not found_field:
          logging.error("Cannot find field {} from profile file {}".format(
              field, profile_file))

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
    "device_duty_cycle_percent": "TPU Duty Cycle Percentage",
    "memory_bw_utilization_relative_to_hw_limit": "Memory Usage Percentage",
    "mxu_utilization_percent": "MXU Utilization Percentage",
}


def _plot_summary(summary, savedir, field):
  plt.clf()

  for mode, metrics in summary.items():
    for prec, df in metrics.items():
      if mode == 'serial':
        plt.axhline(
            y=df['serial:{}:avg'.format(prec)].loc[1],
            label='serial:{}'.format(prec),
            color=_COLORS[mode],
            linestyle=_LINESTYLES[prec],
        )
        plt.axhspan(
            df['serial:{}:min'.format(prec)].loc[1],
            df['serial:{}:max'.format(prec)].loc[1],
            facecolor=_COLORS[mode],
            alpha=0.3,
        )
      else:
        plt.plot(
            df.index.values,
            df['{}:{}:avg'.format(mode, prec)],
            label='{}:{}'.format(mode, prec),
            color=_COLORS[mode],
            linestyle=_LINESTYLES[prec],
        )
        plt.fill_between(
            df.index.values,
            df['{}:{}:min'.format(mode, prec)],
            df['{}:{}:max'.format(mode, prec)],
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


def _get_tpu_profile_fields_enabled_for():
  return [
      "device_duty_cycle_percent",
      "memory_bw_utilization_relative_to_hw_limit",
      "mxu_utilization_percent",
  ]


def tpu_profile_parser_main():
  args = _parse_args(_attach_args())
  pathlib.Path(args.savedir).mkdir(parents=True, exist_ok=True)
  summary = {}
  fields = _get_tpu_profile_fields_enabled_for()

  print(args)

  for field in fields:
    summary[field] = {}
    for mode in args.modes:
      if mode == 'serial':
        summary[field][mode] = _get_serial_metrics(args.outdirs,
                                                   args.device_model,
                                                   args.precs, field)
      else:
        summary[field][mode] = _get_hardware_sharing_metrics(
            args.outdirs, args.device_model, args.precs, mode, field)

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
