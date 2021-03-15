import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from hfta.workflow.timing import _calculate_throughputs, _LINESTYLES, _COLORS

def get_thoughput(args, precs, start=0, end=10,  normalize=True, devide="cuda"):
  throughputs = {}
  throughputs["serial num"] = np.array(range(start, end + 1))
  ensemble_root = os.path.join(args.outdir, "ensemble", args.device, args.device_model)
  serial_root = os.path.join(args.outdir, args.device, args.device_model)
  for prec in precs:
    serial_path = os.path.join(serial_root, prec, 'serial')
    timing_df = pd.read_csv(os.path.join(serial_path, "timing.csv"))
    throughputs["serial:{}".format(prec)] = _calculate_throughputs([timing_df, ], devide)

    ensemble_throughputs = []
    for i in range(start, end + 1):
      end_outdir_path = os.path.join(ensemble_root, prec, 'serial{}'.format(i))
      timing_df = pd.read_csv(os.path.join(end_outdir_path, "timing.csv"))
      ensemble_throughputs.append(_calculate_throughputs([timing_df, ], devide))
    throughputs["hfta:{}".format(prec)] = np.array(ensemble_throughputs)

  if normalize:
    std = throughputs["serial:fp32"] if "fp32" in precs else throughputs["serial:amp"]
    for prec in precs:
      throughputs["hfta:{}".format(prec)] /= std
      throughputs["serial:{}".format(prec)] /= std

  return throughputs


def polt_thoughtput(throughputs, precs, save_path):
  plt.clf()
  for prec in precs:
    for mode in ["hfta", "serial"]:
      data = throughputs["{}:{}".format(mode, prec)]
      if mode == "serial":
        plt.axhline(
          y=data,
          label="{}:{}".format(mode, prec),
          color=_COLORS[mode],
          linestyle=_LINESTYLES[prec],
        )
      else:
        plt.plot(
          throughputs["serial num"],
          data,
          label="{}:{}".format(mode, prec),
          color=_COLORS[mode],
          linestyle=_LINESTYLES[prec],
        )
  lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.title('HFTA Patrialy Fused Throughputs')
  plt.xlabel("serial layers")
  plt.ylabel("Normalized Throughput")
  plt.rcParams['savefig.dpi'] = 300
  plt.savefig('{}/throughputs.png'.format(save_path), bbox_inches='tight')

  df = pd.DataFrame(throughputs).set_index("serial num")
  print(df)
  df.to_csv(os.path.join(save_path, "throughputs.csv"))

def main(args):
  throughputs = get_thoughput(args, args.precs)
  polt_thoughtput(throughputs, args.precs, args.save_dir if args.save_dir is not None else args.outdir)
  pass


def _attach_args(
        parser=argparse.ArgumentParser(description='Timing Results Parser')):
  parser.add_argument(
    '--outdir',
    type=str,
    required=True,
    help='path(s) to the workflow outdir_prefix.',
  )
  parser.add_argument(
    '--device',
    type=str,
    default="cuda",
    choices=['cpu', 'cuda', 'xla'],
    help='cpu, cuda or xla',
  )
  parser.add_argument(
    '--device-model',
    type=str,
    default="v100",
    help='The model of the device (e.g, v100, a100, rtx6000 or TPU v3)',
  )
  parser.add_argument(
    '--precs',
    type=str,
    default=['fp32', 'amp'],
    choices=['fp32', 'amp', 'bf16'],
    nargs='*',
    help='training precision(s)',
  )
  parser.add_argument(
    '--save_dir',
    type=str,
    default=None,
    help='the file path to save the result summary',
  )
  return parser

if __name__ == "__main__":
  main(_attach_args().parse_args())