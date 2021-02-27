import argparse
import logging
import os
import random
import subprocess
from pathlib import Path

from hfta.workflow import (attach_args as attach_workflow_args, workflow,
                           rearrange_runner_kwargs, extract_logging_level)
from hfta.workflow.utils import _init_precs
from hfta.workflow.dcgm_monitor import DcgmMonitor, dcgm_monitor_start, dcgm_monitor_stop


def ensemble_workflow(args, trial_func):
  args.enable_dcgm = False
  TEST_B = 2
  precs = _init_precs(args.device, args.device_model) if args.precs is None else args.precs
  outdir = os.path.join(args.outdir_root, "ensemble", args.device, args.device_model)
  for prec in precs:
    for serial_num in range(10, -1, -1):
      output_dir = os.path.join(outdir, prec, "serial{}".format(serial_num))
      Path(output_dir).mkdir(parents=True, exist_ok=True)
      if args.enable_dcgm and args.device == 'cuda':
        monitor = DcgmMonitor(args.device_model)
        monitor_thread = dcgm_monitor_start(monitor, output_dir)
      print("{}: running experiment with {} serial layers".format(prec, serial_num))
      succeeded = False
      try:
        succeeded = trial_func(TEST_B, args.device, prec, args.epochs, args.iters_per_epoch, output_dir, serial_num=serial_num)
      finally:
        if args.enable_dcgm and args.device == 'cuda':
          dcgm_monitor_stop(monitor, monitor_thread)
      if not succeeded:
        print("Failed to run resnet with {} serial layers".format(serial_num))
        exit(-1)


def main(args):
  work_path = os.path.join(
                os.path.abspath(os.path.expanduser(os.path.dirname(__file__))),
                '../examples/resnet/',
            )
  def trial(
      B=None,
      device=None,
      prec=None,
      epochs=None,
      iters_per_epoch=None,
      outdir=None,
      env_map=None,
      serial_num=0,
  ):

    cmd = [
        'python',
        'main_ensemble.py' if args.ensemble else 'main.py',
        '--dataset',
        args.dataroot,
        '--epochs',
        str(epochs),
        '--iters-per-epoch',
        str(iters_per_epoch),
        '--device',
        device,
    ]

    if args.ensemble:
      cmd.extend(["--serial_num", str(serial_num)])
    if outdir is not None:
      cmd.extend(['--outf', outdir])
    if prec == 'amp' and device == 'cuda':
      cmd.append('--amp')

    num_hps = max(B, 1)
    hyperparam_strs = {
        'lr': [str(random.uniform(0.1, 10)) for _ in range(num_hps)],
        'gamma': [str(random.uniform(0.3, 0.99)) for _ in range(num_hps)],
    }

    for flag, vals in hyperparam_strs.items():
      if B > 0:
        cmd.extend(['--{}'.format(flag)] + vals)
      else:
        cmd.extend(['--{}'.format(flag), vals[0]])

    if B > 0:
      cmd.append('--hfta')

    succeeded = True
    try:
      subprocess.run(
          cmd,
          stdout=subprocess.DEVNULL if outdir is None else open(
              os.path.join(outdir, 'stdout.txt'),
              'w',
          ),
          stderr=subprocess.DEVNULL if outdir is None else open(
              os.path.join(outdir, 'stderr.txt'),
              'w',
          ),
          check=True,
          cwd=work_path,
          env=env_map,
      )
    except subprocess.CalledProcessError as e:
      logging.error(e)
      succeeded = False
    return succeeded

  if args.ensemble:
    ensemble_workflow(args, trial)
  else:
      if workflow(
          trial_func=trial,
          device=args.device,
          device_model=args.device_model,
          outdir_prefix=args.outdir_root,
          precs=args.precs,
          modes=args.modes,
          enable_dcgm=args.enable_dcgm,
          enable_tpu_profiler=args.enable_tpu_profiler,
          tpu_profiler_waittime=10,
          tpu_profiler_duration=10,
          epochs=args.epochs,
          iters_per_epoch=args.iters_per_epoch,
          concurrent_runner_kwargs=args.concurrent_runner_kwargs,
          mps_runner_kwargs=args.mps_runner_kwargs,
          hfta_runner_kwargs=args.hfta_runner_kwargs,
          mig_runner_kwargs=args.mig_runner_kwargs,
      ):
        logging.info('Done!')
      else:
        logging.error('Failed!')


def attach_args(
    parser=argparse.ArgumentParser('Transformer Benchmark Workflow')):
  parser.add_argument(
      '--outdir_root',
      type=str,
      required=True,
      help='output root dir',
  )
  parser.add_argument(
      '--epochs',
      type=int,
      default=5,
      help='number of epochs',
  )
  parser.add_argument(
      '--iters-per-epoch',
      type=int,
      default=1000,
      help='number of iterations per epochs',
  )
  parser.add_argument(
      '--dataroot',
      type=str,
      required=True,
      help='path to the shapenet parts dataset',
  )
  parser.add_argument(
    '--ensemble',
    action='store_true',
    default=False,
    help='run resnet ensemble experiment',
  )
  parser = attach_workflow_args(parser)
  return parser


if __name__ == '__main__':
  args = attach_args().parse_args()
  rearrange_runner_kwargs(args)
  logging.basicConfig(level=extract_logging_level(args))
  args.outdir_root = os.path.abspath(os.path.expanduser(args.outdir_root))
  args.dataroot = os.path.abspath(os.path.expanduser(args.dataroot))
  main(args)
