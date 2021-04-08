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


def partially_fused_workflow(args, trial_func):
  TEST_B = args.partially_fused_models
  precs = _init_precs(args.device,
                      args.device_model) if args.precs is None else args.precs
  outdir = os.path.join(args.outdir_root, "partially_fused", args.device,
                        args.device_model)
  for prec in precs:
    for serial_num in range(11):
      output_dir = os.path.join(outdir, prec, "serial{}".format(serial_num))
      Path(output_dir).mkdir(parents=True, exist_ok=True)
      if args.enable_dcgm and args.device == 'cuda':
        monitor = DcgmMonitor(args.device_model)
        monitor_thread = dcgm_monitor_start(monitor, output_dir)
      print("{}: running experiment with {} serial layers".format(
          prec, serial_num))
      succeeded = False
      try:
        succeeded = trial_func(TEST_B,
                               args.device,
                               prec,
                               args.epochs,
                               args.iters_per_epoch,
                               output_dir,
                               serial_num=serial_num)
      finally:
        if args.enable_dcgm and args.device == 'cuda':
          dcgm_monitor_stop(monitor, monitor_thread)
      if not succeeded:
        print("Failed to run resnet with {} serial layers".format(serial_num))
        exit(-1)


def convergence_workflow(args, trial_func):
  lrs = args.convergence_lrs
  TEST_B = len(lrs)  # for batch_size=128
  gammas = [str(random.uniform(0.3, 0.99)) for _ in range(TEST_B)]
  precs = [
      "fp32",
  ]
  outdir = os.path.join(args.outdir_root, args.device, args.device_model)
  device = args.device
  epoch = args.epochs
  iters = args.iters_per_epoch
  for prec in precs:
    output_dir = os.path.join(outdir, prec)
    model_path = os.path.join(output_dir, "model.pth")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    extra_flag = [
        "--convergence_test", "--save_init_model", "--model_dir", model_path
    ]
    extra_flag.extend(["--lr", lrs[0], "--gamma", gammas[0]])
    print("Saving initinal model")
    succeeded = trial_func(0, device, prec, 0, 0, None, extra_flags=extra_flag)
    if not succeeded:
      print("Failed to save initial model")
      exit(-1)

    extra_flag = [
        "--convergence_test", "--load_init_model", "--model_dir", model_path
    ]

    for i in range(TEST_B):
      print("Running Pytorch example with lr={}".format(lrs[i]))
      output_dir_now = os.path.join(output_dir, "serial",
                                    "lr_{}".format(lrs[i]))
      Path(output_dir_now).mkdir(parents=True, exist_ok=True)
      extra_flag_now = extra_flag + ["--lr", lrs[i], "--gamma", gammas[i]]
      succeeded &= trial_func(0,
                              device,
                              prec,
                              epoch,
                              iters,
                              output_dir_now,
                              extra_flags=extra_flag_now)
    output_dir_now = os.path.join(output_dir, "hfta")
    Path(output_dir_now).mkdir(parents=True, exist_ok=True)
    extra_flag.extend(["--lr"] + lrs + ["--gamma"] + gammas)
    print("Running HFTA example")
    succeeded = trial_func(TEST_B,
                           args.device,
                           prec,
                           args.epochs,
                           args.iters_per_epoch,
                           output_dir_now,
                           extra_flags=extra_flag)

    if not succeeded:
      print("Failed to run resnet with convergence_test")
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
      extra_flags=None,
  ):
    cmd = [
        'python',
        'main_partially_fused.py' if args.partially_fused else 'main.py',
        '--dataset',
        args.dataroot,
        '--epochs',
        str(epochs),
        '--iters-per-epoch',
        str(iters_per_epoch),
        '--device',
        device,
    ]

    if args.partially_fused:
      cmd.extend(["--serial_num", str(serial_num)])
    if outdir is not None:
      cmd.extend(['--outf', outdir])
    if prec == 'amp' and device == 'cuda':
      cmd.append('--amp')
    if args.convergence and extra_flags is not None:
      cmd.extend(extra_flags)

    num_hps = max(B, 1)
    if extra_flags is None or (not ("--lr" in extra_flags)):
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

  if args.convergence:
    convergence_workflow(args, trial)
    logging.info('Done!')
    return

  if args.partially_fused:
    partially_fused_workflow(args, trial)
    args.modes = [
        "serial",
    ]

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
      '--partially-fused',
      action='store_true',
      default=False,
      help='run resnet partially_fused experiment',
  )
  parser.add_argument(
      '--partially-fused-models',
      type=int,
      default=30,  # default for V100-16G
      help='Number of models for partially fused experiments')
  parser.add_argument(
      '--convergence',
      action='store_true',
      default=False,
      help='run resnet convergence experiment',
  )
  parser.add_argument(
      '--convergence-lrs',
      type=float,
      default=["0.002", "0.001", "0.0005"],
      nargs='*',
      help='convergence learning rate, default=["0.002", "0.001", "0.0005"]')

  parser = attach_workflow_args(parser)
  return parser


if __name__ == '__main__':
  args = attach_args().parse_args()
  rearrange_runner_kwargs(args)
  logging.basicConfig(level=extract_logging_level(args))
  args.outdir_root = os.path.abspath(os.path.expanduser(args.outdir_root))
  args.dataroot = os.path.abspath(os.path.expanduser(args.dataroot))
  main(args)
