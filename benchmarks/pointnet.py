import argparse
import logging
import os
import random
import subprocess

from hfta.workflow import (attach_args as attach_workflow_args, workflow,
                           rearrange_runner_kwargs, extract_logging_level)


def main(args):

  def trial(
      B=None,
      device=None,
      prec=None,
      epochs=None,
      iters_per_epoch=None,
      outdir=None,
      env_map=None,
  ):
    cmd = [
        'python',
        'train_{}.py'.format('classification' if args.task ==
                             'cls' else 'segmentation'),
        '--dataset',
        args.dataroot,
        '--feature_transform',
        '--epochs',
        str(epochs),
        '--iters-per-epoch',
        str(iters_per_epoch),
        '--device',
        device,
    ]
    if outdir is not None:
      cmd.extend(['--outf', outdir])
    if prec == 'amp' and device == 'cuda':
      cmd.append('--amp')

    num_hps = max(B, 1)
    hyperparam_strs = {
        'lr': [str(random.uniform(0.0001, 0.003)) for _ in range(num_hps)],
        'beta1': [str(random.uniform(0.8, 0.99)) for _ in range(num_hps)],
        'beta2': [str(random.uniform(0.9, 0.9999)) for _ in range(num_hps)],
        'weight_decay': [str(random.uniform(0.0, 0.3)) for _ in range(num_hps)],
        'gamma': [str(random.uniform(0.0, 0.5)) for _ in range(num_hps)],
        'step_size': [str(random.randint(1, 10) * 10) for _ in range(num_hps)],
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
          cwd=os.path.join(
              os.path.abspath(os.path.expanduser(os.path.dirname(__file__))),
              '../examples/pointnet/',
          ),
          env=env_map,
      )
    except subprocess.CalledProcessError as e:
      logging.error(e)
      succeeded = False
    return succeeded

  if workflow(
      trial_func=trial,
      device=args.device,
      device_model=args.device_model,
      outdir_prefix=os.path.join(args.outdir_root, args.task),
      precs=args.precs,
      modes=args.modes,
      enable_dcgm=args.enable_dcgm,
      epochs=args.epochs,
      iters_per_epoch=args.iters_per_epoch,
      serial_runner_kwargs=args.serial_runner_kwargs,
      concurrent_runner_kwargs=args.concurrent_runner_kwargs,
      mps_runner_kwargs=args.mps_runner_kwargs,
      hfta_runner_kwargs=args.hfta_runner_kwargs,
      mig_runner_kwargs=args.mig_runner_kwargs,
  ):
    logging.info('Done!')
  else:
    logging.error('Failed!')


def attach_args(parser=argparse.ArgumentParser('PointNet Benchmark Workflow')):
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
      '--task',
      type=str,
      required=True,
      choices=['cls', 'seg'],
      help='pointnet classification or segmentation task',
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
