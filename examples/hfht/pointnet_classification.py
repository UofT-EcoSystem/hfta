import argparse
import logging
import numpy as np
import os
import pandas as pd
import random
import subprocess
from pathlib import Path

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from hfta.hfht import (tune_hyperparameters, attach_common_args,
                       rearrange_algorithm_kwargs, handle_integers,
                       generate_fusible_param_flags, generate_nonfusible_param)
from hfta.workflow import extract_logging_level
from hfta.hfht.utils import fuse_dicts


def main(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  rng_state = np.random.RandomState(seed=args.seed)

  fusibles = {
      'lr': hp.uniform('lr', 0.0001, 0.01),
      'beta1': hp.uniform('beta1', 0.001, 0.999),
      'beta2': hp.uniform('beta2', 0.001, 0.999),
      'weight_decay': hp.uniform('weight_decay', 0.0, 0.5),
      'gamma': hp.uniform('gamma', 0.1, 0.9),
      'step_size': hp.choice('step_size', (5, 10, 20, 40)),
  }
  nonfusibles = {
      'batch_size': hp.choice('batch_size', (8, 16, 32)),
      'feature_transform': hp.choice('feature_transform', (True, False)),
  }

  def _run(results_dir, epochs, iters_per_epoch, params, env_vars=None):
    # Build the cmd.
    cmd = [
        'python',
        'train_classification.py',
        '--epochs',
        str(epochs),
        '--iters-per-epoch',
        str(iters_per_epoch),
        '--dataset',
        args.dataset,
        '--dataset_type',
        args.dataset_type,
        '--num_points',
        str(args.num_points),
        '--device',
        args.device,
        '--eval',
        '--seed',
        str(args.seed),
        '--batch_size',
        str(generate_nonfusible_param(params, 'batch_size')),
    ]
    if results_dir is not None:
      cmd.extend(['--outf', results_dir])
    if generate_nonfusible_param(params, 'feature_transform'):
      cmd.append('--feature_transform')
    cmd.extend(
        generate_fusible_param_flags(
            params,
            ['lr', 'beta1', 'beta2', 'weight_decay', 'gamma', 'step_size'],
        ))
    if args.mode == 'hfta':
      cmd.append('--hfta')
    if args.amp:
      cmd.append('--amp')

    # Launch the training process.
    succeeded = True
    try:
      logging.info('--> Running cmd = {}'.format(cmd))
      subprocess.run(
          cmd,
          stdout=subprocess.DEVNULL if results_dir is None else open(
              os.path.join(results_dir, 'stdout.txt'),
              'w',
          ),
          stderr=subprocess.DEVNULL if results_dir is None else open(
              os.path.join(results_dir, 'stderr.txt'),
              'w',
          ),
          check=True,
          cwd=os.path.join(
              os.path.abspath(os.path.expanduser(os.path.dirname(__file__))),
              '../pointnet/'),
          env=env_vars,
      )
    except subprocess.CalledProcessError as e:
      logging.error(e)
      succeeded = False
    return succeeded

  def try_params(ids, epochs, params, env_vars=None):
    """ Running the training process for pointnet classification task.

    Args:
      ids: Either a single int ID (for serial), or a list of IDs (for HFTA).
      epochs: number of epochs to run.
      params: maps hyperparameter name to its value(s). For HFTA, the values are
        provided as a list.
      env_vars: optional, dict(str, str) that includes extra environment that
        needs to be forwarded to the subprocess call

    Returns:
      result(s): A single result dict for serial or a list of result dicts for
        HFTA in the same order as ids.
      early_stop(s): Whether the training process early stopped. A single bool
        for serial or a list of bools for HFTA in the same order as ids.
    """
    epochs = int(round(epochs))
    ids_str = (','.join([str(i) for i in ids]) if isinstance(
        ids,
        (list, tuple),
    ) else str(ids))
    # Allocate result dir.
    results_dir = os.path.join(args.outdir, ids_str)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    # Run training.
    succeeded = _run(
        results_dir,
        epochs,
        args.iters_per_epoch,
        params,
        env_vars=env_vars,
    )
    if not succeeded:
      raise RuntimeError('_run failed!')
    # Gather the results.
    results_frame = pd.read_csv(os.path.join(results_dir, 'eval.csv'))
    if isinstance(ids, (list, tuple)):
      results = [{'acc': acc} for acc in results_frame['acc'].tolist()]
      assert len(results) == len(ids)
      return results, [False] * len(ids)
    else:
      return {'acc': results_frame['acc'][0]}, False

  def dry_run(
      B=None,
      nonfusibles_kvs=None,
      epochs=None,
      iters_per_epoch=None,
      env_vars=None,
  ):
    params = [{
        **handle_integers(sample(fusibles, rng=rng_state)),
        **nonfusibles_kvs
    } for _ in range(max(B, 1))]
    if B > 0:
      params = fuse_dicts(params)
    else:
      params = params[0]
    return _run(None, epochs, iters_per_epoch, params, env_vars=env_vars)

  tune_hyperparameters(
      space={
          **fusibles,
          **nonfusibles
      },
      try_params_callback=try_params,
      dry_run_callback=dry_run,
      mode=args.mode,
      algorithm=args.algorithm,
      nonfusibles=nonfusibles.keys(),
      dry_run_repeats=args.dry_run_repeats,
      dry_run_epochs=args.dry_run_epochs,
      dry_run_iters_per_epoch=args.dry_run_iters_per_epoch,
      metric='acc',
      goal='max',
      algorithm_configs={
          'hyperband': args.hyperband_kwargs,
          'random': args.random_kwargs,
      },
      seed=args.seed,
      outdir=args.outdir,
  )


def attach_args(parser=argparse.ArgumentParser()):
  parser.add_argument(
      '--workers',
      type=int,
      help='number of data loading workers',
      default=4,
  )
  parser.add_argument(
      '--iters-per-epoch',
      type=int,
      default=int(1e9),
      help='number of epochs to train for',
  )
  parser.add_argument('--dataset', type=str, required=True, help="dataset path")
  parser.add_argument(
      '--dataset-type',
      type=str,
      default='shapenet',
      help="dataset type shapenet|modelnet40",
  )
  parser.add_argument(
      '--num-points',
      type=int,
      default=2500,
      help='num of points for dataset',
  )
  parser.add_argument(
      '--device',
      type=str,
      default='cuda',
      choices=['cpu', 'cuda', 'xla'],
      help="the device where this test is running",
  )
  parser.add_argument(
      '--amp',
      default=False,
      action='store_true',
      help='Enable AMP; only used when --device is cuda',
  )
  parser = attach_common_args(parser)
  return parser


if __name__ == '__main__':
  args = attach_args().parse_args()
  rearrange_algorithm_kwargs(args)
  logging.basicConfig(level=extract_logging_level(args))
  args.outdir = os.path.abspath(os.path.expanduser(args.outdir))
  args.dataset = os.path.abspath(os.path.expanduser(args.dataset))
  main(args)
