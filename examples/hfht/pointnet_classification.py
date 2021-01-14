import argparse
import copy
import json
import numpy as np
import os
import pandas as pd
import random
import subprocess
import sys
from pathlib import Path
from pprint import pprint

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from hfta.hfht.algorithms import RandomSearch, Hyperband
from hfta.hfht.schedule import (SerialScheduler, HFTAScheduler,
                                ConcurrentScheduler, MPSScheduler, MIGScheduler)
from hfta.hfht.utils import (handle_integers, generate_fusible_param_flags,
                             generate_nonfusible_param, to_csv_dicts)


def main(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  rng_state = np.random.RandomState(seed=args.seed)

  if not os.path.isabs(args.outf):
    args.outf = os.path.abspath(args.outf)

  space = {
      'lr': hp.uniform('lr', 0.0001, 0.01),
      'beta1': hp.uniform('beta1', 0.001, 0.999),
      'beta2': hp.uniform('beta2', 0.001, 0.999),
      'weight_decay': hp.uniform('weight_decay', 0.0, 0.5),
      'gamma': hp.uniform('gamma', 0.1, 0.9),
      'step_size': hp.choice('step_size', (5, 10, 20, 40)),
      'batch_size': hp.choice('batch_size', (8, 16, 32)),
      'feature_transform': hp.choice('feature_transform', (True, False)),
  }

  def get_params():
    params = sample(space, rng=rng_state)
    return handle_integers(params)

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
    results_dir = os.path.join(args.outf, ids_str)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    # Build the cmd.
    cmd = [
        'python',
        'train_classification.py',
        '--epochs',
        str(epochs),
        '--iters-per-epoch',
        str(args.iters_per_epoch),
        '--outf',
        results_dir,
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

    # modify the environment if needed
    env = dict(os.environ)
    if env_vars is not None:
      env.update(env_vars)

    # Launch the training process.
    print('--> Running cmd = {}'.format(cmd))
    subprocess.run(
        cmd,
        stdout=open(os.path.join(results_dir, 'stdout.txt'), 'w'),
        stderr=open(os.path.join(results_dir, 'stderr.txt'), 'w'),
        check=True,
        env=env,
        cwd='../pointnet.pytorch/utils/',
    )
    # Gather the results.
    results_frame = pd.read_csv(os.path.join(results_dir, 'eval.csv'))
    if isinstance(ids, (list, tuple)):
      results = [{'acc': acc} for acc in results_frame['acc'].tolist()]
      assert len(results) == len(ids)
      return results, [False] * len(ids)
    else:
      return {'acc': results_frame['acc'][0]}, False

  if args.mode == 'hfta':
    scheduler = HFTAScheduler(
        try_params,
        ['batch_size', 'feature_transform'],
        args.capacity_spec,
    )
  elif args.mode == 'concurrent':
    scheduler = ConcurrentScheduler(
        try_params,
        args.concurrent_width,
    )
  elif args.mode == 'mps':
    scheduler = MPSScheduler(
        try_params,
        args.concurrent_width,
    )
  elif args.mode == 'mig':
    scheduler = MIGScheduler(
        try_params,
        args.concurrent_width,
    )
  else:
    scheduler = SerialScheduler(try_params)

  if args.algorithm == 'hyperband':
    tuner = Hyperband(
        get_params,
        scheduler,
        'acc',
        goal='max',
        max_iters=args.max_iters_per_config,
        eta=args.eta,
        skip_last=args.skip_last,
    )
  elif args.algorithm == 'random':
    # To ensure a fair comparison between Hyperband and Random Search,
    # also pass in hyperband related parameters to decide the size
    # of candidate set, and the number of training epoches.
    tuner = RandomSearch(
        get_params,
        scheduler,
        'acc',
        goal='max',
        n_iters=args.n_iters,
        n_configs=args.n_configs,
    )
  else:
    raise ValueError('Invalid algorithm: {}'.format(args.algorithm))

  history, trajectory = tuner.run()

  print("\n=========================================================")
  print("Done the HyperBand search, the final results are:")
  print("{} total, best:\n".format(len(history)))

  for trial in sorted(
      history.values(),
      key=lambda trial: trial['result']['acc'],
      reverse=True,
  )[:5]:
    print("acc: {:.2%} | {:.1f} iterations | run {} ".format(
        trial['result']['acc'],
        trial['iterations'],
        trial['id'],
    ))
    pprint(trial['params'])

  history_csv, trajectory_csv = to_csv_dicts(space, history, trajectory)
  pd.DataFrame(history_csv).to_csv(os.path.join(args.outf, 'history.csv'))
  pd.DataFrame(trajectory_csv).to_csv(os.path.join(args.outf, 'trajectory.csv'))


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
  parser.add_argument('--outf', type=str, default='cls', help='output folder')
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
      '--mode',
      type=str,
      default='serial',
      choices=['serial', 'hfta', 'concurrent', 'mps', 'mig'],
      help="the GPU sharing mode",
  )
  parser.add_argument(
      '--amp',
      default=False,
      action='store_true',
      help='Enable AMP; only used when --device is cuda',
  )
  parser.add_argument(
      '--seed',
      type=int,
      help='Seed',
      default=1117,
  )
  parser.add_argument(
      '--concurrent-width',
      type=int,
      help='the maximum number of concurrent training processes if concurrent '
      'or mps is enabled; this is determined manually from the "worst case" '
      'situation where the GPU memory footprint of the training process is '
      'largest (i.e., FP32 training, batch_size == 64 and '
      'feature_transform == True)',
      default=3,
  )
  parser.add_argument(
      '--capacity-spec',
      type=str,
      default=None,
      help='Path to a JSON spec file that lists the max numbers of models being'
      'trained simultaneously that the current device support for HFTA',
  )
  parser.add_argument(
      '--max-iters-per-config',
      type=int,
      default=250,
      help='Hyperband maximum iterations per configuration',
  )
  parser.add_argument(
      '--eta',
      type=int,
      default=5,
      help='Hyperband configuration downsampling rate',
  )
  parser.add_argument(
      '--skip-last',
      type=int,
      default=1,
      help='Hyperband skipping last waves of configuration downsampling',
  )
  parser.add_argument(
      '--n-iters',
      type=int,
      default=25,
      help='RandomSearch (constant) iterations per configuration',
  )
  parser.add_argument(
      '--n-configs',
      type=int,
      default=60,
      help='RandomSearch total number of configurations',
  )
  parser.add_argument(
      '--algorithm',
      type=str,
      default='hyperband',
      choices=['hyperband', 'random'],
      help="the hyper-parameter tuning algorithm to use",
  )
  return parser


if __name__ == '__main__':
  parser = attach_args()
  main(parser.parse_args())
