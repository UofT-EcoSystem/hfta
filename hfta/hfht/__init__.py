import logging
import numpy as np
import os
import pandas as pd
from hyperopt.pyll.stochastic import sample
from pathlib import Path
from pprint import pformat

from .schedule import (SerialScheduler, ConcurrentScheduler, MPSScheduler,
                       MIGScheduler, HFTAScheduler)
from .algorithms import RandomSearch, Hyperband
from .utils import handle_integers, attach_common_args


def _assert_positive_int_or_none(v):
  assert (v is None) or (isinstance(v, int) and v > 0)


def _expand_dict_if_valid(d, k, v):
  if v is not None:
    d[k] = v


def tune_hyperparameters(
    space={},
    try_params_callback=None,
    dry_run_callback=None,
    mode='hfta',
    algorithm='hyperband',
    nonfusibles=[],
    dry_run_repeats=None,
    dry_run_epochs=None,
    dry_run_iters_per_epoch=None,
    metric=None,
    goal='min',
    algorithm_configs={},
    seed=0,
    outdir=None,
):
  assert len(space) > 0
  assert try_params_callback is not None
  assert dry_run_callback is not None
  assert mode in {'serial', 'concurrent', 'mps', 'mig', 'hfta'}
  assert algorithm in {'hyperband', 'random'}
  assert nonfusibles is not None
  _assert_positive_int_or_none(dry_run_repeats)
  _assert_positive_int_or_none(dry_run_epochs)
  _assert_positive_int_or_none(dry_run_iters_per_epoch)
  assert metric is not None and isinstance(metric, str)
  assert goal in {'min', 'max'}
  assert isinstance(algorithm_configs, dict)
  assert isinstance(seed, int)
  assert outdir is None or isinstance(outdir, str)

  if mode == 'serial':
    scheduler = SerialScheduler(try_params_callback)
  else:
    kwargs = {'dry_run_callback': dry_run_callback, 'nonfusibles': nonfusibles}
    _expand_dict_if_valid(kwargs, 'dry_run_repeats', dry_run_repeats)
    _expand_dict_if_valid(kwargs, 'dry_run_epochs', dry_run_epochs)
    _expand_dict_if_valid(
        kwargs,
        'dry_run_iters_per_epoch',
        dry_run_iters_per_epoch,
    )
    schedulers = {
        'concurrent': ConcurrentScheduler,
        'mps': MPSScheduler,
        'mig': MIGScheduler,
        'hfta': HFTAScheduler,
    }
    scheduler = schedulers[mode](**kwargs)

  rng_state = np.random.RandomState(seed=seed)

  def get_params():
    params = sample(space, rng=rng_state)
    return handle_integers(params)

  tuners = {'hyperband': Hyperband, 'random': Random}
  # To ensure a fair comparison between Hyperband and Random Search,
  # also pass in hyperband related parameters to decide the size
  # of candidate set, and the number of training epoches.
  tuner = tuners[algorithm](
      get_params,
      scheduler,
      metric,
      goal=goal,
      **algorithm_configs,
  )

  history, trajectory = tuner.run()
  logging.info("\n=========================================================")
  logging.info("Done! the final results are:")
  logging.info("{} total, best:\n".format(len(history)))

  for trial in sorted(
      history.values(),
      key=lambda trial: trial['result'][metric],
      reverse=True,
  )[:5]:
    logging.info("{} | {:.1f}iterations | run {} ".format(
        pformat(trial['result']),
        trial['iterations'],
        trial['id'],
    ))
    pformat(trial['params'])

  if outdir is not None:
    logging.info('Saving results to {} ...'.format(outdir))
    Path(outdir).mkdir(parents=True, exist_ok=True)
    pd.json_normalize(history.values()).to_csv(
        os.path.join(outdir, 'history.csv'))
    pd.json_normalize(trajectory).to_csv(os.path.join(outdir, 'trajectory.csv'))

  return history, trajectory
