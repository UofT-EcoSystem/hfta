import logging
import numpy as np
import random
import time

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from hfta.hfht.schedule import SerialScheduler, HFTAScheduler
from hfta.hfht.utils import handle_integers
from hfta.hfht.algorithms import RandomSearch, Hyperband


def test_serial(space, algo):

  def get_params():
    params = sample(space)
    return handle_integers(params)

  def try_params(i, n_iterations, t):
    alpha, beta = t['alpha'], t['beta']
    res = {
        'y': (0.1 + alpha * n_iterations / 100)**(-1) + beta * 0.1,
        'z': alpha * beta / n_iterations
    }
    time.sleep(random.uniform(0.0, 0.2))
    return res, False

  scheduler = SerialScheduler(try_params)
  if algo == 'hyperband':
    algo = Hyperband(get_params, scheduler, 'y', goal='min')
  elif algo == 'random':
    algo = RandomSearch(get_params, scheduler, 'y', goal='min')
  history, trajectory = algo.run()
  print('trajectory = {}'.format(trajectory))


def test_hfta(space, algo):

  def get_params():
    params = sample(space)
    return handle_integers(params)

  def try_params(ids, n_iterations, fused_T):
    alpha, beta = np.array(fused_T['alpha']), np.array(fused_T['beta'])
    res = {
        'y': ((0.1 + alpha * n_iterations / 100)**(-1) + beta * 0.1),
        'z': (alpha * beta / n_iterations),
    }
    time.sleep(random.uniform(0.0, 0.2))
    return [{
        'y': y,
        'z': z
    } for y, z in zip(res['y'].tolist(), res['z'].tolist())], [False] * len(ids)

  def dry_run(
      B=None,
      nonfusibles_kvs=None,
      epochs=None,
      iters_per_epoch=None,
      env_vars=None,
  ):
    assert epochs == 2
    assert iters_per_epoch == 3
    assert env_vars is None
    expected_B = nonfusibles_kvs['beta'] * 3 + 1
    return B <= expected_B

  scheduler = HFTAScheduler(
      try_params_callback=try_params,
      dry_run_callback=dry_run,
      nonfusibles=['beta'],
  )
  if algo == 'hyperband':
    algo = Hyperband(get_params, scheduler, 'y', goal='min')
  elif algo == 'random':
    algo = RandomSearch(get_params, scheduler, 'y', goal='min')
  history, trajectory = algo.run()
  print('trajectory = {}'.format(trajectory))


if __name__ == '__main__':
  logging.basicConfig(level='INFO')
  space = {
      'alpha': hp.uniform('alpha', 0.01, 0.1),
      'beta': hp.choice('beta', (1, 2, 3, 4)),
  }
  print('\n********************* Serial HyperBand ************************')
  test_serial(space, 'hyperband')
  print('\n*********************  HFTA HyperBand  ************************')
  test_hfta(space, 'hyperband')
  print('\n******************** Serial RandomSearch ************************')
  test_serial(space, 'random')
  print('\n********************  HFTA RandomSearch  ************************')
  test_hfta(space, 'random')
