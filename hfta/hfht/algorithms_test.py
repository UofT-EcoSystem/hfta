import random
import time
import numpy as np

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

  scheduler = HFTAScheduler(
      try_params,
      ['beta'],
      './capacity_specs/mock/spec.json',
  )
  if algo == 'hyperband':
    algo = Hyperband(get_params, scheduler, 'y', goal='min')
  elif algo == 'random':
    algo = RandomSearch(get_params, scheduler, 'y', goal='min')
  history, trajectory = algo.run()
  print('trajectory = {}'.format(trajectory))


if __name__ == '__main__':

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
