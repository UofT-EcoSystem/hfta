import logging
import time
from math import log, ceil
from functools import partial


class Algorithm:

  def __init__(self, get_params_callback, scheduler, metric, goal='min'):
    self.get_params = get_params_callback
    self.scheduler = scheduler
    self.metric = metric
    assert goal in {'min', 'max'}
    self.goal = goal
    self.optimizer = partial(
        min if goal == 'min' else max,
        key=lambda trial: self._query_metric_if_not_None(trial),
    )
    self.sorter = partial(
        sorted,
        key=lambda trial: trial['result'][self.metric],
        reverse=(goal == 'max'),
    )

    self.history = {}  # map id to dicts of trial info.
    self.id_counter = 0
    self.best_trial = None
    self.trajectory = []

    # The reference starting timestamp for reach `.run()`.
    self.run_reference_tic = None

  def _allocate_ids(self, T):
    allocated_ids = range(self.id_counter, self.id_counter + len(T))
    self.id_counter += len(T)
    return allocated_ids

  def _worst(self):
    return float('inf') if self.goal == 'min' else float('-inf')

  def _query_metric_if_not_None(self, trial):
    return self._worst() if trial is None else trial['result'][self.metric]

  def _now(self):
    return time.perf_counter() - self.run_reference_tic

  def _run(self):
    raise NotImplementedError('Algorithm is an abstract class!')

  def run(self):
    algo_name = type(self).__name__
    logging.info("Running {} ...".format(algo_name))
    self.run_reference_tic = time.perf_counter()
    results = self._run()
    logging.info("Done {} !".format(algo_name))
    return results

  def _build_trials_and_update_search_states(self, n_iters, ids, T, results,
                                             early_stops, runtimes):
    trials = [{
        'id': tid,
        'params': t,
        'result': r,
        'early_stop': es,
        'runtime': rt,
        'iterations': n_iters,
    } for tid, t, r, es, rt in zip(
        ids,
        T,
        results,
        early_stops,
        runtimes,
    )]
    self.best_trial = self.optimizer(trials + [self.best_trial])
    self.trajectory.append({
        'timestamp': self._now(),
        self.metric: self.best_trial['result'][self.metric],
    })
    self.history.update({trial['id']: trial for trial in trials})
    return trials


class RandomSearch(Algorithm):

  def __init__(
      self,
      get_params_callback,
      scheduler,
      metric,
      goal='min',
      n_iters=10,
      n_configs=81,
  ):
    super(RandomSearch, self).__init__(
        get_params_callback,
        scheduler,
        metric,
        goal=goal,
    )
    self.n_iters = n_iters
    self.n_configs = n_configs

  # can be called multiple times
  def _run(self):

    # n random configurations
    T = [self.get_params() for _ in range(self.n_configs)]

    logging.info("\n*** {} configurations x {:.1f} iterations each".format(
        len(T),
        self.n_iters,
    ))

    allocated_ids = self._allocate_ids(T)
    results, early_stops, runtimes = self.scheduler.execute_params_sets(
        allocated_ids,
        self.n_iters,
        T,
    )
    self._build_trials_and_update_search_states(self.n_iters, allocated_ids, T,
                                                results, early_stops, runtimes)
    return self.history, self.trajectory


class Hyperband(Algorithm):

  def __init__(
      self,
      get_params_callback,
      scheduler,
      metric,
      goal='min',
      max_iters=81,
      eta=3,
      skip_last=2,
  ):
    super(Hyperband, self).__init__(
        get_params_callback,
        scheduler,
        metric,
        goal=goal,
    )

    self.max_iters = max_iters  # maximum iterations per configuration
    self.eta = eta  # defines configuration downsampling rate (default = 3)
    self.skip_last = skip_last

    logeta = lambda x: log(x) / log(self.eta)
    self.s_max = int(logeta(self.max_iters))
    self.B = (self.s_max + 1) * self.max_iters

  # can be called multiple times
  def _run(self):

    for s in reversed(range(self.s_max + 1)):

      # initial number of configurations
      n = int(ceil(self.B / self.max_iters / (s + 1) * self.eta**s))

      # initial number of iterations per config
      r = self.max_iters * self.eta**(-s)

      # n random configurations
      T = [self.get_params() for _ in range(n)]

      for i in range((s + 1) - int(self.skip_last)):  # changed from s + 1

        # Run each of the n configs for <iterations>
        # and keep best (n_configs / eta) configurations

        n_configs = n * self.eta**(-i)
        n_iterations = r * self.eta**(i)

        logging.info("\n*** {} configurations x {:.1f} iterations each".format(
            n_configs, n_iterations))

        allocated_ids = self._allocate_ids(T)
        results, early_stops, runtimes = self.scheduler.execute_params_sets(
            allocated_ids,
            n_iterations,
            T,
        )
        trials = self._build_trials_and_update_search_states(
            n_iterations, allocated_ids, T, results, early_stops, runtimes)

        # filter out early stops, if any
        trials = [trial for trial in trials if not trial['early_stop']]
        # select a number of best configurations for the next loop
        T = [trial['params'] for trial in self.sorter(trials)]
        T = T[0:int(n_configs / self.eta)]

    return self.history, self.trajectory
