import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .dcgm_monitor import dcgm_monitor_thread
from .planner import find_max_B, expovariate_plan


class Runner:

  def __init__(self, trial, probe, outdir_prefix, dcgm_monitor=None):
    """ trial is a function that runs a full experiment.
    The signature of trial is 'def trial(B, outdir) -> bool'
    probe is a function that runs a short experiment for searching max B.
    The signature of probe is 'def probe(B) -> bool'
    """
    self._trial = trial
    self._probe = probe
    self._outdir_prefix = outdir_prefix
    self._Bs = self._init_Bs()
    self._dcgm_monitor = dcgm_monitor

  def _init_Bs(self):
    raise NotImplementedError('Runner is an abstract/interface class!')

  def _run(self, B, outdir):
    raise NotImplementedError('Runner is an abstract/interface class!')

  def dry_run(self, B):
    raise NotImplementedError('Runner is an abstract/interface class!')

  def _outdir(self, B):
    raise NotImplementedError('Runner is an abstract/interface class!')

  def _start_dcgm_monitor_if_needed(self, outdir):
    if self._dcgm_monitor is not None:
      dcgm_monitor_thread = threading.Thread(
          target=dcgm_monitor_thread,
          name='DCGM Monitor Thread',
          args=(self._dcgm_monitor, outdir),
      )

  def _stop_dcgm_monitor_if_needed(self):
    if self._dcgm_monitor is not None:
      self._dcgm_monitor.to_shutdown = True
      dcgm_monitor_thread.join()
      self._dcgm_monitor.reset()

  def run(self):
    self._Bs = self._init_Bs()
    for B in self._Bs:
      outdir = self._outdir(B)
      self._start_dcgm_monitor_if_needed(os.path.join(outdir, 'dcgm_metrics'))
      self._run(B, outdir)
      self._stop_dcgm_monitor_if_needed()


def trial_wrapper(trial, B, outdir):
  Path(outdir).mkdir(parents=True, exist_ok=True)
  return trial(B, outdir)


class SerialRunner(Runner):

  def _init_Bs(self):
    return [1]

  def _outdir(self, B):
    return os.path.join(self._outdir_prefix, 'serial')

  def _run(self, B, outdir):
    assert B == 1
    with ThreadPoolExecutor(max_workers=1) as executor:
      t = executor.submit(trial_wrapper, self._trial, 0, outdir)
      if not t.result():
        raise RuntimeError('SerialRunner: trial_wrapper({}, 0, {}) '
                           'failed!'.format(self._trial, outdir))

  def dry_run(self, B):
    raise NotImplementedError('SerialRunner does not need/support dry_run!')


class ConcurrentRunner(Runner):

  def __init__(
      self,
      trial,
      probe,
      outdir_prefix,
      dcgm_monitor=None,
      dry_run_repeats=10,
      max_num_Bs=5,
      lambd=4.0,
  ):
    super(ConcurrentRunner, self).__init__(
        trial,
        probe,
        outdir_prefix,
        dcgm_monitor=dcgm_monitor,
    )
    self._dry_run_repeats = dry_run_repeats
    self._max_num_Bs = max_num_Bs
    self._lambd = lambd

  def _init_Bs(self):
    max_B = find_max_B(self, dry_run_repeats=self._dry_run_repeats)
    return expovariate_plan(max_B, self._max_num_Bs, lambd=self.lambd)

  def _outdir(self, B):
    return os.path.join(self._outdir_prefix, 'concurrent', 'B{}'.format(B))

  def _run(self, B, outdir):
    with ThreadPoolExecutor(max_workers=B) as executor:
      outdirs = [os.path.join(outdir, 'idx{}'.format(b)) for b in range(B)]
      ts = [
          executor.submit(trial_wrapper, self._trial, 0, outdirs[b])
          for b in range(B)
      ]
      for b, t in enumerate(ts):
        if not t.result():
          raise RuntimeError("ConcurrentRunner: trial_wrapper({}, 0, {}) "
                             "failed!".format(self._trial, outdirs[b]))

  def dry_run(self, B):
    with ThreadPoolExecutor(max_workers=B) as executor:
      ts = [executor.submit(self._probe, 0) for _ in range(B)]
      res = [t.result() for t in ts]
    return all(res)


class HFTARunner(Runner):

  def __init__(
      self,
      trial,
      probe,
      outdir_prefix,
      dcgm_monitor=None,
      dry_run_repeats=1,
      max_num_Bs=10,
      lambd=4.0,
  ):
    super(HFTARunner, self).__init__(
        trial,
        probem,
        outdir_prefix,
        dcgm_monitor=dcgm_monitor,
    )
    self._dry_run_repeats = dry_run_repeats
    self._max_num_Bs = max_num_Bs
    self._lambd = lambd

  def _init_Bs(self):
    max_B = find_max_B(self, dry_run_repeats=self.dry_run_repeats)
    return expovariate_plan(max_B, max_num_Bs, lambd=self._lambd)

  def _outdir(self, B):
    return os.path.join(self._outdir_prefix, 'hfta', 'B{}'.format(B))

  def _run(self, B, outdir):
    with ThreadPoolExecutor(max_workers=1) as executor:
      t = executor.submit(trial_wrapper, self._trial, B, outdir)
      if not t.result():
        raise RuntimeError('HFTARunner: trial_wrapper({}, {}, {}) '
                           'failed!'.format(self._trial, B, outdir))

  def dry_run(self, B):
    with ThreadPoolExecutor(max_workers=1) as executor:
      t = executor.submit(self._probe, B)
      res = t.result()
    return res
