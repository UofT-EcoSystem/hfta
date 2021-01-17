import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from .partition import (build_sets, disassemble_sets,
                        partition_hyperparameter_sets_by_capacity)
from .utils import (hash_dict, run_command, resolve_overlap_runtimes,
                    fuse_dicts)
from ..workflow.plan import find_max_B


class Scheduler:

  def logging_format(self, msg):
    return '{}: {}'.format(self.mode, msg)

  def debug(self, msg):
    logging.debug(self.logging_format(msg))

  def info(self, msg):
    logging.info(self.logging_format(msg))

  def warning(self, msg):
    logging.warning(self.logging_format(msg))

  def error(self, msg):
    logging.error(self.logging_format(msg))

  def critical(self, msg):
    logging.critical(self.logging_format(msg))

  @property
  def mode(self):
    raise NotImplementedError('Runner is an abstract/interface class!')

  def execute_params_sets(self, ids, n_iterations, T):
    """
      Returns:
        results
        early_stops
        seconds
    """
    raise NotImplementedError('This is an interface / abstract class.')


class SerialScheduler(Scheduler):

  def __init__(self, try_params_callback):
    self.try_params = try_params_callback

  @property
  def mode(self):
    return 'serial'

  def execute_params_sets(self, ids, n_iterations, T):
    results, early_stops, runtimes = [], [], []
    for i, t in zip(ids, T):
      self.info('Running id={}, t={}...'.format(i, t))
      tic = time.perf_counter()
      res, es = self.try_params(i, n_iterations, t)
      rt = time.perf_counter() - tic
      self.info('==> res={}, early_stop={}, runtime={}'.format(res, es, rt))
      results.append(res)
      early_stops.append(es)
      runtimes.append(rt)
    return results, early_stops, runtimes


class HardwareSharingScheduler(Scheduler):

  def __init__(
      self,
      dry_run_callback=None,
      dry_run_repeats=None,
      dry_run_epochs=None,
      dry_run_iters_per_epoch=None,
      nonfusibles=None,
      B_limit=None,
  ):
    self._dry_run = dry_run_callback
    self._dry_run_repeats = dry_run_repeats
    self._dry_run_epochs = dry_run_epochs
    self._dry_run_iters_per_epoch = dry_run_iters_per_epoch
    self._nonfusibles = nonfusibles
    self._max_Bs = {}  # map hash_dict(nonfusibles_kvs) to max_B
    self._B_limit = B_limit

  def _try_B(self, B, nonfusibles_kvs):
    raise NotImplementedError(
        'HardwareSharingScheduler is an abstract/interface class!')

  def _find_max_B(self, t):
    nonfusibles_kvs = {nf: t[nf] for nf in self._nonfusibles}
    self.info('Querying max_B for nonfusible '
              'hyper-parameters {} ...'.format(nonfusibles_kvs))
    nonfusibles_key = hash_dict(nonfusibles_kvs)
    if nonfusibles_key not in self._max_Bs:
      self.info('max_B for nonfusibles hyper-parameters {} is unknown! '
                'Searching...'.format(nonfusibles_kvs))

      def try_B(B):
        self.info('Trying B = {} ...'.format(B))
        succeeded = self._try_B(B, nonfusibles_kvs)
        if succeeded:
          self.info('--> OK')
        else:
          self.info('--> FAIL')
        return succeeded

      max_B = find_max_B(
          try_B,
          dry_run_repeats=self._dry_run_repeats,
          B_limit=self._B_limit,
      )
      self.info('Found max_B = {} !'.format(max_B))
      self._max_Bs[nonfusibles_key] = max_B
      self.info('Now the mapping from nonfusible hyper-parameters to '
                'max_B becomes {}'.format(self._max_Bs))
    max_B = self._max_Bs[nonfusibles_key]
    self.info('Queried max_B = {} !'.format(max_B))
    return max_B


class ConcurrentScheduler(HardwareSharingScheduler):

  def __init__(
      self,
      try_params_callback=None,
      dry_run_callback=None,
      dry_run_repeats=10,
      dry_run_epochs=2,
      dry_run_iters_per_epoch=10,
      nonfusibles=None,
  ):
    super(ConcurrentScheduler, self).__init__(
        dry_run_callback=dry_run_callback,
        dry_run_repeats=dry_run_repeats,
        dry_run_epochs=dry_run_epochs,
        dry_run_iters_per_epoch=dry_run_iters_per_epoch,
        nonfusibles=nonfusibles,
    )
    self.try_params = try_params_callback
    self.sudo = "" if os.geteuid() == 0 else "sudo"

  @property
  def mode(self):
    return 'concurrent'

  def _setup(self, num_concurrent):
    pass

  def _teardown(self):
    pass

  def _one_params_set_setup(self, id, n_iterations, t):
    # must return a dictionary of additional environment variables
    # or NoneType for now
    return None

  def _one_params_set_teardown(self, id, n_iterations, t):
    pass

  def _execute_one_params_set(self, id, n_iterations, t):
    self.info('Running id={}, t={}...'.format(id, t))

    env_vars = self._one_params_set_setup(id, n_iterations, t)

    tic = time.perf_counter()
    res, es = self.try_params(id, n_iterations, t, env_vars=env_vars)
    toc = time.perf_counter()

    self._one_params_set_teardown(id, n_iterations, t)

    self.info('==> res={}, early_stop={}, runtime(not normalized)={}'.format(
        res, es, toc - tic))
    return res, es, (tic, toc)

  def execute_params_sets(self, ids, n_iterations, T):
    num_concurrent = min([self._find_max_B(t) for t in T])
    self.info('Concluded that num_concurrent = {}!'.format(num_concurrent))
    return self._execute_params_sets(ids, n_iterations, T, num_concurrent)

  def _try_B(self, B, nonfusibles_kvs):

    def dry_run_wrapper(b):
      env_vars = self._one_params_set_setup(b, 0, None)
      status = self._dry_run(
          B=0,
          nonfusibles_kvs=nonfusibles_kvs,
          epochs=self._dry_run_epochs,
          iters_per_epoch=self._dry_run_iters_per_epoch,
          env_vars=env_vars,
      )
      self._one_params_set_teardown(b, 0, None)
      return status

    self._setup(B)
    with ThreadPoolExecutor(max_workers=B) as executor:
      threads = [executor.submit(dry_run_wrapper, b) for b in range(B)]
      succeeded = all([thread.result() for thread in threads])
    self._teardown()
    return succeeded

  def _execute_params_sets(self, ids, n_iterations, T, num_concurrent):
    results, early_stops, runtimes = [], [], []

    self._setup(num_concurrent)

    try:
      with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        # create a pool of threads
        # launching multiple evaluations asynchronously that may use more
        # processes
        multiple_threads = [
            executor.submit(self._execute_one_params_set,
                            *(ids[i], n_iterations, T[i]))
            for i in range(len(T))
        ]
        multiple_results = [res.result() for res in multiple_threads]
        results = [res[0] for res in multiple_results]
        early_stops = [res[1] for res in multiple_results]
        runtimes_raw = [res[2] for res in multiple_results]
        # we need to find the overlaps for concurrent execution and record
        # the runtime as a fraction of the overlapped duration depending on
        # the number of concurrently running processes
        runtimes = resolve_overlap_runtimes(runtimes_raw)
    finally:
      self._teardown()

    return results, early_stops, runtimes


class MPSScheduler(ConcurrentScheduler):

  def __init__(self, **kwargs):
    super(MPSScheduler, self).__init__(**kwargs)
    self._orig_env = None

  @property
  def mode(self):
    return 'mps'

  # overrides base
  def _setup(self, num_concurrent):
    self._orig_env = dict(os.environ)
    self.info("Set up the environment to use MPS ...")
    cmds_to_run = [
        "{} nvidia-smi -i 0 -c EXCLUSIVE_PROCESS".format(self.sudo),
        "nvidia-cuda-mps-control -d",
    ]
    for cmd in cmds_to_run:
      run_command(cmd)
    # modify the environment, this will be inherited by
    # all subprocesses invoked from self.try_params
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"

  # overrides base
  def _teardown(self):
    self.info("Clean up the environment to use MPS ...")
    cmds_to_run = [
        "{} nvidia-cuda-mps-control ".format(self.sudo),
        "{} nvidia-smi -i 0 -c 0".format(self.sudo),
    ]
    for cmd in cmds_to_run:
      stdinput = None
      if "nvidia-cuda-mps-control" in cmd:
        stdinput = "quit"
      run_command(cmd, stdinput)

    # restore the environment
    os.environ.clear()
    os.environ.update(self._orig_env)


class MIGScheduler(ConcurrentScheduler):

  MIG_PROFILE_CONFIGS = {
      1: '0',  # 1 * 7/7
      2: '9,9',  # 2 * 3 / 7
      3: '14,14,14',  # 3 * 2/7
      4: '14,14,14,19',  # 3 * 2/7 + 1 * 1/7
      5: '14,14,19,19,19',  # 2 * 2/7 + 3 * 1/7
      6: '14,19,19,19,19,19',  # 1 * 2/7 + 5 * 1/7
      7: '19,19,19,19,19,19,19',  # 7 * 1/7
  }

  def __init__(self, **kwargs):
    super(MIGScheduler, self).__init__(B_limit=7, **kwargs)

    # check that mig is enabled
    mgi_query_str = ("nvidia-smi  --query-gpu=mig.mode.current "
                     "--format=csv,noheader")
    mig_query_res = run_command(mgi_query_str)
    if "Enabled" not in mig_query_res:
      raise RuntimeError("Mig is not enabled, please run "
                         "\"{} nvidia-smi  -mig 1\" and reboot the machine "
                         "to enable it".format(self.sudo))

    self.orig_env = None
    self.mig_GPU_IDs = None
    self.device_queue = Queue()
    self.param_id_dev_map = {}

  @property
  def mode(self):
    return 'mig'

  def _setup(self, num_concurrent):
    if num_concurrent > 7 or num_concurrent <= 0:
      raise ValueError("MIG does not support concurrent workload> 7 or <=0")
    self.orig_env = dict(os.environ)
    # clean up the current instances
    # could be the case that none exists but that is OK
    self._destroy_mig_instances()
    # create new instances based on concurrent width
    self._create_mig_instances(num_concurrent)

  def _teardown(self):
    self.info("Clean up the environment for MIG ...")

    self._destroy_mig_instances()
    # restore the environment
    os.environ.clear()
    os.environ.update(self.orig_env)

  def _one_params_set_setup(self, id, n_iterations, t):
    dev = self._acquire_mig_instance()
    env_vars = {"CUDA_VISIBLE_DEVICES": dev, **os.environ}
    self.param_id_dev_map[id] = dev
    return env_vars

  def _one_params_set_teardown(self, id, n_iterations, t):
    dev = self.param_id_dev_map[id]
    self._release_mig_instance(dev)
    del self.param_id_dev_map[id]

  def _create_mig_instances(self, num_concurrent):
    self.info("Set up the environment to use MIG ...")

    mig_profile_str = MIGScheduler.MIG_PROFILE_CONFIGS[num_concurrent]

    # create virtual MIG GPU instances
    cmds_to_run = [
        "{} nvidia-smi mig -cgi {}".format(self.sudo, mig_profile_str),
        "{} nvidia-smi mig -cci".format(self.sudo),
    ]
    for cmd in cmds_to_run:
      run_command(cmd)

    self.mig_GPU_IDs = self._query_mig_devices(num_concurrent)
    for dev_id in self.mig_GPU_IDs:
      self.device_queue.put(dev_id)

  def _destroy_mig_instances(self):
    # destroy compute instance
    # then destroy MIG GPU instance
    cmds_to_run = [
        "{} nvidia-smi mig  -dci".format(self.sudo),
        "{} nvidia-smi mig  -dgi".format(self.sudo),
    ]
    for cmd in cmds_to_run:
      run_command(cmd, ignore_error=True)

  def _query_mig_devices(self, num_concurrent):
    # query the devices
    get_gpu_dev_str = "nvidia-smi -L"
    GPU_IDs_raw = run_command(get_gpu_dev_str).split("\n")
    mig_GPU_IDs_raw = [s for s in GPU_IDs_raw if "MIG" in s]
    assert len(mig_GPU_IDs_raw) == num_concurrent

    # parse the output to reflect the actual output
    search_str = "MIG.*Device.*UUID: (.*)\)"
    mig_GPU_IDs = []
    for s in mig_GPU_IDs_raw:
      try:
        MIG_Id = re.search(search_str, s).groups()[0]
        mig_GPU_IDs.append(MIG_Id)
      except Exception as er:
        raise RuntimeError("Parse Error when searching MIG GPU device from "
                           "string {}\nError: {}".format(s, er))

    return mig_GPU_IDs

  def _acquire_mig_instance(self):

    assert self.mig_GPU_IDs is not None
    dev_id = self.device_queue.get()
    # modify the environment
    return dev_id

  def _release_mig_instance(self, dev):

    self.device_queue.put(dev)


class HFTAScheduler(HardwareSharingScheduler):

  def __init__(
      self,
      try_params_callback=None,
      dry_run_callback=None,
      dry_run_repeats=1,
      dry_run_epochs=2,
      dry_run_iters_per_epoch=3,
      nonfusibles=None,
  ):
    super(HFTAScheduler, self).__init__(
        dry_run_callback=dry_run_callback,
        dry_run_repeats=dry_run_repeats,
        dry_run_epochs=dry_run_epochs,
        dry_run_iters_per_epoch=dry_run_iters_per_epoch,
        nonfusibles=nonfusibles,
    )
    self.try_params = try_params_callback

  @property
  def mode(self):
    return 'hfta'

  def _update_max_Bs_if_needed(self, T):
    [self._find_max_B(t) for t in T]

  def _try_B(self, B, nonfusibles_kvs):

    succeeded = self._dry_run(
        B=B,
        nonfusibles_kvs=nonfusibles_kvs,
        epochs=self._dry_run_epochs,
        iters_per_epoch=self._dry_run_iters_per_epoch,
        env_vars=None,
    )

    return succeeded

  def execute_params_sets(self, ids, n_iterations, T):
    self._update_max_Bs_if_needed(T)
    sets = build_sets(ids, T)
    partitions = partition_hyperparameter_sets_by_capacity(
        sets,
        self._nonfusibles,
        self._max_Bs,
    )
    partitions_ids, partitions_T = disassemble_sets(partitions)
    results, early_stops, runtimes = {}, {}, {}
    for partition_ids, partition_T in zip(partitions_ids, partitions_T):
      self.info('Running partition_ids={}, partition_T={}'.format(
          partition_ids, partition_T))
      # Generate fused T.
      fused_T = fuse_dicts(partition_T)
      # Running a fused trial.
      tic = time.perf_counter()
      res, es = self.try_params(partition_ids, n_iterations, fused_T)
      rt = (time.perf_counter() - tic) / len(partition_ids)
      self.info('==> res={}, early_stop={}, runtime={}'.format(res, es, rt))
      # Map the results back to its ids.
      for i, r, e in zip(partition_ids, res, es):
        results[i] = r
        early_stops[i] = e
        runtimes[i] = rt
    return ([results[i] for i in ids], [early_stops[i] for i in ids],
            [runtimes[i] for i in ids])
