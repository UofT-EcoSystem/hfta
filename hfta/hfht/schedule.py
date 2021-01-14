import json
import random
import time
import concurrent.futures
import os
import re
from queue import Queue
from .partition import (build_sets, disassemble_sets,
                        partition_hyperparameter_sets_by_capacity)
from .utils import (hash_dict, build_capacity_spec, run_command,
                    resolve_overlap_runtimes)


class Scheduler:

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

  def execute_params_sets(self, ids, n_iterations, T):
    results, early_stops, runtimes = [], [], []
    for i, t in zip(ids, T):
      print('Running id={}, t={}...'.format(i, t))
      tic = time.perf_counter()
      res, es = self.try_params(i, n_iterations, t)
      rt = time.perf_counter() - tic
      print('==> res={}, early_stop={}, runtime={}'.format(res, es, rt))
      results.append(res)
      early_stops.append(es)
      runtimes.append(rt)
    return results, early_stops, runtimes


class ConcurrentScheduler(Scheduler):

  def __init__(self, try_params_callback, num_concurrent):
    self.try_params = try_params_callback
    self.num_concurrent = num_concurrent
    self.sudo = "" if os.geteuid() == 0 else "sudo"

  def _setup(self):
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
    print('Running id={}, t={}...'.format(id, t))

    env_vars = self._one_params_set_setup(id, n_iterations, t)

    tic = time.perf_counter()
    res, es = self.try_params(id, n_iterations, t, env_vars=env_vars)
    toc = time.perf_counter()

    self._one_params_set_teardown(id, n_iterations, t)

    print('==> res={}, early_stop={}, runtime(not normalized)={}'.format(
        res, es, toc - tic))
    return res, es, (tic, toc)

  def execute_params_sets(self, ids, n_iterations, T):
    results, early_stops, runtimes = [], [], []

    self._setup()

    try:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=self.num_concurrent) as executor:

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

  def __init__(self, try_params_callback, num_concurrent):
    super().__init__(try_params_callback, num_concurrent)
    self.orig_env = dict(os.environ)

  # overrides base
  def _setup(self):
    self._export_mps_env_vars()

  # overrides base
  def _teardown(self):
    self._clean_up_mps_env_vars()

  def _export_mps_env_vars(self):
    print("Set up the environment to use MPS ...")
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

  def _clean_up_mps_env_vars(self):
    print("Clean up the environment to use MPS ...")
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
    os.environ.update(self.orig_env)


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

  def __init__(self, try_params_callback, num_concurrent):
    super().__init__(try_params_callback, num_concurrent)

    if num_concurrent > 7 or num_concurrent <= 0:
      raise ValueError("MIG does not support concurrent workload> 7 or <=0")

    # check that mig is enabled
    mgi_query_str = ("nvidia-smi  --query-gpu=mig.mode.current "
                     "--format=csv,noheader")
    mig_query_res = run_command(mgi_query_str)
    if "Enabled" not in mig_query_res:
      raise RuntimeError("Mig is not enabled, please run "
                         "\"{} nvidia-smi  -mig 1\" and reboot the machine "
                         "to enable it".format(self.sudo))

    self.orig_env = dict(os.environ)
    self.mig_GPU_IDs = None
    self.device_queue = Queue()
    self.param_id_dev_map = {}

  def _setup(self):
    # clean up the current instances
    # could be the case that none exists but that is OK
    self._destroy_mig_instances()
    # create new instances based on concurrent width
    self._create_mig_instances()

  def _teardown(self):
    print("Clean up the environment for MIG ...")

    self._destroy_mig_instances()
    # restore the environment
    os.environ.clear()
    os.environ.update(self.orig_env)

  def _one_params_set_setup(self, id, n_iterations, t):
    dev = self._acquire_mig_instance()
    env_vars = {"CUDA_VISIBLE_DEVICES": dev}
    self.param_id_dev_map[id] = dev
    return env_vars

  def _one_params_set_teardown(self, id, n_iterations, t):
    dev = self.param_id_dev_map[id]
    self._release_mig_instance(dev)
    del self.param_id_dev_map[id]

  def _create_mig_instances(self):
    print("Set up the environment to use MIG ...")

    mig_profile_str = MIGScheduler.MIG_PROFILE_CONFIGS[self.num_concurrent]

    # create virtual MIG GPU instances
    cmds_to_run = [
        "{} nvidia-smi mig -cgi {}".format(self.sudo, mig_profile_str),
        "{} nvidia-smi mig -cci".format(self.sudo),
    ]
    for cmd in cmds_to_run:
      run_command(cmd)

    self.mig_GPU_IDs = self._query_mig_devices()
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

  def _query_mig_devices(self):
    # query the devices
    get_gpu_dev_str = "nvidia-smi -L"
    GPU_IDs_raw = run_command(get_gpu_dev_str).split("\n")
    mig_GPU_IDs_raw = [s for s in GPU_IDs_raw if "MIG" in s]
    assert len(mig_GPU_IDs_raw) == self.num_concurrent

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


class HFTAScheduler(Scheduler):

  def __init__(self, try_params_callback, nonfusibles, capacity_spec_path):
    self.try_params = try_params_callback
    self.nonfusibles = nonfusibles
    self.capacity_spec = build_capacity_spec(capacity_spec_path)

  def execute_params_sets(self, ids, n_iterations, T):
    sets = build_sets(ids, T)
    partitions = partition_hyperparameter_sets_by_capacity(
        sets,
        self.nonfusibles,
        self.capacity_spec,
    )
    partitions_ids, partitions_T = disassemble_sets(partitions)
    results, early_stops, runtimes = {}, {}, {}
    for partition_ids, partition_T in zip(partitions_ids, partitions_T):
      print('Running partition_ids={}, partition_T={}'.format(
          partition_ids, partition_T))
      # Generate fused T.
      fused_T = {}
      for t in partition_T:
        for k, v in t.items():
          if k not in fused_T:
            fused_T[k] = []
          fused_T[k].append(v)
      # Running a fused trial.
      tic = time.perf_counter()
      res, es = self.try_params(partition_ids, n_iterations, fused_T)
      rt = (time.perf_counter() - tic) / len(partition_ids)
      print('==> res={}, early_stop={}, runtime={}'.format(res, es, rt))
      # Map the results back to its ids.
      for i, r, e in zip(partition_ids, res, es):
        results[i] = r
        early_stops[i] = e
        runtimes[i] = rt
    return ([results[i] for i in ids], [early_stops[i] for i in ids],
            [runtimes[i] for i in ids])
