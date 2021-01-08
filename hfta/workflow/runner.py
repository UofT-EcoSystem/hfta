import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .dcgm_monitor import dcgm_monitor_start, dcgm_monitor_stop, DcgmMonitor
from .plan import find_max_B, expovariate_plan
from .utils import run_command

MAX_ITERS_PER_EPOCH = 1000000000


class Runner:

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

  def _plan_Bs(self, trial_func=None, device='cuda', prec='fp32'):
    raise NotImplementedError('Runner is an abstract/interface class!')

  def _run_B(
      self,
      B,
      trial_func=None,
      device='cuda',
      prec='fp32',
      epochs=10,
      iters_per_epoch=MAX_ITERS_PER_EPOCH,
      outdir_prefix=None,
  ):
    raise NotImplementedError('Runner is an abstract/interface class!')

  def _outdir_mode_B(self, outdir_prefix, B):
    raise NotImplementedError('Runner is an abstract/interface class!')

  def _sweep_Bs(
      self,
      Bs,
      trial_func=None,
      device='cuda',
      device_model='v100',
      prec='fp32',
      outdir_prefix=None,
      enable_dcgm=None,
      epochs=10,
      iters_per_epoch=MAX_ITERS_PER_EPOCH,
  ):
    self.info('Sweeping Bs: {} ...'.format(Bs))
    for B in Bs:
      self.info('Measuring B: {} ...'.format(B))
      outdir = self._outdir_mode_B(outdir_prefix, B)

      if enable_dcgm and device == 'cuda':
        monitor = DcgmMonitor(device_model)
        monitor_thread = dcgm_monitor_start(monitor, outdir)

      try:
        succeeded = self._run_B(
            B,
            trial_func=trial_func,
            device=device,
            prec=prec,
            epochs=epochs,
            iters_per_epoch=iters_per_epoch,
            outdir_prefix=outdir,
        )
      finally:
        if enable_dcgm and device == 'cuda':
          dcgm_monitor_stop(monitor, monitor_thread)
      if not succeeded:
        self.error('B = {} failed!'.format(B))
        return succeeded

    return True

  def run(
      self,
      trial_func=None,
      device='cuda',
      device_model='v100',
      precs=['fp32', 'amp'],
      outdir_prefix=None,
      enable_dcgm=True,
      epochs=10,
      iters_per_epoch=MAX_ITERS_PER_EPOCH,
  ):
    self.info('Sweeping precs: {} ...'.format(precs))
    for prec in precs:
      self.info('Measuring prec: {} ...'.format(prec))
      orig_envs = dict(os.environ)
      if device == 'cuda' and device_model == 'a100' and prec == 'fp32':
        os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

      succeeded = self._sweep_Bs(
          self._plan_Bs(trial_func=trial_func, device=device, prec=prec),
          trial_func=trial_func,
          device=device,
          device_model=device_model,
          prec=prec,
          outdir_prefix=os.path.join(
              outdir_prefix,
              device,
              device_model,
              prec,
          ),
          enable_dcgm=enable_dcgm,
          epochs=epochs,
          iters_per_epoch=iters_per_epoch,
      )

      if device == 'cuda' and device_model == 'a100' and prec == 'fp32':
        os.environ.clear()
        os.environ.update(orig_envs)

      if not succeeded:
        self.error('prec = {} failed!'.format(prec))
        return succeeded

    return True


class HardwareSharingRunner(Runner):
  B_LIMIT = None

  def __init__(
      self,
      dry_run_repeats=None,
      max_num_Bs=None,
      lambd=None,
      dry_run_epochs=None,
      dry_run_iters_per_epoch=None,
  ):
    self._dry_run_repeats = dry_run_repeats
    self._max_num_Bs = max_num_Bs
    self._lambd = lambd
    self._dry_run_epochs = dry_run_epochs
    self._dry_run_iters_per_epoch = dry_run_iters_per_epoch
    self.info('Created with dry_run_repeats = {}, max_num_Bs = {}, lambd = {}, '
              'dry_run_epochs = {}, dry_run_iters_per_epoch = {}'.format(
                  dry_run_repeats,
                  max_num_Bs,
                  lambd,
                  dry_run_epochs,
                  dry_run_iters_per_epoch,
              ))

  def _outdir_mode_B(self, outdir_prefix, B):
    return os.path.join(outdir_prefix, self.mode, 'B{}'.format(B))

  def _plan_Bs(self, trial_func=None, device='cuda', prec='fp32'):

    def try_B(B):
      self.info('Trying B: {} ...'.format(B))
      succeeded = self._run_B(
          B,
          trial_func=trial_func,
          device=device,
          prec=prec,
          epochs=self._dry_run_epochs,
          iters_per_epoch=self._dry_run_iters_per_epoch,
          outdir_prefix=None,
      )
      if succeeded:
        self.info('--> OK')
      else:
        self.info('--> FAIL')
      return succeeded

    self.info('Searching for max B ...')
    max_B = find_max_B(try_B,
                       dry_run_repeats=self._dry_run_repeats,
                       B_limit=self.B_LIMIT)
    self.info('Found max B: {} !'.format(max_B))
    Bs = expovariate_plan(max_B, self._max_num_Bs, lambd=self._lambd)
    self.info('Planned Bs to measure: {} !'.format(Bs))
    return Bs


def mk_outdir_and_run_trial(
    trial_func=None,
    B=0,
    device='cuda',
    prec='fp32',
    epochs=10,
    iters_per_epoch=MAX_ITERS_PER_EPOCH,
    env_map=None,
    outdir=None,
):
  if outdir is not None:
    Path(outdir).mkdir(parents=True, exist_ok=True)
  return trial_func(
      B=B,
      device=device,
      prec=prec,
      epochs=epochs,
      iters_per_epoch=iters_per_epoch,
      outdir=outdir,
      env_map=env_map,
  )


class SerialRunner(Runner):

  @property
  def mode(self):
    return 'serial'

  def _plan_Bs(self, trial_func=None, device='cuda', prec='fp32'):
    return [1]

  def _outdir_mode_B(self, outdir_prefix, B):
    return os.path.join(outdir_prefix, 'serial')

  def _run_B(
      self,
      B,
      trial_func=None,
      device='cuda',
      prec='fp32',
      epochs=10,
      iters_per_epoch=MAX_ITERS_PER_EPOCH,
      outdir_prefix=None,
  ):
    assert B == 1
    with ThreadPoolExecutor(max_workers=1) as executor:
      t = executor.submit(
          mk_outdir_and_run_trial,
          trial_func=trial_func,
          B=0,
          device=device,
          prec=prec,
          epochs=epochs,
          iters_per_epoch=iters_per_epoch,
          outdir=outdir_prefix,
      )
      return t.result()


class ConcurrentRunner(HardwareSharingRunner):

  def __init__(
      self,
      dry_run_repeats=10,
      max_num_Bs=5,
      lambd=4.0,
      dry_run_epochs=2,
      dry_run_iters_per_epoch=10,
  ):
    super(ConcurrentRunner, self).__init__(
        dry_run_repeats=dry_run_repeats,
        max_num_Bs=max_num_Bs,
        lambd=lambd,
        dry_run_epochs=dry_run_epochs,
        dry_run_iters_per_epoch=dry_run_iters_per_epoch,
    )

  @property
  def mode(self):
    return 'concurrent'

  def _run_B(
      self,
      B,
      trial_func=None,
      device='cuda',
      prec='fp32',
      epochs=10,
      iters_per_epoch=MAX_ITERS_PER_EPOCH,
      outdir_prefix=None,
  ):
    assert device == 'cuda'
    with ThreadPoolExecutor(max_workers=B) as executor:
      if outdir_prefix is None:
        outdirs = [None for _ in range(B)]
      else:
        outdirs = [
            os.path.join(outdir_prefix, 'idx{}'.format(b)) for b in range(B)
        ]
      ts = [
          executor.submit(
              mk_outdir_and_run_trial,
              trial_func=trial_func,
              B=0,
              device=device,
              prec=prec,
              epochs=epochs,
              iters_per_epoch=iters_per_epoch,
              outdir=outdirs[b],
          ) for b in range(B)
      ]
      return all([t.result() for t in ts])


class MPSRunner(ConcurrentRunner):

  @property
  def mode(self):
    return 'mps'

  def _run_B(
      self,
      B,
      trial_func=None,
      device='cuda',
      prec='fp32',
      epochs=10,
      iters_per_epoch=MAX_ITERS_PER_EPOCH,
      outdir_prefix=None,
  ):
    sudo = '' if os.geteuid() == 0 else 'sudo'
    run_command('{} nvidia-smi -i 0 -c EXCLUSIVE_PROCESS'.format(sudo))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-mps'
    os.environ['CUDA_MPS_LOG_DIRECTORY'] = '/tmp/nvidia-log'
    run_command('nvidia-cuda-mps-control -d')

    try:
      succeeded = super(MPSRunner, self)._run_B(
          B,
          trial_func=trial_func,
          device=device,
          prec=prec,
          epochs=epochs,
          iters_per_epoch=iters_per_epoch,
          outdir_prefix=outdir_prefix,
      )
    finally:
      run_command('{} nvidia-cuda-mps-control'.format(sudo), 'quit')
      run_command('{} nvidia-smi -i 0 -c 0'.format(sudo))
      del os.environ['CUDA_VISIBLE_DEVICES']
      del os.environ['CUDA_MPS_PIPE_DIRECTORY']
      del os.environ['CUDA_MPS_LOG_DIRECTORY']

    return succeeded


class MIGRunner(ConcurrentRunner):
  MIG_PROFILE_CONFIGS = ('0', '9,9', '14,14,14', '14,14,14,19',
                         '14,14,19,19,19', '14,19,19,19,19,19',
                         '19,19,19,19,19,19,19')
  SUDO = '' if os.geteuid() == 0 else 'sudo'
  B_LIMIT = 7

  def __init__(
      self,
      dry_run_repeats=1,
      max_num_Bs=5,
      lambd=4.0,
      dry_run_epochs=2,
      dry_run_iters_per_epoch=10,
  ):
    super(ConcurrentRunner, self).__init__(
        dry_run_repeats=dry_run_repeats,
        max_num_Bs=max_num_Bs,
        lambd=lambd,
        dry_run_epochs=dry_run_epochs,
        dry_run_iters_per_epoch=dry_run_iters_per_epoch,
    )

  @property
  def mode(self):
    return 'mig'

  def _create_mig_instances(self, B):
    self._distroy_mig_instances()
    B_idx = B - 1
    mig_dev_ids = []
    try:
      prefix = "{} nvidia-smi ".format(self.SUDO)
      run_command("{} mig -cgi {}".format(prefix,
                                          self.MIG_PROFILE_CONFIGS[B_idx]))
      run_command("{} mig -cci".format(prefix))
      cmd = "{} -L".format(prefix)
      cmd_out = run_command(cmd)
      cmd_outs = cmd_out.split("\n")
      for item in cmd_outs:
        if ("MIG" in item):
          mig_dev_ids.append(item.split("UUID: ")[-1][:-1])
    except subprocess.CalledProcessError as e:
      logging.error(e)

    return mig_dev_ids

  def _distroy_mig_instances(self):
    try:
      run_command("{} nvidia-smi mig -dci -i 0".format(self.SUDO))
      run_command("{} nvidia-smi mig -dgi -i 0".format(self.SUDO))
    except subprocess.CalledProcessError as e:
      # 'returncode==6' indicates it can't find any MIG instance to destory.
      # That means all MIG instances have been destoryed.
      if e.returncode != 6:
        logging.error(e)
        return e.returncode
    return 0

  def _run_B(
      self,
      B,
      trial_func=None,
      device='cuda',
      prec='fp32',
      epochs=10,
      iters_per_epoch=MAX_ITERS_PER_EPOCH,
      outdir_prefix=None,
  ):

    assert device == 'cuda'
    mig_instances = self._create_mig_instances(B)

    if len(mig_instances) == 0:
      raise RuntimeError(""" CAN NOT FIND ANY MIG DEVICE!
            Please enable MIG on A100 with follow command:
              <sudo nvidia-smi -mig 1> 
            And then reboot the system.
            NOTE: After enable MIG, only MIGRunner can be used.
      """)


    assert len(mig_instances) == B

    with ThreadPoolExecutor(max_workers=B) as executor:
      if outdir_prefix is None:
        outdirs = [None for _ in range(B)]
      else:
        outdirs = [
            os.path.join(outdir_prefix, 'idx{}'.format(b)) for b in range(B)
        ]
      ts = [
          executor.submit(mk_outdir_and_run_trial,
                          trial_func=trial_func,
                          B=0,
                          device=device,
                          prec=prec,
                          epochs=epochs,
                          iters_per_epoch=iters_per_epoch,
                          outdir=outdirs[b],
                          env_map={
                              "CUDA_VISIBLE_DEVICES": mig_instances[b],
                              **os.environ
                          }) for b in range(B)
      ]
      results = all([t.result() for t in ts])
      self._distroy_mig_instances()
      time.sleep(1)
      return results


class HFTARunner(HardwareSharingRunner):

  def __init__(
      self,
      dry_run_repeats=1,
      max_num_Bs=10,
      lambd=4.0,
      dry_run_epochs=2,
      dry_run_iters_per_epoch=3,
  ):
    super(HFTARunner, self).__init__(
        dry_run_repeats=dry_run_repeats,
        max_num_Bs=max_num_Bs,
        lambd=lambd,
        dry_run_epochs=dry_run_epochs,
        dry_run_iters_per_epoch=dry_run_iters_per_epoch,
    )

  @property
  def mode(self):
    return 'hfta'

  def _run_B(
      self,
      B,
      trial_func=None,
      device='cuda',
      prec='fp32',
      epochs=10,
      iters_per_epoch=MAX_ITERS_PER_EPOCH,
      outdir_prefix=None,
  ):
    with ThreadPoolExecutor(max_workers=1) as executor:
      t = executor.submit(
          mk_outdir_and_run_trial,
          trial_func=trial_func,
          B=B,
          device=device,
          prec=prec,
          epochs=epochs,
          iters_per_epoch=iters_per_epoch,
          outdir=outdir_prefix,
      )
      return t.result()
