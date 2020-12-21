from .runner import (SerialRunner, ConcurrentRunner, HFTARunner, MPSRunner,
                     MAX_ITERS_PER_EPOCH)
from .utils import attach_args, rearrange_runner_kwargs, extract_logging_level


def _init_precs(device, device_model):
  if device == 'cuda':
    precs = ['fp32', 'amp']
  elif device == 'xla':
    precs = ['bf16']
  else:
    precs = ['fp32']
  return precs


def _init_modes(device, device_model):
  if device == 'cuda':
    modes = ['serial', 'concurrent', 'mps', 'hfta']
    if device_model == 'a100':
      modes += ['mig']
  elif device == 'xla':
    modes = ['serial', 'hfta']
  else:
    modes = ['serial', 'concurrent', 'hfta']
  return modes


def workflow(
    trial_func=None,
    device='cuda',
    device_model='v100',
    outdir_prefix=None,
    precs=None,
    modes=None,
    enable_dcgm=True,
    epochs=10,
    iters_per_epoch=MAX_ITERS_PER_EPOCH,
    concurrent_runner_kwargs=None,
    mps_runner_kwargs=None,
    hfta_runner_kwargs=None,
):
  assert callable(trial_func)
  assert device in {'cuda', 'xla', 'cpu'}
  assert isinstance(device_model, str)
  device_model = device_model.lower()
  assert outdir_prefix is None or (isinstance(outdir_prefix, str) and
                                   len(outdir_prefix) > 0)
  assert precs is None or isinstance(precs, (list, tuple))
  assert modes is None or isinstance(modes, (list, tuple))
  assert isinstance(enable_dcgm, bool)
  assert isinstance(epochs, int)
  assert isinstance(iters_per_epoch, int)

  def validate_kwargs(kwargs):
    assert kwargs is None or isinstance(kwargs, dict)

  for kwargs in [
      concurrent_runner_kwargs, mps_runner_kwargs, hfta_runner_kwargs
  ]:
    validate_kwargs(kwargs)

  if precs is None or len(precs) == 0:
    precs = _init_precs(device, device_model)

  if modes is None or len(modes) == 0:
    modes = _init_modes(device, device_model)

  runners = []
  if 'serial' in modes:
    runners.append(SerialRunner())
  if 'concurrent' in modes:
    runners.append(ConcurrentRunner(**concurrent_runner_kwargs))
  if 'mps' in modes:
    runners.append(MPSRunner(**mps_runner_kwargs))
  if 'mig' in modes:
    raise ValueError('mig currently not supported!')
  if 'hfta' in modes:
    runners.append(HFTARunner(**hfta_runner_kwargs))

  run_kwargs = {
      'trial_func': trial_func,
      'device': device,
      'device_model': device_model,
      'precs': precs,
      'outdir_prefix': outdir_prefix,
      'enable_dcgm': enable_dcgm,
      'epochs': epochs,
      'iters_per_epoch': iters_per_epoch,
  }

  for runner in runners:
    runner.info('Starting with run_kwargs = {}'.format(run_kwargs))
    succeeded = runner.run(**run_kwargs)
    if not succeeded:
      runner.error('Failed!')
      return succeeded
  return True
