import argparse
import logging
import subprocess


def run_command(cmd, input=None):
  """
  Run the given cmd and handle the status
  cmd is a string showing the command to run
  """
  stdout = subprocess.check_output(
      cmd.split(),
      universal_newlines=True,
      input=input,
  )
  return stdout


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


def attach_args(parser=argparse.ArgumentParser()):
  parser.add_argument(
      '--log-level',
      type=str,
      default='INFO',
      help='logging level',
  )
  parser.add_argument(
      '--device',
      type=str,
      default='cuda',
      choices=['cuda', 'xla', 'cpu'],
      help='type of the device used for training',
  )
  parser.add_argument(
      '--device-model',
      type=str,
      default='v100',
      help='device model (e.g., v100, a100, rtx6000, v3) used for training',
  )
  parser.add_argument(
      '--precs',
      type=str,
      default=None,
      choices=['fp32', 'amp', 'bf16'],
      nargs='*',
      help='training precision',
  )
  parser.add_argument(
      '--modes',
      type=str,
      default=None,
      choices=['serial', 'concurrent', 'mps', 'mig', 'hfta'],
      nargs='*',
      help='hardware sharing mode',
  )
  parser.add_argument(
      '--enable-dcgm',
      default=True,
      action='store_true',
      help='start DCGM to monitor hardware performance counters',
  )
  parser.add_argument(
      '--enable-tpu-profiler',
      default=True,
      action='store_true',
      help='use TPU profiler to monitor TPU performance counters',
  )
  parser.add_argument(
      '--disable-dcgm',
      dest='enable-dcgm',
      action='store_false',
      help='do not use DCGM',
  )
  for mode in ['concurrent', 'mps', 'hfta', 'mig']:
    parser.add_argument(
        '--{}-dry-run-repeats'.format(mode),
        type=int,
        default=None,
        help='{}: when finding max B, repeating trials many times to see '
        'if co-running multiple training processes is stable'.format(mode),
    )
    parser.add_argument(
        '--{}-max-num-Bs'.format(mode),
        type=int,
        default=None,
        help='{}: max number of Bs for performance measurements'.format(mode),
    )
    parser.add_argument(
        '--{}-lambd'.format(mode),
        type=float,
        default=None,
        help='{}: the lambd attribute when sampling which Bs to use from '
        'an exponential distribution'.format(mode),
    )
    parser.add_argument(
        '--{}-dry-run-epochs'.format(mode),
        type=int,
        default=None,
        help='{}: when finding max B, the number of epochs'.format(mode),
    )
    parser.add_argument(
        '--{}-dry-run-iters-per-epoch'.format(mode),
        type=int,
        default=None,
        help='{}: when finding max B, the number of iterations per '
        'epoch'.format(mode),
    )

  return parser


def rearrange_runner_kwargs(args):
  for mode in ['concurrent', 'mps', 'hfta', 'mig']:
    runner_kwargs_name = '{}_runner_kwargs'.format(mode)
    setattr(args, runner_kwargs_name, {})
    for kw in [
        'dry_run_repeats',
        'max_num_Bs',
        'lambd',
        'dry_run_epochs',
        'dry_run_iters_per_epoch',
    ]:
      val = getattr(args, '{}_{}'.format(mode, kw))
      if val is None:
        continue
      getattr(args, runner_kwargs_name)[kw] = val


def extract_logging_level(args):
  numeric_level = getattr(logging, args.log_level.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: {}'.format(args.log_level))
  return numeric_level
