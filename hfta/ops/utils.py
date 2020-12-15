from torch._six import container_abcs
from itertools import repeat


def testcase_automator(testcase, configs):
  print('Running testcase: {} ...'.format(testcase.__name__))
  for name, vals in configs.items():
    print('\tTesting along {} ...'.format(name))
    for val in vals:
      print('\t\tTry {}={}'.format(name, val))
      kwargs = {name: val}
      testcase(**kwargs)


def _reverse_repeat_tuple(t, n):
  r"""Reverse the order of `t` and repeat each element for `n` times.
  This can be used to translate padding arg used by Conv and Pooling modules
  to the ones used by `F.pad`.
  """
  return tuple(x for x in reversed(t) for _ in range(n))


def _ntuple(n):

  def parse(x):
    if isinstance(x, container_abcs.Iterable):
      return x
    return tuple(repeat(x, n))

  return parse


_single = _ntuple(1)
_pair = _ntuple(2)
