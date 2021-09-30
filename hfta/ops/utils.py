import torch
import numpy as np
import re

RE_PARSE_RATIO = re.compile('Mismatched elements: \d+ \/ \d+ \((\d+)\.(\d+)%\)')


def testcase_automator(testcase, configs):
  print('Running testcase: {} ...'.format(testcase.__name__))
  for name, vals in configs.items():
    print('\tTesting along {} ...'.format(name))
    for val in vals:
      print('\t\tTry {}={}'.format(name, val))
      kwargs = {name: val}
      testcase(**kwargs)


def dump_error_msg(e):
  """ Dump out the exception e message """
  print('\t\t-> Failed with error message:')
  print('[Start] ==============================================')
  print(e)
  print('[ End ] ==============================================\n')


def assert_allclose(
    actual,
    desired,
    rtol=1e-07,
    atol=0,
    equal_nan=True,
    err_msg='',
    verbose=True,
    population_threshold=0.0,
):
  try:
    np.testing.assert_allclose(
        actual,
        desired,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        err_msg=err_msg,
        verbose=verbose,
    )
  except AssertionError as e:
    m = RE_PARSE_RATIO.search(str(e))
    if not m:
      raise e
    else:
      if (float('{}.{}'.format(m.group(1), m.group(2))) / 100 >=
          population_threshold):
        raise e
