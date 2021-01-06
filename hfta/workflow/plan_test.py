import random

from hfta.workflow.plan import find_max_B, expovariate_plan


def testcase_find_max_B(expected_max_B, dry_run_repeats, B_limit, unstable_prob):

  def try_B(B):
    if random.random() > unstable_prob:
      return B <= expected_max_B
    else:
      return B <= random.randint(expected_max_B + 1, 2 * expected_max_B)

  actual_max_B = find_max_B(try_B, dry_run_repeats=dry_run_repeats, B_limit=B_limit)
  assert actual_max_B == min(expected_max_B, B_limit)


def test_find_max_B():
  print('test_find_max_B:')
  print('==================================')
  print('functional testing ...')
  for _ in range(20):
    expected_max_B = random.randint(1, 500)
    B_limit = random.randint(1, 500)
    print('  testcase_find_max_B({}, {}, 1, -1.0)...'.format(expected_max_B, B_limit))
    testcase_find_max_B(expected_max_B, 1, B_limit, -1.0)
  print('  testcase_find_max_B(1, 1, 1, -1.0)...')
  testcase_find_max_B(1, 1, 1, -1.0)

  print('  testcase_find_max_B(0, 1, 1, -1.0)...')
  try:
    testcase_find_max_B(0, 1, 1, -1.0)
    assert False
  except RuntimeError as e:
    assert str(e) == "Cannot fit a single model!"

  print('  testcase_find_max_B(1, 1, 0, -1.0)...')
  try:
    testcase_find_max_B(1, 1, 0, -1.0)
    assert False
  except RuntimeError as e:
    assert str(e) == "B_limit should be greater than 0!"

  def error_rate(unstable_prob, dry_run_repeats, B_limit=None):
    total = 1000
    error = 0
    for _ in range(total):
      try:
        testcase_find_max_B(
            random.randint(1, 500),
            random.randint(1, 500) if B_limit == None else B_limit,
            dry_run_repeats,
            unstable_prob,
        )
      except AssertionError:
        error += 1
    return error / total

  for unstable_prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
    print('stabiility testing for unstable_prob = {}...'.format(unstable_prob))
    for dry_run_repeats in [1, 3, 5, 10, 20]:
      for B_limit in [None, 100, 300, 500]:
        print('  B_limit={}, dry_run_repeats = {}, error_rate = {}'.format(
            B_limit,
          dry_run_repeats,
            error_rate(unstable_prob, dry_run_repeats, B_limit),
      ))


def testcase_expovariate_plan(max_B, max_num_Bs):
  print('  testcase_expovariate_plan({}, {})...'.format(max_B, max_num_Bs))
  Bs = expovariate_plan(max_B, max_num_Bs)
  assert len(Bs) == min(max_num_Bs, max_B)
  assert all([B <= max_B for B in Bs])
  assert max_B in Bs


def test_expovariate_plan():
  print('test_expovariate_plan:')
  print('==================================')
  print('functional testing ...')
  for _ in range(10):
    testcase_expovariate_plan(random.randint(1, 100), random.randint(1, 100))
  testcase_expovariate_plan(1, 1)
  testcase_expovariate_plan(random.randint(1, 100), 1)
  testcase_expovariate_plan(1, random.randint(1, 100))
  print('distribution testing ...')
  for max_B in [10, 30, 50, 100]:
    print('  max_B = {}'.format(max_B))
    for max_num_Bs in [5, 10, 20]:
      print('    max_num_Bs = {}'.format(max_num_Bs))
      for lambd in [1.0, 2.0, 4.0, 8.0]:
        print('      lambd = {}: {}'.format(
            lambd,
            expovariate_plan(max_B, max_num_Bs, lambd),
        ))


if __name__ == '__main__':
  test_find_max_B()
  test_expovariate_plan()
