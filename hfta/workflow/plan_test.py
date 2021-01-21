import random

from hfta.workflow.plan import find_max_B, expovariate_plan


def testcase_find_max_B(max_B, dry_run_repeats, B_limit, unstable_prob):

  def try_B(B):
    if B_limit is not None:
      assert B <= B_limit
    if random.random() > unstable_prob:
      return B <= max_B
    else:
      return B <= random.randint(max_B + 1, 2 * max_B)

  actual_max_B = find_max_B(try_B,
                            dry_run_repeats=dry_run_repeats,
                            B_limit=B_limit)
  expected_max_B = min(max_B, B_limit) if B_limit is not None else max_B
  assert actual_max_B == expected_max_B


def test_find_max_B():
  print('test_find_max_B:')
  print('==================================')
  print('functional testing ...')
  B_limits = [random.randint(1, 500) for _ in range(10)] + ([None] * 10)
  for B_limit in B_limits:
    max_B = random.randint(1, 500)
    print('  testcase_find_max_B({}, {}, 1, -1.0)...'.format(max_B, B_limit))
    testcase_find_max_B(max_B, 1, B_limit, -1.0)

  print('  testcase_find_max_B(1, 1, None, -1.0)...')
  testcase_find_max_B(1, 1, None, -1.0)
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

  def error_rate(unstable_prob, dry_run_repeats):
    total = 1000
    error = 0
    for _ in range(total):
      try:
        testcase_find_max_B(
            random.randint(1, 500),
            dry_run_repeats,
            None,
            unstable_prob,
        )
      except AssertionError:
        error += 1
    return error / total

  for unstable_prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
    print('stabiility testing for unstable_prob = {}...'.format(unstable_prob))
    for dry_run_repeats in [1, 3, 5, 10, 20]:
      print('  dry_run_repeats = {}, error_rate = {}'.format(
          dry_run_repeats,
          error_rate(unstable_prob, dry_run_repeats),
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
