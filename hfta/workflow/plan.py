import random


def find_max_B(try_B, dry_run_repeats=5, B_limit=1000000000):

  def fit(B):
    for _ in range(dry_run_repeats):
      if not try_B(B):
        return False
    return True

  prev_B = None
  curr_B = 1
  while fit(curr_B):
    prev_B = curr_B
    curr_B = min(curr_B * 2, B_limit)
    if (prev_B == curr_B):
      break
  if curr_B == 1:
    raise RuntimeError("Cannot fit a single model!")
  # Now that we know max_B is within [prev_B, curr_B], use binary search to find
  # it.
  left, right = prev_B, curr_B
  while left + 1 < right:
    mid = (left + right) // 2
    if fit(mid):
      left = mid
    else:
      right = mid
  return left


def expovariate_plan(max_B, max_num_Bs, lambd=1.0):
  assert isinstance(max_B, int) and max_B > 0
  assert isinstance(max_num_Bs, int) and max_num_Bs > 0
  max_num_Bs = min(max_num_Bs, max_B)
  Bs = set([max_B])
  if max_num_Bs > 1 and 1 not in Bs:
    Bs.add(1)

  while len(Bs) < max_num_Bs:
    B = int(round(random.expovariate(lambd) * max_B))
    if B > max_B or B < 1:
      continue
    if B in Bs:
      continue
    Bs.add(B)
  return list(sorted(Bs))
