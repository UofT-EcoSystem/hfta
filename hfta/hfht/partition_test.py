import random

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from .partition import (build_sets, disassemble_sets,
                        partition_hyperparameter_sets, limit_partition_size)
from .utils import hash_dict, build_capacity_spec


def test_all_hyperparameter_sets_included_once(sets, partitions):
  included = set([])
  for partition in partitions:
    for hpset in partition:
      hpset_id = hpset['id']
      # Test each hyper-parameter set is included only once.
      assert hpset_id not in included
      included.add(hpset_id)
  # Test all hyper-parameter sets are inlcuded.
  assert len(included) == len(sets)


def nonfusibles_with_same_value(partition, nonfusibles):
  for nonfusible in nonfusibles:
    base_value = partition[0]['params'][nonfusible]
    for hpset in partition:
      if hpset['params'][nonfusible] != base_value:
        return False
  return True


def test_each_partition_nonfusibles_with_same_value(partitions, nonfusibles):
  for partition in partitions:
    assert nonfusibles_with_same_value(partition, nonfusibles)


def test_min_num_partitions(partitions, nonfusibles):
  # The number of partitions is minimized iff:
  # Merging partition A and partition B together results in invalid partition C
  # for any pairs of A and B in partitions
  # where "invalid" is defined as the nonfusible hyper-parameters in a partition
  # have more than one value.
  N = len(partitions)
  for i in range(N):
    for j in range(i + 1, N):
      assert nonfusibles_with_same_value(
          partitions[i] + partitions[j],
          nonfusibles,
      ) == False


def test_partition_size_limited(partitions, nonfusibles, capacity_spec):
  for partition in partitions:
    base_key = hash_dict({
        nonfusible: partition[0]['params'][nonfusible]
        for nonfusible in nonfusibles
    })
    for hpset in partition:
      key = hash_dict({
          nonfusible: hpset['params'][nonfusible] for nonfusible in nonfusibles
      })
      assert base_key == key
    limit = capacity_spec.get(base_key, 1)
    assert len(partition) <= limit


def test():
  space = {
      'lr': hp.uniform('lr', 0.0001, 0.01),
      'beta1': hp.uniform('beta1', 0.001, 0.999),
      'beta2': hp.uniform('beta2', 0.001, 0.999),
      'weight_decay': hp.uniform('weight_decay', 0.0, 0.5),
      'gamma': hp.uniform('gamma', 0.1, 0.9),
      'step_size': hp.choice('step_size', (5, 10, 20, 40)),
      'batch_size': hp.choice('batch_size', (8, 16, 32, 64)),
      'feature_transform': hp.choice('feature_transform', (True, False)),
  }
  capacity_spec = build_capacity_spec(
      'capacity_specs/cuda/v100/amp/pointnet_classification.json')
  sets_size = random.randint(1, 100)
  ids = range(0, sets_size)
  T = [sample(space) for _ in range(sets_size)]
  map_ids_T = {i: t for i, t in zip(ids, T)}
  sets = build_sets(ids, T)
  nonfusibles = ['batch_size', 'feature_transform']
  partitions = partition_hyperparameter_sets(sets, nonfusibles)
  test_all_hyperparameter_sets_included_once(sets, partitions)
  test_each_partition_nonfusibles_with_same_value(partitions, nonfusibles)
  test_min_num_partitions(partitions, nonfusibles)

  max_partition_size = max([len(partition) for partition in partitions])
  partitions = limit_partition_size(partitions, nonfusibles, capacity_spec)
  test_partition_size_limited(partitions, nonfusibles, capacity_spec)
  test_all_hyperparameter_sets_included_once(sets, partitions)
  test_each_partition_nonfusibles_with_same_value(partitions, nonfusibles)

  partitions_ids, partitions_T = disassemble_sets(partitions)
  for partition_ids, partition_T in zip(partitions_ids, partitions_T):
    for i, t in zip(partition_ids, partition_T):
      assert hash_dict(map_ids_T[i]) == hash_dict(t)


if __name__ == '__main__':
  for _ in range(20):
    test()
