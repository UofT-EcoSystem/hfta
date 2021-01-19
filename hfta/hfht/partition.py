from .utils import hash_dict


def build_sets(ids, T):
  return [{'id': i, 'params': t} for i, t in zip(ids, T)]


def disassemble_sets(partitions):
  q = lambda k: [[hpset[k] for hpset in partition] for partition in partitions]
  return q('id'), q('params')


def partition_hyperparameter_sets(sets, nonfusibles):
  # Assume all hyper-parameter sets in a single partition at the beginning.
  partitions = [sets]
  for nonfusible in nonfusibles:
    # Partition based on the values of a certain nonfusible hyper-parameter.
    pidx = 0
    while pidx < len(partitions):
      partition = partitions[pidx]
      base_value = partition[0]['params'][nonfusible]
      refined_partition = []
      new_partition = []
      for hpset in partition:
        if hpset['params'][nonfusible] == base_value:
          refined_partition.append(hpset)
        else:
          new_partition.append(hpset)
      partitions[pidx] = refined_partition
      if len(new_partition) > 0:
        partitions.append(new_partition)
      pidx += 1
  return partitions


def limit_partition_size(partitions, nonfusibles, capacity_spec):
  pidx = 0
  while pidx < len(partitions):
    partition = partitions[pidx]
    partition_size = len(partition)
    limit_key = hash_dict({
        nonfusible: partition[0]['params'][nonfusible]
        for nonfusible in nonfusibles
    })
    # If we don't know the partition limit, assume it's 1.
    limit = capacity_spec.get(limit_key, 1)
    if partition_size <= limit:
      pidx += 1
      continue
    refined_partition = partition[:limit]
    new_partition = partition[limit:]
    partitions[pidx] = refined_partition
    partitions.append(new_partition)
  return partitions


def partition_hyperparameter_sets_by_capacity(sets, nonfusibles, capacity_spec):
  partitions = partition_hyperparameter_sets(sets, nonfusibles)
  return limit_partition_size(partitions, nonfusibles, capacity_spec)
