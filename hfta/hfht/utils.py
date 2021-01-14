import json
import subprocess


def hash_dict(d):
  return ','.join('{}:{}'.format(k, d[k]) for k in sorted(d.keys()))


# Handle floats which should be integers
# Works with flat params
def handle_integers(params):
  new_params = {}
  for k, v in params.items():
    if type(v) == float and int(v) == v:
      new_params[k] = int(v)
    else:
      new_params[k] = v

  return new_params


def build_capacity_spec(capacity_spec_path):
  # Map hash_dict(spec['nonfusibles']) to max_num_models.
  return {
      hash_dict(spec['nonfusibles']): spec['max_num_models']
      for spec in json.load(open(capacity_spec_path, 'r'))
  }


def run_command(cmd, input=None, ignore_error=False):
  """
  Run the given cmd and handle the status
  cmd is a string showing the command to run
  """
  print("Running command: {}".format(cmd))
  try:
    output = subprocess.check_output(
            cmd.split(),
            universal_newlines=True,
            input=input,
        )
    print(output)
    return output
  except subprocess.CalledProcessError as e:
    print("Failed to run {}. The error output is:\n{}".format(cmd, e.output))
    if not ignore_error:
      raise


def resolve_overlap_runtimes(runtime_raw):
  """
    Find the individual runtime of a list of concurrently running processes
    based on the intervals the that the processes are co-running.
    i.e.
    (i)   A --------------------------------- B
    (ii)                      C-------------------------------------D
    (iii)            E-----------------F
    (iv)                                   G ------------------------------H
    has 7 overlapping segments:

    A -------E------C---------F-----G---B--------------D------H
        (1)    (2)      (3)     (4)  (5)       (6)        (7)

    Then:
    (i)'s runtime = (1) + (2) / 2 + (3) / 3 + (4) / 2 + (5) / 3
    (ii)'s runtime = (3) / 3 + (4) / 2 + (5) / 3 + (6) / 2
    (iii)'s runtime = (2) / 2 + (3) / 3
    (iv)'s runtime = (5) / 3 + (6) / 2 + (7)

    runtime_raw: list of 2-item tuples of start/end of a process
    """
  # some corner cases
  if len(runtime_raw) == 0:
    return []
  if len(runtime_raw) == 1:
    return [runtime_raw[0][1] - runtime_raw[0][0]]

  # organize the times based on the time it happened
  flatten_runtimes = []
  for idx, (s, e) in enumerate(runtime_raw):
    flatten_runtimes += [(s, idx * 2), (e, idx * 2 + 1)]
  flatten_runtimes.sort(key=lambda x: x[0])

  # re-simulate the process running sequence and keep
  # track of how many processes are run between events
  proc_runtimes = [0 for _ in range(len(runtime_raw))]
  prev_event = flatten_runtimes[0][0]
  prev_idx = flatten_runtimes[0][1]

  # first event must be the start of a process, hence even
  assert prev_idx % 2 == 0

  cur_procs = set([prev_idx // 2])

  for time, idx in flatten_runtimes[1:]:
    # update the current running processes based on the current event
    num_concurrent = float(len(cur_procs))
    for p in cur_procs:
      proc_runtimes[p] += (time - prev_event) / num_concurrent

    this_proc = idx // 2
    is_start = (idx % 2) == 0
    if is_start:
      cur_procs.add(this_proc)
    else:
      cur_procs.remove(this_proc)
    prev_event = time
  assert len(cur_procs) == 0
  return proc_runtimes


def generate_fusible_param_strings(params, name):
  if isinstance(params[name], (list, tuple)):
    return [str(v) for v in params[name]]
  else:
    return [str(params[name])]


def generate_fusible_param_flags(params, names):
  res = []
  for name in names:
    res.extend(['--{}'.format(name)] +
               generate_fusible_param_strings(params, name))
  return res


def generate_nonfusible_param(params, name):
  if isinstance(params[name], (list, tuple)):
    base_value = params[name][0]
    for v in params[name]:
      assert v == base_value
    return base_value
  else:
    return params[name]


def to_csv_dicts(space, history, trajectory):
  history_csv = {
      'id': [],
      'acc': [],
      'early_stop': [],
      'runtime': [],
      'epochs': [],
  }
  history_csv.update({k: [] for k in space.keys()})
  for i, trial in history.items():
    for key in ['id', 'early_stop', 'runtime']:
      history_csv[key].append(trial[key])
    history_csv['acc'].append(trial['result']['acc'])
    history_csv['epochs'].append(int(round(trial['iterations'])))
    for key in space.keys():
      history_csv[key].append(trial['params'][key])

  trajectory_csv = {
      'timestamp': [t['timestamp'] for t in trajectory],
      'acc': [t['acc'] for t in trajectory],
  }
  return history_csv, trajectory_csv
