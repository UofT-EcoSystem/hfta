import logging
import os
import threading
import time
from multiprocessing import Process

try:
  from tensorflow.python.profiler import profiler_client
  from tensorflow.errors import UnavailableError
except ImportError:
  pass


class TpuMonitor:
  """
  TPU performance monitor & result parser
  """

  def logging_format(self, msg):
    return '{}: {}'.format("TpuMonitor", msg)

  def debug(self, msg):
    logging.debug(self.logging_format(msg))

  def info(self, msg):
    logging.info(self.logging_format(msg))

  def warning(self, msg):
    logging.warning(self.logging_format(msg))

  def error(self, msg):
    logging.error(self.logging_format(msg))

  def __init__(self, wait_time, duration, outdir):
    self.info(
        "Start TPU Monitor and it will monitor for {} seconds after waiting for {} seconds."
        .format(duration / 1000, wait_time))
    self.args = self.get_profiler_args(duration, outdir)
    self.wait_time = wait_time

  def get_profiler_args(self, duration, outdir):
    self.debug(
        "Initialize TPU profiler arguments with outdir: {}".format(outdir))
    dir_list = outdir.split("/")
    idx = 0
    for i in range(len(dir_list)):
      if dir_list[i] == "benchmarks":
        idx = i + 1
        break

    logdir = "/".join(dir_list[idx:])
    ret = {
        "service_addr":
            "{}:{}".format(os.environ.get("TPU_IP_ADDRESS"), "8466"),
        "logdir":
            "{}/{}".format(os.environ.get("STORAGE_BUCKET"), logdir),
        "duration_ms":
            duration,
        "worker_list":
            '',
        "num_tracing_attempts":
            10,
        "options":
            None
    }
    logging.debug(ret)
    return ret

  def start_monitoring(self):
    success = False
    sleep_time = 2

    # Sleep for wait_time seconds to avoid the training warmup
    time.sleep(self.wait_time)

    while not success:
      try:
        profiler_client.trace(**self.args)
      except UnavailableError as e:
        self.warning(
            "Failed to capture TPU profile, retry in {} seconds".format(
                sleep_time))
        time.sleep(sleep_time)
      else:
        success = True
        self.info("Successfully captured TPU profile")


def tpu_monitor_thread(monitor):

  # Check tensorflow installation
  try:
    from tensorflow.python.profiler import profiler_client
    from tensorflow.errors import UnavailableError
  except ImportError:
    logging.error(
        "Failed to start TPU monitor thread because tensorflow packages cannot be imported. Please install tensorflow first."
    )
    logging.info("Continue the TPU experiment without running TPU profiler.")
    return

  # Check necessary env vars
  for env_var in ["TPU_NAME", "TPU_IP_ADDRESS", "STORAGE_BUCKET"]:
    if os.environ.get(env_var) is None:
      logging.error(
          "Failed to start TPU monitor thread because {} was not defined.".
          format(env_var))
      logging.info("Continue the TPU experiment without running TPU profiler.")
      return
    else:
      logging.debug("{} is {}".format(env_var, os.environ.get(env_var)))

  monitor.start_monitoring()


def tpu_monitor_start(monitor):
  logging.debug("Start TPU monitoring thread")
  t = threading.Thread(
      target=tpu_monitor_thread,
      name='TPU Monitor Thread',
      args=(monitor,),
  )
  t.start()
  return t


def tpu_monitor_stop(monitor, thread):
  logging.debug("Stop TPU monitoring thread")
  thread.join()


# gsutil cp -r ${STORAGE_BUCKET}/cls/**.overview_page.json ./

# Sample output from profiler_client.monitor
# for query in range(0, 100):
#   print(profiler_client.monitor(self.args['service_addr'], self.args['duration_ms'], self.args['level']))
# Timestamp: 13:00:11
# TPU type: TPU v3
# Number of TPU cores: 1 (Replica count = 1, num cores per replica = 1)
# Per-core batch size: 32
# TPU idle time (lower is better): 17.9%
# Utilization of TPU Matrix Units (higher is better): 0.058%
# Step time: 70.6ms (avg), 70.6ms (min), 70.6ms (max)
# Infeed percentage: 0.000% (avg), 0.000% (min), 0.000% (max)


def tpu_profile_parser_main():
  pass
