import logging
import os
import threading
from multiprocessing import Process

try:
  from tensorflow.python.profiler import profiler_client
except ImportError:
  logging.warning("Cannot import TPU profiler_client from tensorflow.")
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

  def __init__(self, name, duration):
    """
    name: the name and log directory of the test run
    duration: how long do you want to capture the profile (ms)
    """
    self.info("Start TPU Monitor ({})".format(name))
    self.args = self.get_profiler_args(name, duration)
    self.proc = Process(target=profiler_client.trace, kwargs=self.args)

  def get_profiler_args(self, name, duration):

    ret = {
        'service_addr': os.environ.get("TPU_IP_ADDRESS") + ":8466",
        'logdir': os.environ.get("STORAGE_BUCKET") + "/" + name,
        'duration_ms': duration,
        'worker_list': '',
        'num_tracing_attempts': 10,
        'options': None
    }

    return ret

  def start_monitoring(self):
    self.proc.start()

  def stop_monitoring(self):
    self.proc.join()

    # gsutil cp -r ${STORAGE_BUCKET}/cls/**.overview_page.json ./


def tpu_monitor_thread(monitor, outdir):
  # TODO: try to start the client and retry when it failed
  # Check necessary env vars
  for env_var in ["TPU_NAME", "TPU_IP_ADDRESS", "STORAGE_BUCKET"]:
    assert os.environ.get(
        env_var
    ) is not None, "Failed to start tpu monitor thread because {} must be defined.".format(
        env_var)
    logging.info("{} is {}".format(env_var, os.environ.get(env_var)))


def tpu_monitor_start(monitor, outdir):
  # TODO: need to export outdir to env var
  logging.debug("Start tpu monitor thread with outdir: {}".format(outdir))
  t = threading.Thread(
      target=tpu_monitor_thread,
      name='TPU Monitor Thread',
      args=(monitor, outdir),
  )
  t.start()
  return t


def tpu_monitor_stop(monitor, thread):
  logging.debug("Stop tpu monitor thread")
  thread.join()
