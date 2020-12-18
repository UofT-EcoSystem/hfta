import pandas as pd
import psutil
import subprocess
import time

DCGM_FIELDS = {
    'DCGM_FI_PROF_GR_ENGINE_ACTIVE': 1001,
    'DCGM_FI_PROF_DRAM_ACTIVE': 1005,
    'DCGM_FI_DEV_GPU_UTIL': 203,
    'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE': 1004,
    'DCGM_FI_PROF_PIPE_FP16_ACTIVE': 1008,
    'DCGM_FI_PROF_PIPE_FP32_ACTIVE': 1007,
    'DCGM_FI_PROF_PIPE_FP64_ACTIVE': 1006,
    'DCGM_FI_PROF_SM_OCCUPANCY': 1003,
    'DCGM_FI_PROF_SM_ACTIVE': 1002,
    'DCGM_FI_DEV_FB_TOTAL': 250,
    'DCGM_FI_DEV_FB_FREE': 251,
    'DCGM_FI_DEV_FB_USED': 252,
    'DCGM_FI_PROF_PCIE_TX_BYTES': 1009,
    'DCGM_FI_PROF_PCIE_RX_BYTES': 1010,
    'DCGM_FI_DEV_MEM_COPY_UTIL': 204,
}


class DcgmMonitor:

  def __init__(self, device_model):
    if device_model in {'v100', 'a100'}:
      self.fields = [
          'DCGM_FI_DEV_GPU_UTIL',
          'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE',
          'DCGM_FI_PROF_PIPE_FP16_ACTIVE',
          'DCGM_FI_PROF_PIPE_FP32_ACTIVE',
          'DCGM_FI_PROF_PIPE_FP64_ACTIVE',
          'DCGM_FI_PROF_SM_OCCUPANCY',
          'DCGM_FI_PROF_SM_ACTIVE',
          'DCGM_FI_DEV_FB_USED',
          'DCGM_FI_PROF_PCIE_RX_BYTES',
          'DCGM_FI_PROF_PCIE_TX_BYTES',
          'DCGM_FI_DEV_MEM_COPY_UTIL',
          'DCGM_FI_PROF_GR_ENGINE_ACTIVE',
          'DCGM_FI_PROF_DRAM_ACTIVE',
      ]
    else:
      self.fields = [
          "DCGM_FI_DEV_GPU_UTIL",
          "DCGM_FI_DEV_FB_USED",
          "DCGM_FI_DEV_MEM_COPY_UTIL",
      ]
    self.field_ids = [DCGM_FIELDS[f] for f in fields]
    self.field_ids_str = ','.join(map(str, self.field_ids))
    self.reset()

  def reset(self):
    self.metrics = {f: [] for f in self.fields}
    self.metrics.update({
        'timestamp': [],
        'cpu_percent': [],
        'host_mem_total': [],
        'host_mem_available': [],
    })
    self.to_shutdown = False

  def sample_metrics(self, interval=1.0):
    cpu = psutil.cpu_percent(interval=interval, percpu=False)
    self.metrics['cpu_percent'].append(cpu)
    mem = psutil.virtual_memory()
    self.metrics['host_mem_total'].append(mem.total)
    self.metrics['host_mem_available'].append(mem.available)
    self.metrics['timestamp'].append(time.time())
    dcgmi_out = subprocess.check_output(
        ['dcgmi', 'dmon', '-e', self.field_ids_str, '-c', '5'],)
    dcgmi_samples = {f: [] for f in self.fields}
    for line in dcgmi_out.split('\n')[-4:-1]:
      # THIS ASSUMES THAT THE OUTPUT OF DCGM MONITOR HAS THE FORMAT GPU X METRIC1 METRIC2 ...
      for idx, val in enumerate(line.split()[2:]):
        if val == 'N/A':
          continue
        dcgmi_samples[self.fields[idx]].append(float(val))
    for f, vals in dcgmi_samples.items():
      if len(vals) > 0:
        self.metrics[f].append(sum(vals) / len(vals))
      else:
        self.metrics[f].append(float('nan'))

  def save(self, output_dir):
    csv_path = os.path.join(output_dir, 'dcgm_metrics.csv')
    pd.DataFrame(self.metrics).to_csv(csv_path)
    print('Saving metrics to {} !'.format(csv_path))


def dcgm_monitor_thread(monitor, outdir):
  while not monitor.to_shutdown:
    monitor.sample_metrics()
    time.sleep(9)
  monitor.save(outdir)
