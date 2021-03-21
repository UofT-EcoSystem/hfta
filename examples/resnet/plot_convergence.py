import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
  parser = argparse.ArgumentParser(description='Plot Convergence Curve.')
  parser.add_argument('--merge-size', type=int, default=1)
  parser.add_argument('--outdir', type=str, required=True)
  parser.add_argument(
      '--device',
      type=str,
      default="cuda",
      choices=['cpu', 'cuda', 'xla'],
      help='cpu, cuda or xla',
  )
  parser.add_argument(
      '--device-model',
      type=str,
      default="v100",
      help='The model of the device (e.g, v100, a100, rtx6000 or TPU v3)',
  )
  parser.add_argument(
      '--prec',
      type=str,
      default='fp32',
      choices=['fp32', 'amp', 'bf16'],
      help='training precision(s)',
  )
  args = parser.parse_args()
  outdir = os.path.join(args.outdir, args.device, args.device_model, args.prec)

  serial_data = {}
  hfta_data = {}
  hfta_raw_data = pd.read_csv(os.path.join(outdir, "hfta", "convergence.csv"))
  lrs = [float(lr) for lr in hfta_raw_data.columns[1:].values]
  for i, lr in enumerate(lrs):
    data = pd.read_csv(
        os.path.join(outdir, "serial/lr_{}".format(lr), "convergence.csv"))
    serial_data[lr] = data[str(lr)].values
    hfta_data[lr] = hfta_raw_data[str(lr)].values

  for i, lr in enumerate(lrs):
    hfta_data[lr] = hfta_data[lr].reshape((-1, args.merge_size)).mean(axis=1)
    serial_data[lr] = serial_data[lr].reshape(
        (-1, args.merge_size)).mean(axis=1)

  # NOTE only support 3 different results.
  # Ploting more results requires more colors
  color_serial = ["blue", "forestgreen", "cornflowerblue"]
  color_hfta = ["red", "darkviolet", "darkorange"]
  plt.rcParams['savefig.dpi'] = 500
  for i, lr in enumerate(lrs):
    idx = np.linspace(1,
                      len(hfta_data[lr]) * args.merge_size + 1,
                      len(hfta_data[lr]))
    plt.plot(
        idx,
        serial_data[lr],
        label='Serial:LR={}'.format(lr),
        color=color_serial[i],
        linewidth=0.5,
        linestyle="-",
    )
    plt.plot(
        idx,
        hfta_data[lr],
        label='HFTA:LR={}'.format(lr),
        color=color_hfta[i],
        linewidth=0.5,
        linestyle="--",
    )
  plt.xlabel("Train iterations")
  plt.ylabel("Train loss")
  plt.savefig(os.path.join(outdir, "convergence.png"))
  plt.close()


if __name__ == '__main__':
  main()
