# Measure the training throughput on A100 GPU

- [Measure the training throughput on A100 GPU](#measure-the-training-throughput-on-a100-gpu)
  * [Requirements](#requirements)
    + [Google GCP](#google-gcp)
    + [Amazon EC2](#amazon-ec2)
  * [Steps](#steps)
    + [Enable MIG](#enable-mig)
    + [Prepare codebase](#prepare-codebase)
    + [Acquiring NVIDIA Nsight Compute CLI and DCGM](#acquiring-nvidia-nsight-compute-cli-and-dcgm)
    + [Download and launch docker image](#download-and-launch-docker-image)
      - [Build docker image](#build-docker-image)
      - [Launch a docker container](#launch-a-docker-container)
    + [Install HFTA and its dependencies as a Python package](#install-hfta-and-its-dependencies-as-a-python-package)
    + [Run experiments](#run-experiments)
    + [Plot speedup curves](#plot-speedup-curves)

## Requirements

You need to have access to GPU resource on major GPU cloud platforms such as Google Cloud Platform (GCP) or Amazon EC2.


### Google GCP
If yo have never used GCP before, you should request an account/project from GCP under <https://cloud.google.com> to get access to the GPU VMs.

After getting access, navigate to the "VM Instance" pane to create an VM with A100 GPUs.
- The GPU instance we used for accessing A100 GPUs on GCP is `a2-highgpu-1g`. 
- We request `1 x NVIDIA Tesla A100` GPU. 
- The host resource contains 12 vCPUs, 85 GB memory. 
- We used a `Standard persistent disk` with 200GB of storage capacity. 

After the VM instances is created and launched, we recommend you use [gcloud command-line tool](https://cloud.google.com/sdk/gcloud) to interact with the VM.

__Note: please make sure to turn off your VM instances as soon as you finish the experiments, as the instances are quite costly__


### Amazon EC2

If you have never used Amazon EC2 before, you should request an account from <https://aws.amazon.com/ec2> to get access to the GPU VMs

After getting access, navigate to the "EC2 Dashboard" -> "Launch instance"  pane to create an VM with A100 GPUs.
- The GPU instance we used for accessing A100 GPUs on Amazon EC2 is [`p4d.24xlarge`](https://aws.amazon.com/ec2/instance-types/p4/). 
- The instance is configured as a complete machine with 8 GPUs, you cannot select the number of GPUs used
- We used `Deep Learning AMI (Ubuntu 18.04) Version 36.0(ID: 063585f0e06d22308)`
- The host resource contains 96 vCPUs, 1176 GB memory, but docker instances will limit this. 
- We used a standard EBS boot disk with 100GB of storage capacity. 

__Note: please make sure to turn off your VM instances as soon as you finish the experiments, as the instances are quite costly__


## Steps

### Enable MIG

*Note that the following commands and outputs are taken from Amazon EC2 instances, GCP can be slightly different* 

This assumes you are inside the shell of the VM and has `sudo` access.

First query through `nvidia-smi` to see if mig is enabled
```bash
ubuntu@ip-xxxxxxxx:~$ nvidia-smi -i 0 --query-gpu=pci.bus_id,mig.mode.current --format=csv
pci.bus_id, mig.mode.current
00000000:10:1C.0, Disabled
```

If the prompt shows Disabled as above, enable mig via te command below
```bash
ubuntu@ip-xxxxxxxx:~$ sudo nvidia-smi  -mig 1
Enabled MIG Mode for GPU 00000000:10:1C.0
Enabled MIG Mode for GPU 00000000:10:1D.0
Enabled MIG Mode for GPU 00000000:20:1C.0
Enabled MIG Mode for GPU 00000000:20:1D.0
Enabled MIG Mode for GPU 00000000:90:1C.0
Enabled MIG Mode for GPU 00000000:90:1D.0
Enabled MIG Mode for GPU 00000000:A0:1C.0
Enabled MIG Mode for GPU 00000000:A0:1D.0
All done.

ubuntu@ip-xxxxxxxx:~$ nvidia-smi -i 0 --query-gpu=pci.bus_id,mig.mode.current --format=csv
pci.bus_id, mig.mode.current
00000000:10:1C.0, Enabled
```

Reboot the instance for MIG configuration to take effect
```bash
ubuntu@ip-xxxxxxxx:~$ sudo reboot now
Connection to ec2-xxxxxxxx.compute-1.amazonaws.com closed by remote host.
Connection to ec2-xxxxxxxx.compute-1.amazonaws.com closed.
```

SSH into the instance again and verify
```bash
ubuntu@ip-xxxxxxxx:~$ nvidia-smi -i 0 --query-gpu=pci.bus_id,mig.mode.current --format=csv
pci.bus_id, mig.mode.current
00000000:10:1C.0, Enabled
```

### Prepare codebase 

First clone the repo and navigate to the project

```bash
# clone the code base
git clone https://github.com/UofT-EcoSystem/hfta.git
cd hfta
```

### Acquiring NVIDIA Nsight Compute CLI and DCGM

We require two installation files (`.deb`) for Nsight Compute and DCGM pre-downloaded to build the docker image.

- Nsight Compute: version `3.1.72`, downloaded under `third_party/nsys/nsys_cli_2020.3.1.72.deb`
- DCGM version: version `2.0.10`, downloaded under  `third_party/dcgm/datacenter-gpu-manager_2.0.10_amd64.deb`

In order to download the `.deb` files, you need to register a NVIDIA developer account via: <https://developer.nvidia.com/login>, after that, you can download the .deb file:
- NVIDIA Nsight Computer at: <https://developer.nvidia.com/nsight-compute>
- NVIDIA DCGM at: <https://developer.nvidia.com/dcgm>


### Download and launch docker image

Follow the commands below to prepare and launch the docker image, this will take approximately 10 mins

#### Build docker image 
```bash
# build the image, select from native1.6-cu10.2, nvidia20.06, nvidia20.08
# this will take about 10 mins to complete
bash docker/build.sh <the version of the image, e.g. nvidia20.08>
```

#### Launch a docker container 
```bash
# launch the image
# you will need to provide a placeholder mount point for data directory
# default is under ${HOME}/datasets
ubuntu@ip-xxxxxxxx:~/hfta$ bash docker/launch.sh <optional: data directory mount point>

=============
== PyTorch ==
=============

NVIDIA Release 20.08 (build 15516749)
PyTorch Version 1.7.0a0+8deb4fe

Container image Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

Copyright (c) 2014-2020 Facebook Inc.
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
Copyright (c) 2015      Google Inc.
Copyright (c) 2015      Yangqing Jia
Copyright (c) 2013-2016 The Caffe contributors
All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
NVIDIA modifications are covered by the license terms that apply to the underlying project or file.

NOTE: MOFED driver for multi-node communication was not detected.
      Multi-node communication performance may be reduced.

root@c7ee88f34a48:/home/ubuntu/hfta# 

```

###  Install HFTA and its dependencies as a Python package

```bash
root@c7ee88f34a48:/home/ubuntu/hfta: pip install -e . 

... # additional outputs not shown 
Installing collected packages: pandas, hfta
  Attempting uninstall: pandas
    Found existing installation: pandas 1.1.3
    Uninstalling pandas-1.1.3:
      Successfully uninstalled pandas-1.1.3
  Running setup.py develop for hfta
Successfully installed hfta pandas-1.1.5
```


### Run experiments

1. Prepare datasets
```bash
cd /home/ubuntu/hfta
source datasets/prepare_datasets.sh
# Download the dataset by calling helper functions defined in `prepare_datasets.sh`. For example: run
# `prepare_bert` for BERT experiment.
prepare_bert
```

2. Prepare experiment workflow helper functions
```bash
# The command below will set the target device and device model to be CUDA, A100, 
# and the output directory to be `./MLSys21/benchmarks`. 
# You can change the output directory as what you want.
source benchmarks/workflow.sh cuda a100 /MLSys21/benchmarks
```

3. Run experiments by calling workflow helper functions. The workflow functions are defined in the `_workflow_<modelname>.sh` files under `<repo root>/benchmarks`.
```bash
# The functions are generally named as `workflow_<modelname>`.
# For example, in order to run BERT experiment, do
workflow_bert
```

### Plot speedup curves

After the workflow experiment is done, run bash function below to process the output and plot the speedup curves. The plot functions are also defined in `_workflow_<modelname>.sh` files.

```bash
# In general, plot functions are defined as plot_<exp name>
# For example, for BERT experiment, run
plot_bert
```

Finally, you should be able to see the `.csv` and `.png` files under the output directory (`./MLSys21/benchmarks`).
