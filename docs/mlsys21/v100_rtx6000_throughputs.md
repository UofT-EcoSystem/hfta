  # Measure the training throughput on V100 or RTX6000 GPUs

- [Measure the training throughput on V100 or RTX6000 GPUs](#measure-the-training-throughput-on-v100-or-rtx6000-gpus)
  - [Requirements](#requirements)
    - [Amazon EC2 for V100](#amazon-ec2-for-v100)
    - [Local Machine for RTX6000](#local-machine-for-rtx6000)
  - [Steps](#steps)
    - [Prepare codebase](#prepare-codebase)
    - [Acquiring NVIDIA Nsight Systems CLI and DCGM](#acquiring-nvidia-nsight-systems-cli-and-dcgm)
    - [Download and launch docker image](#download-and-launch-docker-image)
      - [Build docker image](#build-docker-image)
      - [Reuse prebuilt docker image](#reuse-prebuilt-docker-image)
      - [Launch a docker container](#launch-a-docker-container)
    - [Install HFTA and its dependencies as a Python package](#install-hfta-and-its-dependencies-as-a-python-package)
    - [Run experiments](#run-experiments)
    - [Plot speedup curves](#plot-speedup-curves)

## Requirements

For V100, you need to have access to GPU resources on major GPU cloud platforms such as Amazon EC2. We will give an example of Amazon EC2. 

For RTX6000, you need to have access to a local machine that has at least one RTX6000 GPU.


### Amazon EC2 for V100

If you have never used Amazon EC2 before, you should request an account from <https://aws.amazon.com/ec2> to get access to the GPU VMs.

After getting access, navigate to the "EC2 Dashboard" -> "Launch instance"  pane to create an VM with V100 GPUs.
- The GPU instance we used for accessing V100 GPUs on Amazon EC2 is [`p3.2xlarge`](https://aws.amazon.com/ec2/instance-types/p3/).
- The p3.2xlarge instance contains 8 vCPUs and 61 GB host memory. If you selected a larger instance with more GPUs, docker can limit the amount of host resource allocated per GPU.
- We used `NVIDIA Deep Learning AMI v20.06.3`.
- The host resource contains 8 vCPUs, 61 GB memory, but docker instances will limit this.
- We used a standard EBS boot disk with 100GB of storage capacity.

__Note: please make sure to turn off your VM instances as soon as you finish the experiments, as the instances are quite costly__

### Local Machine for RTX6000
If you have a machine with at least one RTX6000, you need to configure the NVIDIA driver and nvidia-docker properly. Please refer to the steps in https://github.com/NVIDIA/nvidia-docker#getting-started on how to setup a fresh machine for our experiments.

The software specification of the machine with RTX6000 under our experiments is:
> NVIDIA GPU Driver Version: 450.66
> OS: Ubuntu 18.04.5 LTS
> Docker Version: 20.10.2, build 2291f61
> `nvidia-docker2` Version: 2.5.0-1

## Steps



### Prepare codebase

First, clone the repo and navigate to the project.

```bash
# clone the code base
git clone https://github.com/UofT-EcoSystem/hfta.git
cd hfta
```

### Acquiring NVIDIA Nsight Systems CLI and DCGM

We require two installation files (`.deb`) for Nsight Systems and DCGM pre-downloaded to build the docker image.

- Nsight Systems: version `3.1.72`, downloaded under `third_party/nsys/nsys_cli_2020.3.1.72.deb`
- DCGM version: version `2.0.10`, downloaded under  `third_party/dcgm/datacenter-gpu-manager_2.0.10_amd64.deb`

In order to download the `.deb` files, you need to register a NVIDIA developer account via: <https://developer.nvidia.com/login>, after that, you can download the .deb file:
- NVIDIA Nsight Systems CLI at: <https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2020-3> (Select the "Linux CLI Only" option)
- NVIDIA DCGM at: <https://developer.nvidia.com/dcgm>


### Download and launch docker image

Follow the commands below to prepare and launch the docker image, this will take approximately 10 mins.

#### Build docker image
```bash
# build the image, select native1.6-cu10.2 for V100 and RTX6000 
# this will take about 10 mins to complete
bash docker/build.sh native1.6-cu10.2
```

#### Reuse prebuilt docker image
If you do not wish to build the docker image from scratch, you can reuse the
prebuilt docker image that we provide:
```
docker pull wangshangsam/hfta:mlsys21_native1.6-cu10.2
docker tag wangshangsam/hfta:mlsys21_native1.6-cu10.2 hfta:dev
```


#### Launch a docker container
```bash
# launch the image
# you will need to provide a placeholder mount point for the data directory
# default is under ${HOME}/datasets
ubuntu@ip-xxxxxxxx:~/hfta$ bash docker/launch.sh <optional: data directory mount point> <optional: image tag>

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

2. Prepare experiment workflow helper functions by running benchmarks/workflow.sh
benchmarks/workflow.sh" expects 3 arguments in total
- DEVICE: {cuda, tpu}, default is `cuda`
- DEVICE_MODEL: {a100, v100, etc, or v2/v3 for tpu}, default is `v100`
- OUTDIR_ROOT: {the output directory of the benchmarked results}, default is `benchmarks`

The first two arguments, are often needed in every run,. Please refer to the script for the usage of the arguments
```bash
# The command below will set the target device and device model to be CUDA, V100,
# and the output directory by default is ./benchmarks, but you can specify other things
source benchmarks/workflow.sh cuda v100 <optional: output root dir >
# For CUDA, RTX6000
source benchmarks/workflow.sh cuda rtx6000 <optional: output root dir>
```

3. Run experiments by calling workflow helper functions. The workflow functions are defined in the `_workflow_<modelname>.sh` files under `<repo root>/benchmarks`.
```bash
# The functions are generally named as `workflow_<modelname>`.
# For example, in order to run BERT experiment, run
workflow_bert
# For partially fused Rsenet experiment, run
workflow_resnet_partially_fused
# For Rsenet convergence experiment, run
workflow_convergence 
```

### Plot speedup curves

After the workflow experiment is done, run bash function below to process the output and plot the speedup curves. The plot functions are also defined in `_workflow_<modelname>.sh` files.

```bash
# In general, plot functions are defined as plot_<exp name>
# For example, for BERT experiment, run
plot_bert
# For partially fused Rsenet experiment, run
plot_resnet_partially_fused
# For Rsenet convergence experiment, run
plot_resnet_convergence
```

Finally, you should be able to see the `.csv` and `.png` files under the output directory (`./benchmarks (or the directory you specified above)`).