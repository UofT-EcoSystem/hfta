# Measure the training throughput on TPU v3

- [Measure the training throughput on TPU v3](#measure-the-training-throughput-on-tpu-v3)
  - [Requirements](#requirements)
  - [Steps](#steps)
    - [Set up TPU instance and VM](#set-up-tpu-instance-and-vm)
    - [Prepare codebase and download docker image](#prepare-codebase-and-download-docker-image)
    - [Run experiments](#run-experiments)
    - [Plot speedup curves](#plot-speedup-curves)

## Requirements

You need to have access to TPU resource on Google Cloud Platform (GCP).

## Steps

### Set up TPU instance and VM

1. Follow instructions [here](https://github.com/pytorch/xla#VMImage) to set up GCP VM and TPU instance.
2. The type of VM machine used in our experiments is `n1-highmem-8 (8 vCPUs, 52 GB memory)`.
3. The VM disk type is `Standard persistent disk` with 120 GB capacity.
4. The type of TPU instance is `v3-8` with software version `pytorch-1.7`.

### Prepare codebase and download docker image

1. Start the VM and TPU instance you just created and ensure the VM knows the TPU IP address (See instructions [here](https://github.com/pytorch/xla#VMImage)).
2. Clone this repo: `git clone https://github.com/UofT-EcoSystem/hfta.git`
3. `cd hfta`
4. Download and enter the docker image: `bash docker/launch_xla.sh`. The docker image will generally be more than 20 GB.
5. Install basic requirements for HFTA: `pip install -e .[xla]`
6. Install additional requirements for benchmarking: `pip install plyfile`

### Run experiments

1. Prepare datasets
   1. Under the root directory of the repo, run `source datasets/prepare_datasets.sh`.
   2. Download the dataset by calling helper functions defined in `prepare_datasets.sh`. For example: run `prepare_bert` for BERT experiment.
2. Prepare experiment workflow helper functions
   1. Under the root directory of the repo, run `source benchmarks/workflow.sh xla v3 ./MLSys21/benchmarks`.
   2. The above command will set the target device to be `TPU v3` and the output directory to be `./MLSys21/benchmarks`. You can change the output directory as what you want.
3. Run experiments by calling workflow helper functions
   1. For example, in order to run BERT experiment, call bash function: `workflow_bert`
   2. The workflow functions are defined in the `_workflow_<modelname>.sh` files under `<repo root>/benchmarks`.
   3. The functions are generally named as `workflow_<modelname>`.

### Plot speedup curves

1. After the workflow experiment is done, run bash function to process the output and plot the speedup curves.
2. For example, run `plot_bert` for BERT experiment.
3. The plot functions are also defined in `_workflow_<modelname>.sh` files.
4. Finally, you should be able to see the `.csv` and `.png` files under the output directory (`./MLSys21/benchmarks`).
