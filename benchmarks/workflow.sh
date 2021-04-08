#!/bin/bash

DEVICE=${1:-"cuda"}
DEVICE_MODEL=${2:-"v100"}
OUTDIR_ROOT=${3:-"benchmarks/"}

# Note that the variables above are used in the sourced files below
. benchmarks/_workflow_pointnet.sh
. benchmarks/_workflow_dcgan.sh
. benchmarks/_workflow_mobilenet.sh
. benchmarks/_workflow_transformer.sh
. benchmarks/_workflow_bert.sh
. benchmarks/_workflow_resnet.sh
