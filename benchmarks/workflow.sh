#!/bin/bash

DEVICE=${1:-"cuda"}
DEVICE_MODEL=${2:-"v100"}
OUTDIR_ROOT=${3:-"benchmarks/"}

_workflow_pointnet () {
  local task=$1
  local repeats=$2
  if [ "${task}" == "cls" ]; then
    local epochs=5
    prepare_pointnet_cls
  elif [ "${task}" == "seg" ]; then
    local epochs=10
    prepare_pointnet_seg
  else
    echo "Unknown task: ${task} !"
    return -1
  fi

  local i
  for ((i=0; i<${repeats}; i++)); do
    python benchmarks/pointnet.py \
      --outdir_root ${OUTDIR_ROOT}/pointnet/run${i} \
      --epochs ${epochs} \
      --iters-per-epoch 1000 \
      --dataroot datasets/shapenetcore_partanno_segmentation_benchmark_v0/ \
      --task ${task} \
      --device ${DEVICE} \
      --device-model ${DEVICE_MODEL}
  done
}

_plot_speedups_pointnet() {
  local task=$1
  local repeats=$2
  local outdirs=()
  for ((i=0; i<${repeats}; i++)); do
    outdirs+=("${OUTDIR_ROOT}/pointnet/run${i}/${task}/")
  done
  timing_parser \
    --outdirs "${array[@]}" \
    --device ${DEVICE}\
    --device-model ${DEVICE_MODEL} \
    --save ${OUTDIR_ROOT}/pointnet/${task}-${DEVICE}-${DEVICE_MODEL} \
    --plot
}

workflow_pointnet_cls () {
  local repeats=${1:-"3"}
  _workflow_pointnet cls ${repeats}
  _plot_speedups_pointnet cls ${repeats}
}

workflow_pointnet_seg () {
  local repeats=${1:-"3"}
  _workflow_pointnet seg ${repeats}
  _plot_speedups_pointnet seg ${repeats}
}
