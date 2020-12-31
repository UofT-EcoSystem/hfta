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
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/pointnet/run*/${task}/
  do
    outdirs+=(${outdir})
  done
  timing_parser \
    --outdirs "${outdirs[@]}" \
    --device ${DEVICE}\
    --device-model ${DEVICE_MODEL} \
    --save ${OUTDIR_ROOT}/pointnet/${task}-${DEVICE}-${DEVICE_MODEL} \
    --plot
}

_plot_dcgm_pointnet() {
  local task=$1
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/pointnet/run*/${task}/
  do
    outdirs+=(${outdir})
  done
  dcgm_parser \
    --outdirs "${outdirs[@]}" \
    --device-model ${DEVICE_MODEL} \
    --savedir ${OUTDIR_ROOT}/pointnet/dcgm-${task}-${DEVICE}-${DEVICE_MODEL}/ \
    --plot
}


_plot_dcgm_dcgan() {
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/dcgan/run*/
  do
    outdirs+=(${outdir})
  done
  dcgm_parser \
    --outdirs "${outdirs[@]}" \
    --device-model ${DEVICE_MODEL} \
    --savedir ${OUTDIR_ROOT}/dcgan/dcgm-${DEVICE}-${DEVICE_MODEL}/ \
    --plot
}

_plot_speedups_dcgan() {
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/dcgan/run*/
  do
    outdirs+=(${outdir})
  done
  timing_parser \
    --outdirs "${outdirs[@]}" \
    --device ${DEVICE}\
    --device-model ${DEVICE_MODEL} \
    --save ${OUTDIR_ROOT}/dcgan/${DEVICE}-${DEVICE_MODEL} \
    --plot
}

workflow_dcgan () {
  local repeats=${1:-"3"}
  local epochs=5

  local i
  for ((i=0; i<${repeats}; i++)); do
    python3.6 benchmarks/dcgan.py \
      --outdir_root ${OUTDIR_ROOT}/dcgan/run${i}/ \
      --epochs ${epochs} \
      --iters-per-epoch 500 \
      --dataroot ../datasets/lsun_small/ \
      --device ${DEVICE} \
      --device-model ${DEVICE_MODEL}
  done
}


workflow_pointnet_cls () {
  local repeats=${1:-"3"}
  _workflow_pointnet cls ${repeats}
}

plot_pointnet_cls () {
  _plot_speedups_pointnet cls
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_pointnet cls
  fi
}

workflow_pointnet_seg () {
  local repeats=${1:-"3"}
  _workflow_pointnet seg ${repeats}
}

plot_pointnet_seg () {
  _plot_speedups_pointnet seg
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_pointnet seg
  fi
}
