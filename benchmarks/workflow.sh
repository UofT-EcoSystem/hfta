#!/bin/bash

DEVICE=${1:-"cuda"}
DEVICE_MODEL=${2:-"v100"}
OUTDIR_ROOT=${3:-"benchmarks/"}

source benchmarks/utils.sh

_workflow_pointnet () {
  local task=$1
  local repeats=$2
  local use_mig=${3:-"false"}

  _pointnet_warmup_data ${task}

  if [ "${task}" == "cls" ]; then
    local epochs=5
  elif [ "${task}" == "seg" ]; then
    local epochs=10
  else
    echo "Unknown task: ${task} !"
    return -1
  fi

  if [ "${use_mig}" == "true" ]; then
    local modes_flag="--modes mig"
  elif [ "${use_mig}" == "false" ]; then
    if [ "${DEVICE}" == "cuda" ]; then
      local modes_flag="--modes serial concurrent mps hfta"
    elif [ "${DEVICE}" == "xla" ]; then
      local modes_flag="--modes serial hfta"
    elif [ "${DEVICE}" == "cpu" ]; then
      local modes_flag="--modes serial concurrent hfta"
    else
      echo "Unknown device: ${DEVICE} !"
      return -1
    fi
  else
    echo "Unknown use_mig: ${use_mig} !"
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
      ${modes_flag} \
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

_dcgan_warmup_data() {
  local dataroot=${1:-"datasets/lsun/"}
  md5sum ${dataroot}/bedroom_train_lmdb/data.mdb > /dev/null
  md5sum ${dataroot}/bedroom_train_lmdb/lock.mdb > /dev/null
}

workflow_dcgan () {
  local repeats=${1:-"3"}
  local epochs=5
  local dataroot="datasets/lsun/"

  _dcgan_warmup_data ${dataroot}
  local i
  for ((i=0; i<${repeats}; i++)); do
    python benchmarks/dcgan.py \
      --outdir_root ${OUTDIR_ROOT}/dcgan/run${i}/ \
      --epochs ${epochs} \
      --iters-per-epoch 300 \
      --modes serial hfta \
      --dataroot ${dataroot} \
      --device ${DEVICE} \
      --device-model ${DEVICE_MODEL}
  done

  if [ "${DEVICE}" != "cuda" ]; then
    return
  fi

  local precs=("fp32" "amp")
  local modes=("mps" "concurrent")
  for ((i=0; i<${repeats}; i++)); do
    for mode in ${modes[@]}; do
      for prec in ${precs[@]}; do
        echo "${mode}" "${prec}"
        _dcgan_warmup_data ${dataroot}
        python benchmarks/dcgan.py \
          --outdir_root ${OUTDIR_ROOT}/dcgan/run${i}/ \
          --epochs ${epochs} \
          --iters-per-epoch 300 \
          --modes ${mode} \
          --prec ${prec} \
          --dataroot ${dataroot} \
          --device ${DEVICE} \
          --device-model ${DEVICE_MODEL}
      done
    done
  done
}

workflow_dcgan_mig () {
  local repeats=${1:-"3"}
  local epochs=5
  local dataroot="datasets/lsun/"

  _dcgan_warmup_data ${dataroot}
  local i
  for ((i=0; i<${repeats}; i++)); do
    python benchmarks/dcgan.py \
      --outdir_root ${OUTDIR_ROOT}/dcgan/run${i}/ \
      --epochs ${epochs} \
      --iters-per-epoch 300 \
      --modes mig \
      --dataroot ${dataroot} \
      --device ${DEVICE} \
      --device-model ${DEVICE_MODEL}
  done
}

plot_dcgan () {
  _plot_speedups_dcgan
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_dcgan
  fi
}

workflow_pointnet_cls () {
  local repeats=${1:-"3"}
  _workflow_pointnet cls ${repeats}
}

workflow_pointnet_cls_mig () {
  local repeats=${1:-"3"}
  _workflow_pointnet cls ${repeats} true
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

workflow_pointnet_seg_mig () {
  local repeats=${1:-"3"}
  _workflow_pointnet seg ${repeats} true
}

plot_pointnet_seg () {
  _plot_speedups_pointnet seg
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_pointnet seg
  fi
}
