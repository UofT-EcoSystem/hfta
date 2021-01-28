#!/bin/bash

_pointnet_warmup_data () {
  local task=$1
  if [ "${task}" == "cls" ]; then
    python examples/pointnet/train_classification.py \
      --epochs 1 \
      --iters-per-epoch 1000 \
      --dataset datasets/shapenetcore_partanno_segmentation_benchmark_v0/ \
      --eval \
      --device cpu \
      --warmup-data-loading
  elif [ "${task}" == "seg" ]; then
    python examples/pointnet/train_segmentation.py \
      --epochs 1 \
      --iters-per-epoch 1000 \
      --dataset datasets/shapenetcore_partanno_segmentation_benchmark_v0/ \
      --device cpu \
      --warmup-data-loading
  else
    echo "Unknown task: ${task} !"
    return -1
  fi
}

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

