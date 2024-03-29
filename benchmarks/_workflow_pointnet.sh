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
  local hfta_dry_run_repeats=1

  # For TPU, we need to retry to find a stable max_B
  if [ "${DEVICE}" == "xla" ]; then
    hfta_dry_run_repeats=3
  fi

  _pointnet_warmup_data ${task}

  if [ "${task}" == "cls" ]; then
    local epochs=5
  elif [ "${task}" == "seg" ]; then
    local epochs=10
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
      --device-model ${DEVICE_MODEL} \
      --hfta-dry-run-repeats ${hfta_dry_run_repeats}
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

_plot_tpu_profile_pointnet() {
  local task=$1
  local all_outdirs="$(gsutil ls -d ${STORAGE_BUCKET}/pointnet/run*/${task})"
  local outdir_arr=($all_outdirs)
  tpu_profile_parser \
    --outdirs ${outdir_arr[@]} \
    --savedir ${OUTDIR_ROOT}/pointnet/tpuprofile-${task}-${DEVICE}-${DEVICE_MODEL}/ \
    --plot
}

workflow_pointnet_cls () {
  local repeats=${1:-"3"}
  _workflow_pointnet cls ${repeats}
}

plot_pointnet_cls () {
  _plot_speedups_pointnet cls
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_pointnet cls
  elif [ "${DEVICE}" == "xla" ]; then
    _plot_tpu_profile_pointnet cls
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
  elif [ "${DEVICE}" == "xla" ]; then
    _plot_tpu_profile_pointnet seg
  fi
}
