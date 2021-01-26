#!/bin/bash

_mobilenet_warmup_data() {
  local dataset=$1
  local version=$2

  # warm up only required for imagenet
  if [ "${dataset}" == "imagenet" ]; then
    python examples/mobilenet/main.py \
      --dataset imagenet\
      --version ${version} \
      --epochs 1 \
      --dataroot ./datasets/imagenet \
      --eval \
      --warmup-data-loading
  elif [ "${dataset}" == "cifar10" ]; then
    echo "dataset ${dataset} skipped for warmup"
    return 0
  else
    echo "Unknown dataset"
    return -1
  fi
}


_workflow_mobilenet() {
  local dataset=${1:-"cifar10"} # cifar10 or imagenet
  local version=${2:-"v2"} # v2 v3s or v3l
  local repeats=${3:-"3"}

  _mobilenet_warmup_data ${dataset} ${version}

  local epochs=5
  local i
  for ((i=0; i<${repeats}; i++)); do
    python benchmarks/mobilenet.py \
      --dataset ${dataset} \
      --version ${version} \
      --outdir_root ${OUTDIR_ROOT}/mobilenet/run${i}/ \
      --epochs ${epochs} \
      --iters-per-epoch 500 \
      --dataroot ./datasets/${dataset} \
      --device ${DEVICE} \
      --device-model ${DEVICE_MODEL}
  done
}

_plot_speedups_mobilenet() {
  local dataset=$1
  local version=$2
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/mobilenet/run*/${dataset}/${version}
  do
    outdirs+=(${outdir})
  done
  timing_parser \
    --outdirs "${outdirs[@]}" \
    --device ${DEVICE}\
    --device-model ${DEVICE_MODEL} \
    --save ${OUTDIR_ROOT}/mobilenet/${DEVICE}-${DEVICE_MODEL} \
    --plot
}

_plot_dcgm_mobilenet() {
  local dataset=$1
  local version=$2
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/mobilenet/run*/${dataset}/${version}
  do
    outdirs+=(${outdir})
  done
  dcgm_parser \
    --outdirs "${outdirs[@]}" \
    --device-model ${DEVICE_MODEL} \
    --savedir ${OUTDIR_ROOT}/mobilenet/dcgm-${DEVICE}-${DEVICE_MODEL}/ \
    --plot
}

workflow_mobilenet_imangenet_v2() {
  local repeats=${1:-"3"}
  _workflow_mobilenet imagenet v2 ${repeats}
}

workflow_mobilenet_imangenet_v3s() {
  local repeats=${1:-"3"}
  _workflow_mobilenet imagenet v3s ${repeats}
}

workflow_mobilenet_imangenet_v3l() {
  local repeats=${1:-"3"}
  _workflow_mobilenet imagenet v3l ${repeats}
}

workflow_mobilenet_cifar10_v2() {
  local repeats=${1:-"3"}
  _workflow_mobilenet cifar10 v2 ${repeats}
}

workflow_mobilenet_cifar10_v3s() {
  local repeats=${1:-"3"}
  _workflow_mobilenet cifar10 v3s ${repeats}
}

workflow_mobilenet_cifar10_v3l() {
  local repeats=${1:-"3"}
  _workflow_mobilenet cifar10 v3l ${repeats}
}

plot_mobilenet_imagenet_v2() {
  _plot_speedups_mobilenet imagenet v2
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_mobilenet imagenet v2
  fi
}

plot_mobilenet_imagenet_v3s() {
  _plot_speedups_mobilenet imagenet v3s
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_mobilenet imagenet v3s
  fi
}

plot_mobilenet_imagenet_v3l() {
  _plot_speedups_mobilenet imagenet v3l
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_mobilenet imagenet v3l
  fi
}


plot_mobilenet_cifar10_v2() {
  _plot_speedups_mobilenet cifar10 v2
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_mobilenet cifar10 v2
  fi
}

plot_mobilenet_cifar10_v3s() {
  _plot_speedups_mobilenet cifar10 v3s
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_mobilenet cifar10 v3s
  fi
}

plot_mobilenet_cifar10_v3l() {
  _plot_speedups_mobilenet cifar10 v3l
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_mobilenet cifar10 v3l
  fi
}
