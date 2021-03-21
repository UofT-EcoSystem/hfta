#!/bin/bash

_workflow_resnet_partially_fused () {
  local repeats=$1
  local epochs=5
  local iters_per_epoch=1000

  local i
  for ((i=0; i<${repeats}; i++)); do
    python benchmarks/resnet.py \
      --outdir_root ${OUTDIR_ROOT}/resnet/run_partially_fused${i} \
      --epochs ${epochs} \
      --iters-per-epoch ${iters_per_epoch} \
      --dataroot datasets/cifar10/ \
      --device ${DEVICE} \
      --device-model ${DEVICE_MODEL} \
      --partially-fused
  done
}

_workflow_resnet () {
  local repeats=$1
  local epochs=5
  local iters_per_epoch=1000

  local i
  for ((i=0; i<${repeats}; i++)); do
    python benchmarks/resnet.py \
      --outdir_root ${OUTDIR_ROOT}/resnet/run${i} \
      --epochs ${epochs} \
      --iters-per-epoch ${iters_per_epoch} \
      --dataroot datasets/cifar10/ \
      --device ${DEVICE} \
      --device-model ${DEVICE_MODEL}
  done
}

_plot_speedups_resnet() {
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/resnet/run*/
  do
    outdirs+=(${outdir})
  done
  timing_parser \
    --outdirs "${outdirs[@]}" \
    --device ${DEVICE}\
    --device-model ${DEVICE_MODEL} \
    --save ${OUTDIR_ROOT}/resnet/${DEVICE}-${DEVICE_MODEL} \
    --plot
}

_plot_dcgm_resnet() {
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/resnet/run*/
  do
    outdirs+=(${outdir})
  done
  dcgm_parser \
    --outdirs "${outdirs[@]}" \
    --device-model ${DEVICE_MODEL} \
    --savedir ${OUTDIR_ROOT}/resnet/dcgm-${DEVICE}-${DEVICE_MODEL}/ \
    --plot
}


workflow_resnet () {
  local repeats=${1:-"3"}
  _workflow_resnet ${repeats}
}

workflow_resnet_partially_fused () {
  local repeats=${1:-"3"}
  _workflow_resnet_partially_fused ${repeats}
}

workflow_convergence () {
  local epochs=100
  local iters_per_epoch=1000

  python benchmarks/resnet.py \
    --outdir_root ${OUTDIR_ROOT}/resnet/run_convergence \
    --epochs ${epochs} \
    --iters-per-epoch ${iters_per_epoch} \
    --dataroot datasets/cifar10/ \
    --device ${DEVICE} \
    --device-model ${DEVICE_MODEL} \
    --convergence
}

plot_resnet () {
  _plot_speedups_resnet
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_resnet
  fi
}

plot_resnet_partially_fused () {
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/resnet/run_partially_fused*/
  do
    python ./examples/resnet/plot_partially_fused.py \
      --outdir ${outdir}
  done
}

plot_resnet_convergence () {
  local outdir=${OUTDIR_ROOT}/resnet/run_convergence/
  python ./examples/resnet/plot_convergence.py \
    --device ${DEVICE} \
    --device-model ${DEVICE_MODEL} \
    --prec fp32 \
    --merge-size 100 \
    --outdir ${OUTDIR_ROOT}/resnet/run_convergence/
}