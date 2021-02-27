#!/bin/bash

_workflow_resnet_ensemble () {
  local repeats=$1
  local epochs=5
  local iters_per_epoch=1000

  local i
  for ((i=0; i<${repeats}; i++)); do
    python benchmarks/resnet.py \
      --outdir_root ${OUTDIR_ROOT}/resnet/run${i} \
      --epochs ${epochs} \
      --iters-per-epoch ${iters_per_epoch} \
      --dataroot datasets/ \
      --device ${DEVICE} \
      --device-model ${DEVICE_MODEL} \
      --ensemble
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
      --dataroot datasets/ \
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

workflow_resnet_ensemble () {
  local repeats=${1:-"3"}
  _workflow_resnet_ensemble ${repeats}
}

plot_resnet () {
  _plot_speedups_resnet
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_resnet
  fi
}

