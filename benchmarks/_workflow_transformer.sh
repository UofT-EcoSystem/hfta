#!/bin/bash

_workflow_transformer () {
  local repeats=$1
  local epochs=5
  local iters_per_epoch=1000

  local i
  for ((i=0; i<${repeats}; i++)); do
    python benchmarks/transformer.py \
      --outdir_root ${OUTDIR_ROOT}/transformer/run${i} \
      --epochs ${epochs} \
      --iters-per-epoch ${iters_per_epoch} \
      --dataroot datasets/wikitext-2/ \
      --device ${DEVICE} \
      --device-model ${DEVICE_MODEL}
  done
}

_plot_speedups_transformer() {
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/transformer/run*/
  do
    outdirs+=(${outdir})
  done
  timing_parser \
    --outdirs "${outdirs[@]}" \
    --device ${DEVICE}\
    --device-model ${DEVICE_MODEL} \
    --save ${OUTDIR_ROOT}/transformer/${DEVICE}-${DEVICE_MODEL} \
    --plot
}

_plot_dcgm_transformer() {
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/transformer/run*/
  do
    outdirs+=(${outdir})
  done
  dcgm_parser \
    --outdirs "${outdirs[@]}" \
    --device-model ${DEVICE_MODEL} \
    --savedir ${OUTDIR_ROOT}/transformer/dcgm-${DEVICE}-${DEVICE_MODEL}/ \
    --plot
}


workflow_transformer () {
  local repeats=${1:-"3"}
  _workflow_transformer ${repeats}
}

plot_transformer () {
  _plot_speedups_transformer
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_transformer
  fi
}

