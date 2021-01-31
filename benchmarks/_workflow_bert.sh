#!/bin/bash

_workflow_bert () {
  local repeats=$1
  local epochs=5
  local iters_per_epoch=1000

  if [ "${DEVICE}" == "cuda" ]; then
    if [ "${DEVICE_MODEL}" == "a100" ]; then
      local modes_flag="--modes serial concurrent mps hfta mig"
    else
      local modes_flag="--modes serial concurrent mps hfta"
    fi
  elif [ "${DEVICE}" == "xla" ]; then
    local modes_flag="--modes serial hfta"
  elif [ "${DEVICE}" == "cpu" ]; then
    local modes_flag="--modes serial concurrent hfta"
  else
    echo "Unknown device: ${DEVICE} !"
    return -1
  fi

  local i
  for ((i=0; i<${repeats}; i++)); do
    python benchmarks/bert.py \
      --outdir_root ${OUTDIR_ROOT}/bert/run${i} \
      --epochs ${epochs} \
      --iters-per-epoch ${iters_per_epoch} \
      --dataroot datasets/wikitext-2/ \
      ${modes_flag} \
      --device ${DEVICE} \
      --device-model ${DEVICE_MODEL}
  done
}

_plot_speedups_bert() {
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/bert/run*/
  do
    outdirs+=(${outdir})
  done
  timing_parser \
    --outdirs "${outdirs[@]}" \
    --device ${DEVICE}\
    --device-model ${DEVICE_MODEL} \
    --save ${OUTDIR_ROOT}/bert/${DEVICE}-${DEVICE_MODEL} \
    --plot
}

_plot_dcgm_bert() {
  local outdirs=()
  for outdir in ${OUTDIR_ROOT}/bert/run*/
  do
    outdirs+=(${outdir})
  done
  dcgm_parser \
    --outdirs "${outdirs[@]}" \
    --device-model ${DEVICE_MODEL} \
    --savedir ${OUTDIR_ROOT}/bert/dcgm-${DEVICE}-${DEVICE_MODEL}/ \
    --plot
}


workflow_bert () {
  local repeats=${1:-"3"}
  _workflow_bert ${repeats}
}

plot_bert () {
  _plot_speedups_bert
  if [ "${DEVICE}" == "cuda" ]; then
    _plot_dcgm_bert
  fi
}

