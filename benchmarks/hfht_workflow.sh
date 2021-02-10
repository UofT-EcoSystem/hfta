#!/bin/bash

DEVICE=${1:-"cuda"}
DEVICE_MODEL=${2:-"v100"}
OUTDIR_ROOT=${3:-"benchmarks/hfht"}

source benchmarks/utils.sh

_get_modes() {
  if [ "${DEVICE}" == "cuda" ]; then
    modes=("serial" "concurrent" "mps" "hfta")
  elif [ "${DEVICE}" == "xla" ]; then
    modes=("serial" "hfta")
  elif [ "${DEVICE}" == "cpu" ]; then
    modes=("serial" "concurrent" "hfta")
  else
    echo "Unknown DEVICE ${DEVICE} !"
    return -1
  fi
}

_get_precs() {
  if [ "${DEVICE}" == "cuda" ]; then
    precs=("fp32" "amp")
  elif [ "${DEVICE}" == "xla" ]; then
    precs=("bf16")
  elif [ "${DEVICE}" == "cpu" ]; then
    precs=("fp32")
  else
    echo "Unknown DEVICE ${DEVICE} !"
    return -1
  fi
}

_sweep() {
  local base_cmd=$1
  local outdir_root=$2
  local repeats=$3
  local modes
  _get_modes
  local precs
  _get_precs
  for ((i=0; i<${repeats}; i++)); do
    for algorithm in random hyperband
    do
      local cmd_algo="${base_cmd} --algorithm ${algorithm}"
      for mode in "${modes[@]}"
      do
        local cmd_mode="${cmd_algo} --mode ${mode}"
        for prec in "${precs[@]}"
        do
          local cmd=${cmd_mode}
          if [ "${prec}" == "amp" ]; then
            cmd+=" --amp"
          fi
          if [ "${prec}" == "fp32" ] && [ "${DEVICE}" == "cuda" ] \
              && [ "${DEVICE_MODEL}" == "a100" ]; then
            cmd="NVIDIA_TF32_OVERRIDE=0 ${cmd}"
          fi
          cmd+=" --outdir ${outdir_root}/run${i}/${algorithm}/${DEVICE}/${DEVICE_MODEL}/${prec}/${mode}"
          echo "Running ${cmd} ..."
          eval ${cmd}
        done
      done
    done
  done
}


hfht_workflow_pointnet_cls() {
  local repeats=${1:-"3"}
  local base_cmd="\
    python examples/hfht/pointnet_classification.py \
    --dataset datasets/shapenetcore_partanno_segmentation_benchmark_v0/ \
    --device ${DEVICE}"
  echo "Warmup ..."
  _pointnet_warmup_data cls
  _sweep "${base_cmd}" ${OUTDIR_ROOT}/pointnet_cls ${repeats}
}
