#!/bin/bash

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

