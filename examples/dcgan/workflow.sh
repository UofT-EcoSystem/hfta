#!/bin/bash

OUTPUTDIR=${1:-"output"}
DATAROOT=${2:-"/home/ubuntu/datasets/lsun"}
DEVICE=${3:-"cuda"}
DEVICE_MODEL=${4:-"v100"}  # "v100", "v3"
EPOCHS=${5:-"5"}
ITERS_PER_EPOCH=${6:-"300"}
MAX_B=${7:-"1000"}
PYTHON=${8:-"python"}
PYTHON2=${9:-"/usr/bin/python2.7"}
SUDO=${10:-""}

generate_lrs() {
  # Generate a list of lrs based on the number of jobs.
  # Inputs:
  #   $1 : Number of jobs.
  # Outputs:
  #   ${lrs} : A bash array that contains the list of learning rates to try.
  local B=$1
  local rand_nums=(`shuf -i 1000-3000 -n ${B}`)
  lrs=()
  local n
  for n in ${rand_nums[@]}; do
    lrs+=(0.000${n})
  done
}

create_output_dir() {
  # Create an output dir if not existed and return its path.
  # Inputs:
  #   $1 : "serial", "concurrent", "concurrent_mps" or "hfta".
  #   $2 : "cpu", "cuda", "xla".
  #   $3 : "v100", "v3"
  #   $4 : "fp32", "amp"
  #   $5 : Number of jobs.
  #   $6 : Job index.
  local mode=$1
  local device=$2
  local device_model=$3
  local prec=$4
  local B=$5
  local idx=$6
  if [ "${device}" == "cuda" ]; then
    local dev_dir=${device}-${prec}
  else
    local dev_dir=${device}
  fi
  if [ "${mode}" == "serial" ]; then
    output_dir=${OUTPUTDIR}/${mode}/${dev_dir}/${device_model}
  elif [ "$mode" == "hfta" ]; then
    output_dir=${OUTPUTDIR}/${mode}/${dev_dir}/${device_model}/B${B}
  else
    output_dir=${OUTPUTDIR}/${mode}/${dev_dir}/${device_model}/B${B}/idx${idx}
  fi
  mkdir -p ${output_dir}
  rm -rf ${output_dir}/*
}

dcgm_start() {
  # DCGM start marker.
  # Inputs:
  #   $1 : "serial", "concurrent", "concurrent_mps" or "hfta".
  #   $2 : "cpu", "cuda", "xla".
  #   $3 : "v100", "v3"
  #   $4 : "fp32", "amp"
  #   $5 : Number of jobs.
  local mode=$1
  local device=$2
  local device_model=$3
  local prec=$4
  local B=$5
  if [ "${device}" == "cuda" ]; then
    local dev_dir=${device}-${prec}
    if [ "${mode}" == "serial" ]; then
      local dcgm_output_dir=${OUTPUTDIR}/${mode}/${dev_dir}/${device_model}/dcgm_metrics
    else
      local dcgm_output_dir=${OUTPUTDIR}/${mode}/${dev_dir}/${device_model}/B${B}/dcgm_metrics
    fi
    mkdir -p ${dcgm_output_dir}
    rm -rf ${dcgm_output_dir}/*
    nv-hostengine
    if [ "${device_model}" == "v100" ] || [ "${device_model}" == "a100" ]; then
      local fields=( \
        "DCGM_FI_DEV_GPU_UTIL" \
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE" \
        "DCGM_FI_PROF_PIPE_FP16_ACTIVE" \
        "DCGM_FI_PROF_PIPE_FP32_ACTIVE" \
        "DCGM_FI_PROF_PIPE_FP64_ACTIVE" \
        "DCGM_FI_PROF_SM_OCCUPANCY" \
        "DCGM_FI_PROF_SM_ACTIVE" \
        "DCGM_FI_DEV_FB_USED" \
        "DCGM_FI_PROF_PCIE_RX_BYTES" \
        "DCGM_FI_PROF_PCIE_TX_BYTES" \
        "DCGM_FI_DEV_MEM_COPY_UTIL" \
      )
      echo "Dcgm running on full mode."
    else
      local fields=( \
        "DCGM_FI_DEV_GPU_UTIL" \
        "DCGM_FI_DEV_FB_USED" \
        "DCGM_FI_DEV_MEM_COPY_UTIL" \
      )
    fi
    ${PYTHON2} ../dcgm_monitor.py \
      --output-dir ${dcgm_output_dir} \
      --fields ${fields[@]} &
    dcgm_monitor_pid=$!
    echo "DCGM Monitor started!"
  fi
}

dcgm_stop() {
  # DCGM stop marker.
  # Inputs:
  #   $1 : "cpu", "cuda", "xla".
  local device=$1
  if [ "${device}" == "cuda" ]; then
    kill ${dcgm_monitor_pid}
    wait ${dcgm_monitor_pid}
    unset dcgm_monitor_pid
    echo "DCGM Monitor stopped!"
    nv-hostengine -t
  fi
}

get_amp_flags() {
  # $1: "cpu", "cuda", "xla"
  local device=$1
  if [ "${device}" == "cuda" ]; then
    amp_flags=(" " "--amp")
  else
    amp_flags=(" ")
  fi
}


MIG_PROFILE_CONFIGS=('0' '9,9' '14,14,14' '14,14,14,19' '14,14,19,19,19' '14,19,19,19,19,19' '19,19,19,19,19,19,19')


create_mig_instances() {
  # $1: B
  # distroy existing instances
  distroy_mig_instances
  B_idx=$(( B-1 ))
  $SUDO nvidia-smi mig -cgi ${MIG_PROFILE_CONFIGS[${B_idx}]}
  $SUDO nvidia-smi mig -cci
  # This depends on the output format from nvidia-smi -L, if it breaks, check if the output has changed
  ID_STRING=$(nvidia-smi -L |  grep MIG | awk '{print $6}' | tr -d '()')
  mig_dev_ids=($ID_STRING)

}


distroy_mig_instances() {
    $SUDO nvidia-smi mig  -dci -i 0
    $SUDO nvidia-smi mig  -dgi -i 0
}

serial() {
  # $1: amp flags, "--fp32" or "--amp" or "ALL"
  # $2: "cpu", "cuda", 'xla'
  # $3: "v100", "v3", "a100"
  # $4: epochs
  # $5: iterations per epochs
  # $6: dataroot

  local amp_flags=${1:-"ALL"}
  local device=${2:-"${DEVICE}"}
  local device_model=${3:-"${DEVICE_MODEL}"}
  local epochs=${4:-"${EPOCHS}"}
  local iters_per_epoch=${5:-"${ITERS_PER_EPOCH}"}
  local dataroot=${6:-"${DATAROOT}"}

  if [ "${amp_flags}" == "ALL" ]; then
    get_amp_flags ${device}
  elif [ "${amp_flags}" == "--fp32" ]; then
    amp_flags=(" ")
  else
    amp_flags=(${amp_flags})
  fi
  echo "Running serial ..."
  echo "amp_flags: ${amp_flags[@]}"
  local amp_flag
  for amp_flag in "${amp_flags[@]}"; do
    echo "  amp_flag=${amp_flag}"
    local output_dir
    if [ "${amp_flag}" == "--amp" ]; then
      export NVIDIA_TF32_OVERRIDE=1
      create_output_dir "serial" ${device} ${device_model} amp 0 0
      dcgm_start "serial" ${device} ${device_model} amp 0
    else
      export NVIDIA_TF32_OVERRIDE=0
      create_output_dir "serial" ${device} ${device_model} fp32 0 0
      dcgm_start "serial" ${device} ${device_model} fp32 0
    fi
    ${PYTHON} main.py \
      --dataset lsun \
      --dataroot ${dataroot} \
      --iters-per-epoch ${iters_per_epoch} \
      --epochs ${epochs} \
      --device ${device} \
      --lr 0.0002 \
      ${amp_flag} \
      --outf ${output_dir} \
      > ${output_dir}/out.txt
    dcgm_stop ${device}
    sleep 10
  done
}

check_concurrent_device() {
  # $1: "cpu", "cuda", "xla"
  local device=$1
  if [ "${device}" == "xla" ]; then
    echo "xla not supported in concurrent!"
    return 1
  fi
}

concurrent() {
  # $1: B_start
  # $2: amp flags, "--fp32" or "--amp" or "ALL"
  # $3: "cpu", "cuda", 'xla'
  # $4: "v100", "v3", "a100"
  # $6: epochs
  # $6: iterations per epochs
  # $7: dataroot
  # $8: mode
  local B_start=${1:-1}
  local amp_flags=${2:-"ALL"}
  local device=${3:-"${DEVICE}"}
  local device_model=${4:-"${DEVICE_MODEL}"}
  local epochs=${5:-"${EPOCHS}"}
  local iters_per_epoch=${6:-"${ITERS_PER_EPOCH}"}
  local dataroot=${7:-"${DATAROOT}"}
  local mode=${8:-"concurrent"}

  if [ "${amp_flags}" == "ALL" ]; then
    get_amp_flags ${device}
  elif [ "${amp_flags}" == "--fp32" ]; then
    amp_flags=(" ")
  else
    amp_flags=(${amp_flags})
  fi

  echo "Running ${mode} ..."
  echo "B_start: ${B_start}"
  echo "amp_flags: ${amp_flags[@]}"
  check_concurrent_device ${device}
  local amp_flag
  for amp_flag in "${amp_flags[@]}"; do
    echo "  amp_flag=${amp_flag}"
    local B
    for ((B=${B_start}; B<"${MAX_B}"; B++)); do
      if [ ${B} -ge 12 ] && [ "${device_model}" == "a100" ]; then
        local iters_per_epoch=100
      fi
      echo "    B=${B}"
      echo "    iters_per_epoch=${iters_per_epoch}"
      local lrs
      generate_lrs ${B}
      if [ "${amp_flag}" == "--amp" ]; then
        export NVIDIA_TF32_OVERRIDE=1
        dcgm_start ${mode} ${device} ${device_model} amp ${B}
      else
        export NVIDIA_TF32_OVERRIDE=0
        dcgm_start ${mode} ${device} ${device_model} fp32 ${B}
      fi
      local jobs_status="Success"
      local pids=()
      local idx
      for ((idx=0; idx<${B}; idx++)); do
        echo "      idx=${idx}"
        local output_dir
        if [ "${amp_flag}" == "--amp" ]; then
          create_output_dir ${mode} ${device} ${device_model} amp ${B} ${idx}
        else
          create_output_dir ${mode} ${device} ${device_model} fp32 ${B} ${idx}
        fi
        local lr=${lrs[${idx}]}
        ${PYTHON} main.py \
          --dataset lsun \
          --dataroot ${dataroot} \
          --iters-per-epoch ${iters_per_epoch} \
          --epochs ${epochs} \
          --device ${device} \
          --lr ${lr} \
          ${amp_flag} \
          --outf ${output_dir} \
          > ${output_dir}/out.txt &
        pids+=($!)
      done
      echo "pids = ${pids[@]}"
      local pid
      for pid in ${pids[@]}; do
        if ! wait $pid; then
          jobs_status="Fail"
          echo "pid=${pid} failed!"
        fi
      done
      dcgm_stop ${device}
      sleep 10
      if [ "${jobs_status}" == "Fail" ]; then
        echo "Encounter OOM at B=${B}!"
        rm -rf "$(dirname "${output_dir}")"
        break
      fi
      if [ ${B} -ge 8 ] && [ "${device_model}" == "a100" ]; then
        B=${B}+1
      fi
    done
  done
}

mig() {
  # $1: "cuda"
  # $2: "a100"
  # $3: epochs
  # $4: iterations per epochs
  # $5: dataroot
  # $6: mode - "mig"
  local device=${1:-"${DEVICE}"}
  local device_model=${2:-"${DEVICE_MODEL}"}
  local epochs=${3:-"${EPOCHS}"}
  local iters_per_epoch=${4:-"${ITERS_PER_EPOCH}"}
  local dataroot=${5:-"${DATAROOT}"}
  local mode=${6:-"mig"}

  echo "Running ${mode} ..."
  local MAX_B_MIG=${MAX_B}
  if [ ${MAX_B_MIG} -gt 8 ]; then
    MAX_B_MIG=8
    echo "Setting MAX_B to 8 for MIG"
  fi
  check_concurrent_device ${device}
  local amp_flags
  get_amp_flags ${device}
  local amp_flag
  for amp_flag in "${amp_flags[@]}"; do
    echo "  amp_flag=${amp_flag}"
    local B
    for ((B=1; B<"${MAX_B_MIG}"; B++)); do
      echo "    B=${B}"
      local lrs
      generate_lrs ${B}
      if [ "${amp_flag}" == "--amp" ]; then
        dcgm_start ${mode} ${device} ${device_model} amp ${B}
      else
        dcgm_start ${mode} ${device} ${device_model} fp32 ${B}
      fi

      # create mig instances based on the number of concurrent array jobs
      local mig_dev_ids
      create_mig_instances ${B}

      local jobs_status="Success"
      local pids=()
      local idx
      for ((idx=0; idx<${B}; idx++)); do
        echo "      idx=${idx}"
        local output_dir
        if [ "${amp_flag}" == "--amp" ]; then
          create_output_dir ${mode} ${device} ${device_model} amp ${B} ${idx}
        else
          create_output_dir ${mode} ${device} ${device_model} fp32 ${B} ${idx}
        fi

        local MIG_DEVICE=${mig_dev_ids[${idx}]}
        echo "Setting CUDA_VISIBLE_DEVICES=${mig_dev_ids[${idx}]}"

        local lr=${lrs[${idx}]}
        CUDA_VISIBLE_DEVICES=$MIG_DEVICE ${PYTHON} main.py \
          --dataset lsun \
          --dataroot ${dataroot} \
          --iters-per-epoch ${iters_per_epoch} \
          --epochs ${epochs} \
          --device ${device} \
          --lr ${lr} \
          ${amp_flag} \
          --outf ${output_dir} \
          > ${output_dir}/out.txt &
        pids+=($!)
      done
      echo "pids = ${pids[@]}"
      local pid
      for pid in ${pids[@]}; do
        if ! wait $pid; then
          jobs_status="Fail"
          echo "pid=${pid} failed!"
        fi
      done
      dcgm_stop ${device}

      sleep 10

      if [ "${jobs_status}" == "Fail" ]; then
        echo "Encounter OOM at B=${B}!"
        rm -rf "$(dirname "${output_dir}")"
        break
      fi
    done
  done
}

concurrent_mps() {
  # $1: B_start
  # $2: amp flags, "--fp32" or "--amp" or "ALL"
  # $3: "cpu", "cuda", 'xla'
  # $4: "v100", "v3", "a100"
  # $6: epochs
  # $6: iterations per epochs
  # $7: dataroot
  local B_start=${1:-1}
  local amp_flags=${2:-"ALL"}
  local device=${3:-"${DEVICE}"}
  local device_model=${4:-"${DEVICE_MODEL}"}
  local epochs=${5:-"${EPOCHS}"}
  local iters_per_epoch=${6:-"${ITERS_PER_EPOCH}"}
  local dataroot=${7:-"${DATAROOT}"}

  ${SUDO} nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
  export CUDA_VISIBLE_DEVICES=0
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
  nvidia-cuda-mps-control -d &

  concurrent ${B_start} ${amp_flags} ${device} ${device_model} ${epochs} \
    ${iters_per_epoch} ${dataroot} "concurrent_mps"

  ${SUDO} echo quit | nvidia-cuda-mps-control
  ${SUDO} nvidia-smi -i 0 -c 0
  unset CUDA_VISIBLE_DEVICES
  unset CUDA_MPS_PIPE_DIRECTORY
  unset CUDA_MPS_LOG_DIRECTORY
}

hfta() {
  # $1: B_start
  # $2: amp flags, "--fp32" or "--amp" or "ALL"
  # $3: "cpu", "cuda", 'xla'
  # $4: "v100", "v3", "a100"
  # $6: epochs
  # $6: iterations per epochs
  # $7: dataroot
  local B_start=${1:-1}
  local amp_flags=${2:-"ALL"}
  local device=${3:-"${DEVICE}"}
  local device_model=${4:-"${DEVICE_MODEL}"}
  local epochs=${5:-"${EPOCHS}"}
  local iters_per_epoch=${6:-"${ITERS_PER_EPOCH}"}
  local dataroot=${7:-"${DATAROOT}"}
  if [ "${amp_flags}" == "ALL" ]; then
    get_amp_flags ${device}
  elif [ "${amp_flags}" == "--fp32" ]; then
    amp_flags=(" ")
  else
    amp_flags=(${amp_flags})
  fi

  echo "Running hfta ..."
  echo "amp_flags: ${amp_flags[@]}"
  local amp_flag
  for amp_flag in "${amp_flags[@]}"; do
    echo "  amp_flag=${amp_flag}"
    local B
    for ((B=${B_start}; B<${MAX_B}; B++)); do
      if [ ${B} -ge 100 ] && [ "${device_model}" == "a100" ]; then
        local iters_per_epoch=100
      fi
      echo "    B=${B}"
      echo "    iters_per_epoch=${iters_per_epoch}"
      local lrs
      generate_lrs ${B}
      local output_dir
      if [ "${amp_flag}" == "--amp" ]; then
        export NVIDIA_TF32_OVERRIDE=1
        create_output_dir "hfta" ${device} ${device_model} amp $B 0
        dcgm_start "hfta" ${device} ${device_model} amp $B
      else
        export NVIDIA_TF32_OVERRIDE=0
        create_output_dir "hfta" ${device} ${device_model} fp32 $B 0
        dcgm_start "hfta" ${device} ${device_model} fp32 $B
      fi
      ${PYTHON} main.py \
        --dataset lsun \
        --dataroot ${dataroot} \
        --iters-per-epoch ${iters_per_epoch} \
        --epochs ${epochs} \
        --device ${device} \
        --lr ${lrs[@]} \
        ${amp_flag} \
        --outf ${output_dir} \
        --hfta \
        > ${output_dir}/out.txt &
      local pid=$!
      local jobs_status="Success"
      if ! wait $pid; then
        jobs_status="Fail"
        echo "pid=${pid} failed!"
      fi
      dcgm_stop ${device}
      sleep 10
      if [ "${jobs_status}" == "Fail" ]; then
        echo "Encounter OOM at B=${B}!"
        rm -rf ${output_dir}
        break
      fi
      if [ ${B} -ge 13 ] && [ "${device_model}" == "a100" ]; then
        B=${B}+11
      fi
    done
  done
}
