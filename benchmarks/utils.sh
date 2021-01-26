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
