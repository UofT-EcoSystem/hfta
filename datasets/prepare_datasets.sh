#!/bin/bash

_download_shapenet() {
  if [ ! -d "datasets/shapenetcore_partanno_segmentation_benchmark_v0" ]; then
    local url=https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip
    local zip_path=datasets/shapenetcore_partanno_segmentation_benchmark_v0.zip
    wget ${url} --no-check-certificate -O ${zip_path}
    unzip -q ${zip_path} -d datasets/
    rm -rf ${zip_path}
  fi
}

prepare_pointnet_cls() {
  _download_shapenet
  python examples/pointnet/train_classification.py \
    --epochs 1 \
    --iters-per-epoch 1000 \
    --dataset datasets/shapenetcore_partanno_segmentation_benchmark_v0/ \
    --eval \
    --warmup-data-loading
}

prepare_pointnet_seg() {
  _download_shapenet
  python examples/pointnet/train_segmentation.py \
    --epochs 1 \
    --iters-per-epoch 1000 \
    --dataset datasets/shapenetcore_partanno_segmentation_benchmark_v0/ \
    --warmup-data-loading
}
