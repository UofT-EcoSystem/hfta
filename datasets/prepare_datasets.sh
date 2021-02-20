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

_download_wikitext2() {
  if [ ! -d "datasets/wikitext-2" ]; then
    local url=https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
    local zip_path=datasets/wikitext-2-v1.zip
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
    --warmup-data-loading \
    --device cpu
}

prepare_pointnet_seg() {
  _download_shapenet
  python examples/pointnet/train_segmentation.py \
    --epochs 1 \
    --iters-per-epoch 1000 \
    --dataset datasets/shapenetcore_partanno_segmentation_benchmark_v0/ \
    --warmup-data-loading \
    --device cpu
}

_check_md5(){
  local md5_file=${1-"./DCGAN_Lsun_Data.md5"}
  md5sum --status -c ${md5_file} > /dev/null
  return $?
}

prepare_dcgan() {
  if _check_md5 "./datasets/DCGAN_Lsun_Data.md5"; then
    echo "Lsun dataset have been downloaded!"
    return 0
  fi

  if ! _check_md5 "./datasets/DCGAN_Download.md5"; then
    echo "Start download Lsun dataset!"
    cd datasets
    rm -rf ./lsun
    git clone https://github.com/fyu/lsun.git
    cd lsun
    python download.py -c bedroom
  else
    echo "Find zip file of  Lsun dataset!"
    cd datasets/lsun
  fi
  echo "Extracting Lsun dataset!"
  unzip bedroom_train_lmdb.zip
  unzip bedroom_val_lmdb.zip
  cd ../../
  echo "Lsun dataset have been downloaded and setup!"
}

prepare_bert() {
  _download_wikitext2
}

prepare_transformer() {
  _download_wikitext2
}

prepare_cifar10() {
  if [ ! -d "datasets/cifar-10-batches-py" ]; then
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xf cifar-10-python.tar.gz
    mv cifar-10-batches-py cifar10
  fi
}