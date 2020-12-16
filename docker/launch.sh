#!/bin/bash

set -e

DATASETS=${1:-"${HOME}/datasets"}
TAG=${2:-"0.1"}

docker run \
  --privileged \
  --cap-add=ALL \
  --gpus all \
  -it \
  --rm \
  --ipc=host \
  -e NVIDIA_MIG_CONFIG_DEVICES="all" \
  -v $(pwd):$(pwd) \
  -v ${DATASETS}:${DATASETS} \
  -w $(pwd) \
  hfta:${TAG} \
  /bin/bash
