#!/bin/bash

set -e

DATASETS=${1:-"${HOME}/datasets"}
URL=${2:-"hfta:dev"}

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
  ${URL} \
  /bin/bash
