TPU_IP_ADDRESS=$1
DATASETS=${2:-"${HOME}/datasets"}

docker run \
  -it \
  --rm \
  --shm-size 16G \
  -e XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470" \
  -v $(pwd):$(pwd) \
  -v ${DATASETS}:${DATASETS} \
  -w $(pwd) \
  gcr.io/tpu-pytorch/xla:r1.7 \
  /bin/bash
