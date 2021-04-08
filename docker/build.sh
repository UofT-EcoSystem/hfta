#!/bin/bash

set -e

VERSION=${1:-"native1.6-cu10.2"}
PUSH=${2-"none"}
URL=${3:-"hfta:dev"}

docker image build -f docker/${VERSION}.Dockerfile -t ${URL} .
if [ "${PUSH}" == "push" ]; then
  docker push ${URL}
elif [ "${PUSH}" == "none" ]; then
  echo "Keep image ${URL} locally."
else
  echo "Invalid \${PUSH} argument: ${PUSH} !"
  return 1
fi
