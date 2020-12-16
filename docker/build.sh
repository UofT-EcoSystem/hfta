#!/bin/bash

set -e

VERSION=${1:-"native1.6-cu10.2"}
TAG=${2:-"0.1"}

docker image build -f docker/${VERSION}.Dockerfile -t hfta:${TAG} .
