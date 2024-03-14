#!/bin/bash
# This script builds a docker container, and pip wheels for all supported
# Python versions.  Run from the root array_record directory:
#
# ./oss/docker_runner.sh

set -e -x

# Clean previous images.
for PYTHON_VERSION in 3.9 3.10 3.11 3.12
do
  docker rmi -f array_record:${PYTHON_VERSION}
done

ARCH=$(uname -m)
if [ "$ARCH" == "x86_64" ]; then
  CROSSTOOL_TOP="@sigbuild-r2.12-python${PYTHON_VERSION}_config_cuda//crosstool:toolchain"
  AUDITWHEEL_PLATFORM="manylinux2014_x86_64"
elif [ "$ARCH" == "aarch64" ]; then
  CROSSTOOL_TOP="@ml2014_aarch64_config_aarch64//crosstool:toolchain"
  AUDITWHEEL_PLATFORM="manylinux2014_aarch64"
fi

# Build wheel for each Python version.
for PYTHON_VERSION in 3.9 3.10 3.11 3.12
do
  DOCKER_BUILDKIT=1 docker build --progress=plain --no-cache \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
    -t array_record:${PYTHON_VERSION} - < "oss/build.Dockerfile.${ARCH}"

  docker run --rm -a stdin -a stdout -a stderr \
    --env PYTHON_VERSION=${PYTHON_VERSION} \
    --env CROSSTOOL_TOP=${CROSSTOOL_TOP} \
    --env AUDITWHEEL_PLATFORM=${AUDITWHEEL_PLATFORM} \
    -v $PWD:/tmp/array_record \
    --name array_record array_record:${PYTHON_VERSION} \
    bash oss/build_whl.sh
done

ls $PWD/all_dist/*.whl

