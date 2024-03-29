#!/bin/bash

# Builds ArrayRecord from source code located in SOURCE_DIR producing wheels
# under $SOURCE_DIR/all_dist.
function build_and_test_array_record() {
  SOURCE_DIR=$1

  # Automatically decide which platform to build for by checking on which
  # platform this runs.
  ARCH=$(uname -m)
  AUDITWHEEL_PLATFORM="manylinux2014_${ARCH}"

  # Using a previous version of Blaze to avoid:
  # https://github.com/bazelbuild/bazel/issues/8622
  export BAZEL_VERSION="5.4.0"

  # Build wheels for multiple Python minor versions.
  PYTHON_MAJOR_VERSION=3
  for PYTHON_MINOR_VERSION in 9 10 11 12
  do
    PYTHON_VERSION=${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}
    PYTHON_BIN=/opt/python/cp${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-cp${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}/bin

    # Cleanup older images.
    docker rmi -f array_record:${PYTHON_VERSION}
    docker rm -f array_record

    DOCKER_BUILDKIT=1 docker build --progress=plain --no-cache \
      --build-arg ARCH=${ARCH} \
      --build-arg AUDITWHEEL_PLATFORM=${AUDITWHEEL_PLATFORM} \
      --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
      --build-arg PYTHON_BIN=${PYTHON_BIN} \
      --build-arg BAZEL_VERSION=${BAZEL_VERSION} \
      -t array_record:${PYTHON_VERSION} - < ${SOURCE_DIR}/oss/build.Dockerfile

    docker run --rm -a stdin -a stdout -a stderr \
      --env PYTHON_BIN="${PYTHON_BIN}/python" \
      --env BAZEL_VERSION=${BAZEL_VERSION} \
      --env AUDITWHEEL_PLATFORM=${AUDITWHEEL_PLATFORM} \
      -v $SOURCE_DIR:/tmp/array_record \
      --name array_record array_record:${PYTHON_VERSION} \
      bash oss/build_whl.sh
  done

  ls ${SOURCE_DIR}/all_dist/*.whl
}