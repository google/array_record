#!/bin/bash

set -e -x

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/array_record}"

setup_env_vars_py() {
  # This controls the python binary to use.
  PYTHON_MAJOR_VERSION="$1"
  PYTHON_MINOR_VERSION="$2"
  PYENV_PYTHON_VERSION="${PYTHON_MAJOR_VERSION}"'.'"${PYTHON_MINOR_VERSION}"
  PYTHON='python'"${PYENV_PYTHON_VERSION}"
  export PYTHON
  PYTHON_BIN="$(which python)"
  export PYTHON_BIN
}

# Builds ArrayRecord from source code located in SOURCE_DIR producing wheels
# under $SOURCE_DIR/all_dist.
build_and_test_array_record() {
  printf 'Creating ArrayRecord wheel for Python Version %s\n' "$PYTHON_VERSION"
  case "$(uname)" in
    Darwin*|CYGWIN*|MINGW*|MSYS_NT*)
      setup_env_vars_py "$PYTHON_MAJOR_VERSION" "$PYTHON_MINOR_VERSION"
      "$PYTHON_BIN" -m pip install -U setuptools wheel etils[epath]
      sh "${SOURCE_DIR}"'/oss/build_whl.sh'
      ;;
    *)
      # Automatically decide which platform to build for by checking on which
      # platform this runs.
      AUDITWHEEL_PLATFORM='manylinux2014_'"$(uname -m)"
      docker rmi -f array_record:${PYTHON_VERSION}
      docker rm -f array_record
      DOCKER_BUILDKIT=1 docker build --progress=plain --no-cache \
        --build-arg AUDITWHEEL_PLATFORM="${AUDITWHEEL_PLATFORM}" \
        --build-arg PYTHON_VERSION="${PYTHON_MAJOR_VERSION}""${PYTHON_MINOR_VERSION}" \
        --build-arg BAZEL_VERSION="${BAZEL_VERSION}" \
        -t array_record:"${PYTHON_VERSION}" "${SOURCE_DIR}"'/oss'

      docker run --rm -a stdin -a stdout -a stderr \
        --env PYTHON_VERSION="${PYTHON_MAJOR_VERSION}"'.'"${PYTHON_MINOR_VERSION}" \
        --env PYTHON_MAJOR_VERSION="${PYTHON_MAJOR_VERSION}" \
        --env PYTHON_MINOR_VERSION="${PYTHON_MINOR_VERSION}" \
        --env BAZEL_VERSION="${BAZEL_VERSION}" \
        --env AUDITWHEEL_PLATFORM="${AUDITWHEEL_PLATFORM}" \
        -v "${SOURCE_DIR}":"${OUTPUT_DIR}" \
        --name array_record array_record:"${PYTHON_VERSION}" \
        sh oss/build_whl.sh
      ;;
  esac
}
