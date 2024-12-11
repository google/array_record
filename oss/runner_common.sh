#!/bin/bash

# Builds ArrayRecord from source code located in SOURCE_DIR producing wheels
# under $SOURCE_DIR/all_dist.
function build_and_test_array_record_linux() {
  SOURCE_DIR=$1

  # Automatically decide which platform to build for by checking on which
  # platform this runs.
  AUDITWHEEL_PLATFORM="manylinux2014_$(uname -m)"

  # Using a previous version of Blaze to avoid:
  # https://github.com/bazelbuild/bazel/issues/8622
  export BAZEL_VERSION="5.4.0"

  # Build wheels for multiple Python minor versions.
  PYTHON_MAJOR_VERSION=3
  for PYTHON_MINOR_VERSION in 10 11 12
  do
    PYTHON_VERSION=${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}
    PYTHON_BIN=/opt/python/cp${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-cp${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}/bin

    # Cleanup older images.
    docker rmi -f array_record:${PYTHON_VERSION}
    docker rm -f array_record

    DOCKER_BUILDKIT=1 docker build --progress=plain --no-cache \
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

function install_and_init_pyenv {
  pyenv_root=${1:-$HOME/.pyenv}
  export PYENV_ROOT=$pyenv_root
  if [[ ! -d $PYENV_ROOT ]]; then
    echo "Installing pyenv.."
    git clone https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
    pushd "$PYENV_ROOT"
    git checkout "v2.4.21"
    popd
    export PATH="/home/kbuilder/.local/bin:$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
  fi

  echo "Python setup..."
  pyenv install -s "$PYENV_PYTHON_VERSION"
  pyenv global "$PYENV_PYTHON_VERSION"
  export PYTHON_BIN=$(pyenv which python)
}

function setup_env_vars_py {
  # This controls the python binary to use.
  PYTHON_MAJOR_VERSION=$1
  PYTHON_MINOR_VERSION=$2
  # This is for pyenv install.
  PYENV_PYTHON_VERSION=${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}
  PYTHON="python$PYENV_PYTHON_VERSION"
}

function update_bazel_macos {
  BAZEL_VERSION=$1
  ARCH="$(uname -m)"
  curl -L https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-darwin-${ARCH}.sh -O
  ls
  chmod +x bazel-*.sh
  ./bazel-${BAZEL_VERSION}-installer-darwin-${ARCH}.sh --user
  rm -f ./bazel-${BAZEL_VERSION}-installer-darwin-${ARCH}.sh
  # Add new bazel installation to path
  export PATH="/Users/kbuilder/bin:$PATH"
}

function install_ar_deps {
  $PYTHON_BIN -m pip install -U \
    absl-py \
    build \
    etils[epath] \
    setuptools \
    twine \
    wheel;
}

function build_and_test_array_record_macos() {
  SOURCE_DIR=$1
  # Set up Bazel.
  # Using a previous version of Bazel to avoid:
  # https://github.com/bazelbuild/bazel/issues/8622
  export BAZEL_VERSION="5.4.0"
  update_bazel_macos ${BAZEL_VERSION}
  bazel --version

  PYTHON_MAJOR_VERSION=3
  for PYTHON_MINOR_VERSION in 10 11 12
  do
    # Set up Pyenv.
    PYTHON_VERSION=${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}
    echo "Creating array_record wheel for Python Version $PYTHON_VERSION"
    setup_env_vars_py $PYTHON_MAJOR_VERSION $PYTHON_MINOR_VERSION
    install_and_init_pyenv
    install_ar_deps

    # Build and test ArrayRecord.
    bash ${SOURCE_DIR}/oss/build_whl.sh
  done

  ls ${SOURCE_DIR}/all_dist/*.whl
}