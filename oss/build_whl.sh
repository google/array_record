#!/bin/bash

# Build wheel for the python version specified by $PYTHON_VERSION.
# Optionally, can set the environment variable $PYTHON_BIN to refer to a
# specific python interpreter.

set -e -x

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/array_record}"

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function main() {
  # Remove .bazelrc if it already exists
  [ -e .bazelrc ] && rm .bazelrc

  write_to_bazelrc "build --incompatible_default_to_explicit_init_py"
  write_to_bazelrc "build --enable_platform_specific_config"
  write_to_bazelrc "build --@rules_python//python/config_settings:python_version=${PYTHON_VERSION}"
  write_to_bazelrc "test --@rules_python//python/config_settings:python_version=${PYTHON_VERSION}"
  write_to_bazelrc "test --action_env PYTHON_VERSION=${PYTHON_VERSION}"
  write_to_bazelrc "test --test_timeout=300"

  write_to_bazelrc "build -c opt"
  write_to_bazelrc "build --cxxopt=-std=c++17"
  write_to_bazelrc "build --host_cxxopt=-std=c++17"
  write_to_bazelrc "build --experimental_repo_remote_exec"
  write_to_bazelrc "common --check_direct_dependencies=error"
  PLATFORM="$(uname)"

  if [ -n "${CROSSTOOL_TOP}" ]; then
    write_to_bazelrc "build --crosstool_top=${CROSSTOOL_TOP}"
    write_to_bazelrc "test --crosstool_top=${CROSSTOOL_TOP}"
  fi

  export USE_BAZEL_VERSION="${BAZEL_VERSION}"
  bazel clean
  bazel build ... --action_env MACOSX_DEPLOYMENT_TARGET='11.0' --action_env PYTHON_BIN_PATH="${PYTHON_BIN}"
  bazel test --verbose_failures --test_output=errors ... --action_env PYTHON_BIN_PATH="${PYTHON_BIN}"

  DEST="/tmp/array_record/all_dist"
  # Create the directory, then do dirname on a non-existent file inside it to
  # give us an absolute paths with tilde characters resolved to the destination
  # directory.
  mkdir -p "${DEST}"
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"
  mkdir "${TMPDIR}/array_record"

  echo $(date) : "=== Copy array_record files"

  cp setup.py "${TMPDIR}"
  cp LICENSE "${TMPDIR}"
  rsync -avm -L  --exclude="bazel-*/" . "${TMPDIR}/array_record"
  rsync -avm -L  --include="*.so" --include="*_pb2.py" \
    --exclude="*.runfiles" --exclude="*_obj" --include="*/" --exclude="*" \
    bazel-bin/cpp "${TMPDIR}/array_record"
  rsync -avm -L  --include="*.so" --include="*_pb2.py" \
    --exclude="*.runfiles" --exclude="*_obj" --include="*/" --exclude="*" \
    bazel-bin/python "${TMPDIR}/array_record"

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"
  if [ "$(uname)" = "Darwin" ]; then
    "$PYTHON_BIN" setup.py bdist_wheel --python-tag py3"${PYTHON_MINOR_VERSION}" --plat-name macosx_11_0_"$(uname -m)"
  else
    "$PYTHON_BIN" setup.py bdist_wheel --python-tag py3"${PYTHON_MINOR_VERSION}"
  fi

  if [ -n "${AUDITWHEEL_PLATFORM}" ]; then
    echo $(date) : "=== Auditing wheel"
    auditwheel repair --plat ${AUDITWHEEL_PLATFORM} -w dist dist/*.whl
    cp dist/*manylinux*.whl "${DEST}"
  else
    cp dist/*.whl "${DEST}"
  fi

  echo $(date) : "=== Listing wheel"
  ls -lrt "${DEST}"/*.whl
  popd

  echo $(date) : "=== Output wheel file is in: ${DEST}"

  # Install ArrayRecord from the wheel and run smoke tests.
  $PYTHON_BIN -m pip install --find-links="${DEST}" --pre array-record
  $PYTHON_BIN -c 'import array_record'
  $PYTHON_BIN -c 'from array_record.python import array_record_data_source'
  # TF is not available on Python 3.13 and above.
  if [ "$(uname)" != "Darwin" ] && (( "${PYTHON_MINOR_VERSION}" < 13 )); then
    $PYTHON_BIN -m pip install jax tensorflow grain
    $PYTHON_BIN oss/test_with_grain.py
    $PYTHON_BIN oss/test_with_tf.py
  fi
}

main
