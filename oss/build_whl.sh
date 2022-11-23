#!/bin/bash
# build wheel for python version specified in $PYTHON

set -e -x

export PYTHON_MINOR_VERSION="${PYTHON_MINOR_VERSION}"
PYTHON="python3${PYTHON_MINOR_VERSION:+.$PYTHON_MINOR_VERSION}"

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function main() {
  # Remove .bazelrc if it already exists
  [ -e .bazelrc ] && rm .bazelrc

  write_to_bazelrc "build -c opt"
  write_to_bazelrc "build --cxxopt=-std=c++17"
  write_to_bazelrc "build --host_cxxopt=-std=c++17"
  write_to_bazelrc "build --linkopt=\"-lrt -lm\""
  write_to_bazelrc "build --experimental_repo_remote_exec"
  write_to_bazelrc "build --action_env=PYTHON_BIN_PATH=\"/usr/bin/$PYTHON\""
  write_to_bazelrc "build --action_env=PYTHON_LIB_PATH=\"/usr/lib/$PYTHON\""
  write_to_bazelrc "build --python_path=\"/usr/bin/$PYTHON\""

  bazel clean
  bazel build $@ ...
  bazel test $@ ...

  DEST="/tmp/array_record_pip_pkg"
  # Create the directory, then do dirname on a non-existent file inside it to
  # give us an absolute paths with tilde characters resolved to the destination
  # directory.
  mkdir -p "${DEST}"
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"
  mkdir "${TMPDIR}/array_record"

  echo "=== Copy array_record files"

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
  ${PYTHON} setup.py bdist_wheel --python-tag py3${PYTHON_MINOR_VERSION}

  echo $(date) : "=== Auditing wheel"
  auditwheel repair --plat manylinux2014_x86_64 -w dist dist/*.whl
  cp dist/*.whl "${DEST}"
  popd

  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
