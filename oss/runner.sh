#!/bin/bash
# This script copies array_record from internal repo, builds a docker, and
# builds pip wheels for all Python versions.

set -e -x

export TMP_FOLDER="/tmp/array_record"

# Clean previous folders/images.
[ -f $TMP_FOLDER ] && rm -rf $TMP_FOLDER
for PYTHON_VERSION in 3.8 3.9 3.10
do
  docker rmi -f array_record:${PYTHON_VERSION}
done

# Synchronize Copybara in $TMP_FOLDER.
copybara array_record/oss/copy.bara.sky g3folder_to_gitfolder ../../ \
  --init-history --folder-dir=$TMP_FOLDER --ignore-noop

cd $TMP_FOLDER

# Build wheel for each Python version.
for PYTHON_VERSION in 3.8 3.9 3.10
do
  DOCKER_BUILDKIT=1 docker build --progress=plain --no-cache \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
    -t array_record:${PYTHON_VERSION} - < oss/build.Dockerfile

  docker run --rm -a stdin -a stdout -a stderr \
    --env PYTHON_VERSION=${PYTHON_VERSION} \
    -v $TMP_FOLDER:/tmp/array_record \
    --name array_record array_record:${PYTHON_VERSION} \
    bash oss/build_whl.sh
done

ls $TMP_FOLDER/all_dist/*.whl
