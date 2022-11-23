#!/bin/bash
# This script copy array_record from internal repo, build a docker, and build pip wheels

set -e -x

export TMP_FOLDER="/tmp/array_record"

[ -f $TMP_FOLDER ] && rm -rf $TMP_FOLDER
copybara array_record/oss/copy.bara.sky g3folder_to_gitfolder ../../ \
  --init-history --folder-dir=$TMP_FOLDER --ignore-noop

cd $TMP_FOLDER
DOCKER_BUILDKIT=1 docker build --progress=plain --no-cache \
  -t array_record:latest - < oss/build.Dockerfile

docker run --rm -a stdin -a stdout -a stderr \
  -v $TMP_FOLDER:/tmp/array_record --name array_record array_record:latest \
  bash oss/build_whl_runner.sh

ls $TMP_FOLDER/*.whl