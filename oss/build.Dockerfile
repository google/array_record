# Constructs the environment within which we will build the array_record pip wheels.
#
# From /tmp/array_record,
# ❯ DOCKER_BUILDKIT=1 docker build \
#     --build-arg PYTHON_VERSION=3.9 \
#     -t array_record:latest - < oss/build.Dockerfile
# ❯ docker run --rm -it -v /tmp/array_record:/tmp/array_record \
#      array_record:latest bash

ARG PYTHON_VERSION
FROM tensorflow/build:2.12-python${PYTHON_VERSION}
LABEL maintainer="Array_record team <array-record@google.com>"

ENV DEBIAN_FRONTEND=noninteractive

# Install supplementary Python interpreters
RUN mkdir /tmp/python
RUN --mount=type=cache,target=/var/cache/apt \
  apt update && \
  apt install -yqq \
    apt-utils \
    build-essential \
    checkinstall \
    libffi-dev

# Install pip dependencies needed for array_record
RUN --mount=type=cache,target=/root/.cache \
  python${PYTHON_VERSION} -m pip install -U pip && \
  python${PYTHON_VERSION} -m pip install -U \
    absl-py \
    auditwheel \
    etils[epath] \
    patchelf \
    setuptools \
    twine \
    wheel;

WORKDIR "/tmp/array_record"
