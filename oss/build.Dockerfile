# Constructs the environment within which we will build the array_record pip wheels.
#
# From /tmp/array_record,
# ❯ DOCKER_BUILDKIT=1 docker build -t array_record:latest - < oss/build.Dockerfile
# ❯ docker run --rm -it -v /tmp/array_recor:/tmp/array_reco \
#      array_record:latest bash

ARG base_image="tensorflow/build:2.10-python3.9"
FROM $base_image
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
    libffi-dev \
    neovim

# 3.9 is the built-in interpreter version in this image.
RUN for v in 3.8.15; do \
    wget "https://www.python.org/ftp/python/$v/Python-${v}.tar.xz" && \
    rm -rf "/tmp/python${v}" && mkdir -p "/tmp/python${v}" && \
    tar xvf "Python-${v}.tar.xz" -C "/tmp/python${v}" && \
    cd "/tmp/python${v}/Python-${v}" && \
    ./configure 2>&1 >/dev/null && \
    make -j8 altinstall 2>&1 >/dev/null && \
    ln -sf "/usr/local/bin/python${v%.*}" "/usr/bin/python${v%.*}"; \
  done

# For each python interpreter, install pip dependencies needed for array_record
RUN --mount=type=cache,target=/root/.cache \
  for p in 3.8 3.9; do \
    python${p} -m pip install -U pip && \
    python${p} -m pip install -U \
      absl-py \
      auditwheel \
      patchelf \
      setuptools \
      twine \
      wheel; \
  done

WORKDIR "/tmp/array_record"
