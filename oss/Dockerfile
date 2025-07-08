# Constructs the environment within which we will build the pip wheels.


ARG AUDITWHEEL_PLATFORM

FROM quay.io/pypa/${AUDITWHEEL_PLATFORM}

ARG PYTHON_VERSION
ARG BAZEL_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN ulimit -n 1024 && yum install -y rsync
ENV PYTHON_BIN_PATH=/opt/python/cp${PYTHON_VERSION}-cp${PYTHON_VERSION}/bin
ENV PATH="${PYTHON_BIN_PATH}:${PATH}"

ENV PYTHON_BIN=${PYTHON_BIN_PATH}/python

# Download the correct bazel version and make sure it's on path.
RUN BAZEL_ARCH_SUFFIX="$(uname -m | sed s/aarch64/arm64/)" \
  && curl -sSL --fail -o /usr/local/bin/bazel "https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-$BAZEL_ARCH_SUFFIX" \
  && chmod a+x /usr/local/bin/bazel

# Install dependencies needed for array_record.
RUN --mount=type=cache,target=/root/.cache \
  $PYTHON_BIN -m pip install -U \
    absl-py \
    auditwheel \
    etils[epath] \
    patchelf \
    setuptools \
    twine \
    wheel;

WORKDIR "/tmp/array_record"
