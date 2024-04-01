# Constructs the environment within which we will build the pip wheels.


ARG AUDITWHEEL_PLATFORM

FROM quay.io/pypa/${AUDITWHEEL_PLATFORM}

ARG PYTHON_VERSION
ARG PYTHON_BIN
ARG BAZEL_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN yum install -y rsync
ENV PATH="${PYTHON_BIN}:${PATH}"

# Download the correct bazel version and make sure it's on path.
RUN BAZEL_ARCH_SUFFIX="$(uname -m | sed s/aarch64/arm64/)" \
  && curl -sSL --fail -o /usr/local/bin/bazel "https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-$BAZEL_ARCH_SUFFIX" \
  && chmod a+x /usr/local/bin/bazel

# Install dependencies needed for array_record.
RUN --mount=type=cache,target=/root/.cache \
  ${PYTHON_BIN}/python -m pip install -U \
    absl-py \
    auditwheel \
    etils[epath] \
    patchelf \
    setuptools \
    twine \
    wheel;

WORKDIR "/tmp/array_record"