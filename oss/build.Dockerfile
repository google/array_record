# Constructs the environment within which we will build the pip wheels.


ARG AUDITWHEEL_PLATFORM

FROM quay.io/pypa/${AUDITWHEEL_PLATFORM}

ARG ARCH
ARG PYTHON_VERSION
ARG PYTHON_BIN
ARG BAZEL_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN yum install -y rsync
ENV PATH="${PYTHON_BIN}:${PATH}"

# Install bazel.
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux ${ARCH}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux ${ARCH}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-$ARCH.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-$ARCH.sh

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