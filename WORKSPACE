workspace(name = "array_record")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Abseil LTS 20230125.0
http_archive(
    name = "com_google_absl",
    sha256 = "987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed",
    strip_prefix = "abseil-cpp-20230802.1",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/refs/tags/20230802.1.tar.gz",
    ],
)

# Version: pypi-v0.11.0, 2020/10/27
git_repository(
    name = "com_google_absl_py",
    commit = "127c98870edf5f03395ce9cf886266fa5f24455e",
    remote = "https://github.com/abseil/abseil-py",
    shallow_since = "1673401277 -0800",
)

# Needed by com_google_riegeli
http_archive(
    name = "org_brotli",
    sha256 = "84a9a68ada813a59db94d83ea10c54155f1d34399baf377842ff3ab9b3b3256e",
    strip_prefix = "brotli-3914999fcc1fda92e750ef9190aa6db9bf7bdb07",
    urls = ["https://github.com/google/brotli/archive/3914999fcc1fda92e750ef9190aa6db9bf7bdb07.zip"],  # 2022-11-17
)

# GoogleTest/GoogleMock framework. Used by most unit-tests.
http_archive(
    name = "com_google_googletest",
    sha256 = "24e06e79a78ca5794ec6ad2bf0a1f05515cd1d05a9e10d9a6dc853078b2f3914",
    strip_prefix = "googletest-main",
    urls = ["https://github.com/google/googletest/archive/main.zip"],
)

# V3.4.0, 20210818
http_archive(
    name = "eigen3",
    build_file_content =
        """
cc_library(
    name = 'eigen3',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['Eigen/**', 'unsupported/Eigen/**']),
    visibility = ['//visibility:public'],
)
""",
    sha256 = "b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626",
    strip_prefix = "eigen-3.4.0",
    urls = [
        "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2",
    ],
)

# `pybind11_bazel` (https://github.com/pybind/pybind11_bazel): 20230130
http_archive(
    name = "pybind11_bazel",
    sha256 = "b35f3abc3d52ee5c753fdeeb2b5129b99e796558754ca5d245e28e51c1072a21",
    strip_prefix = "pybind11_bazel-5f458fa53870223a0de7eeb60480dd278b442698",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/5f458fa53870223a0de7eeb60480dd278b442698.tar.gz"],
)

# V2.10.3, 20230130
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    sha256 = "201966a61dc826f1b1879a24a3317a1ec9214a918c8eb035be2f30c3e9cfbdcb",
    strip_prefix = "pybind11-2.10.3",
    urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip"],
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(name = "local_config_python")

# proto_library, cc_proto_library, and java_proto_library rules implicitly
# depend on @com_google_protobuf for protoc and proto runtimes.
# This statement defines the @com_google_protobuf repo.
http_archive(
    name = "com_google_protobuf",
    sha256 = "dc167b7d23ec0d6e4a3d4eae1798de6c8d162e69fa136d39753aaeb7a6e1289d",
    strip_prefix = "protobuf-23.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v23.1.tar.gz"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

http_archive(
    name = "com_google_riegeli",
    sha256 = "5615438b3809fdd62266030e2c6f19c457a15bfb6ef3aa8132503e8584305f8a",
    strip_prefix = "riegeli-254e6d74ee0d325676739fe5075e5a1a895624cf",
    urls = [
        "https://github.com/google/riegeli/archive/254e6d74ee0d325676739fe5075e5a1a895624cf.tar.gz",
    ],
)

# Riegeli's dependencies
http_archive(
    name = "net_zstd",
    build_file = "@com_google_riegeli//third_party:net_zstd.BUILD",
    sha256 = "b6c537b53356a3af3ca3e621457751fa9a6ba96daf3aebb3526ae0f610863532",
    strip_prefix = "zstd-1.4.5/lib",
    urls = ["https://github.com/facebook/zstd/archive/v1.4.5.zip"],  # 2020-05-22
)

http_archive(
    name = "lz4",
    build_file = "@com_google_riegeli//third_party:lz4.BUILD",
    sha256 = "4ec935d99aa4950eadfefbd49c9fad863185ac24c32001162c44a683ef61b580",
    strip_prefix = "lz4-1.9.3/lib",
    urls = ["https://github.com/lz4/lz4/archive/refs/tags/v1.9.3.zip"],  # 2020-11-16
)

http_archive(
    name = "snappy",
    build_file = "@com_google_riegeli//third_party:snappy.BUILD",
    sha256 = "7ee7540b23ae04df961af24309a55484e7016106e979f83323536a1322cedf1b",
    strip_prefix = "snappy-1.2.0",
    urls = ["https://github.com/google/snappy/archive/1.2.0.zip"],  # 2024-04-04
)

http_archive(
    name = "crc32c",
    build_file = "@com_google_riegeli//third_party:crc32.BUILD",
    sha256 = "338f1d9d95753dc3cdd882dfb6e176bbb4b18353c29c411ebcb7b890f361722e",
    strip_prefix = "crc32c-1.1.0",
    urls = ["https://github.com/google/crc32c/archive/1.1.0.zip"],  # 2019-05-24
)

http_archive(
    name = "zlib",
    build_file = "@com_google_riegeli//third_party:zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = ["http://zlib.net/fossils/zlib-1.2.11.tar.gz"],  # 2017-01-15
)

http_archive(
    name = "highwayhash",
    build_file = "@com_google_riegeli//third_party:highwayhash.BUILD",
    sha256 = "5380cb7cf19e7c9591f31792b7794d48084f6a3ab7c03d637cd6a32cf2ee8686",
    strip_prefix = "highwayhash-a7f68e2f95fac08b24327d74747521cf634d5aff",
    urls = ["https://github.com/google/highwayhash/archive/a7f68e2f95fac08b24327d74747521cf634d5aff.zip"],  # 2023-08-09
)

# Tensorflow, 20230705
http_archive(
    name = "org_tensorflow",
    sha256 = "63025cb60d00d9aa7a88807651305a38abb9bb144464e2419c03f13a089d19a6",
    strip_prefix = "tensorflow-2.12.1",
    urls = ["https://github.com/tensorflow/tensorflow/archive/v2.12.1.zip"],
)

load("@org_tensorflow//tensorflow/tools/toolchains:cpus/aarch64/aarch64_compiler_configure.bzl", "aarch64_compiler_configure")  # buildifier: disable=load-on-top

# This import (along with the org_tensorflow archive) is necessary to provide the devtoolset-9 toolchain
load("@org_tensorflow//tensorflow/tools/toolchains/remote_config:configs.bzl", "initialize_rbe_configs")  # buildifier: disable=load-on-top

initialize_rbe_configs()

aarch64_compiler_configure()
