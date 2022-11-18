workspace(name = "com_google_array_record")

# Might be better than http_archive
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Abseil LTS 20220623.1
http_archive(
    name = "com_google_absl",
    sha256 = "91ac87d30cc6d79f9ab974c51874a704de9c2647c40f6932597329a282217ba8",  # SHARED_ABSL_SHA
    strip_prefix = "abseil-cpp-20220623.1",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/refs/tags/20220623.1.tar.gz",
    ],
)
http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-eb9225ce361affe561592e0912320b9db84985d0",
    url = "https://github.com/google/googletest/archive/eb9225ce361affe561592e0912320b9db84985d0.zip",
    sha256 = "a7db7d1295ce46b93f3d1a90dbbc55a48409c00d19684fcd87823037add88118",
)

# V3.4.0, 20210818
http_archive(
  name = "eigen3",
  sha256 = "b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626",
  strip_prefix = "eigen-3.4.0",
  urls = [
      "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2",
  ],
  build_file_content =
"""
cc_library(
    name = 'eigen3',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['Eigen/**', 'unsupported/Eigen/**']),
    visibility = ['//visibility:public'],
)
"""
)

## `pybind11_bazel`
# See https://github.com/pybind/pybind11_bazel
http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-72cbbf1fbc830e487e3012862b7b720001b70672",
  sha256 = "516c1b3a10d87740d2b7de6f121f8e19dde2c372ecbfe59aef44cd1872c10395",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/72cbbf1fbc830e487e3012862b7b720001b70672.tar.gz"],
)
# V2.9.2, 20220330
http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.9.2",
  urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.zip"],
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

# V3.20.1, 20220421
# proto_library, cc_proto_library, and java_proto_library rules implicitly
# depend on @com_google_protobuf for protoc and proto runtimes.
# This statement defines the @com_google_protobuf repo.
http_archive(
    name = "com_google_protobuf",
    sha256 = "8b28fdd45bab62d15db232ec404248901842e5340299a57765e48abe8a80d930",
    strip_prefix = "protobuf-3.20.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.20.1.tar.gz"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# Riegeli does not cut releases, so we reference the head
http_archive(
    name = "com_google_riegeli",
    strip_prefix = "riegeli-master",
    urls = [
        "https://github.com/google/riegeli/archive/master.zip",
    ],
)
# Dependencies required by riegeli
http_archive(
    name = "org_brotli",
    patch_args = ["-p1"],
    patches = ["@com_google_riegeli//third_party:brotli.patch"],
    sha256 = "fec5a1d26f3dd102c542548aaa704f655fecec3622a24ec6e97768dcb3c235ff",
    strip_prefix = "brotli-68f1b90ad0d204907beb58304d0bd06391001a4d",
    urls = ["https://github.com/google/brotli/archive/68f1b90ad0d204907beb58304d0bd06391001a4d.zip"],  # 2021-08-18
)
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
    sha256 = "38b4aabf88eb480131ed45bfb89c19ca3e2a62daeb081bdf001cfb17ec4cd303",
    strip_prefix = "snappy-1.1.8",
    urls = ["https://github.com/google/snappy/archive/1.1.8.zip"],  # 2020-01-14
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
    sha256 = "cf891e024699c82aabce528a024adbe16e529f2b4e57f954455e0bf53efae585",
    strip_prefix = "highwayhash-276dd7b4b6d330e4734b756e97ccfb1b69cc2e12",
    urls = ["https://github.com/google/highwayhash/archive/276dd7b4b6d330e4734b756e97ccfb1b69cc2e12.zip"],  # 2019-02-22
)
