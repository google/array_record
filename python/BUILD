# Python binding for ArrayRecord

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@pypi//:requirements.bzl", "requirement")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "gcs_client",
    srcs = ["gcs_client.cc"],
    hdrs = ["gcs_client.h"],
    linkstatic = True,
    deps = [
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@google_cloud_cpp//:common",
        "@google_cloud_cpp//:storage",
    ],
)

cc_test(
    name = "gcs_client_test",
    srcs = ["gcs_client_test.cc"],
    deps = [
        ":gcs_client",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@google_cloud_cpp//:common",
        "@google_cloud_cpp//:storage",
        "@googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "array_record_module",
    srcs = ["array_record_module.cc"],
    # Restrict exported symbols of array_record_module.so in OSS builds to only
    # PyInit_array_record_module. By default in OSS Bazel builds, pybind11_bazel
    # exports all statically linked C++ symbols (including Protobuf and Abseil)
    # globally. When imported in the same process as TensorFlow
    # (libtensorflow_framework), duplicate Protobuf/Abseil symbols collide in
    # the dynamic linker and cause a segmentation fault on macOS and Linux.
    additional_linker_inputs = select({
        "@platforms//os:linux": [":array_record_module.lds"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@platforms//os:osx": ["-Wl,-exported_symbol,_PyInit_array_record_module"],
        "@platforms//os:linux": ["-Wl,--version-script=$(location :array_record_module.lds)"],
        "//conditions:default": [],
    }),
    deps = [
        ":gcs_client",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "//cpp:array_record_reader",
        "//cpp:array_record_writer",
        "//cpp:thread_pool",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:fd_writer",
        "@riegeli//riegeli/gcs:gcs_object",
        "@riegeli//riegeli/gcs:gcs_reader",
    ],
)

py_test(
    name = "array_record_module_test",
    srcs = ["array_record_module_test.py"],
    data = [":array_record_module.so"],
    imports = [".."],
    deps = [
        "@abseil-py//absl/testing:absltest",
    ],
)

py_library(
    name = "array_record_data_source",
    srcs = ["array_record_data_source.py"],
    data = [":array_record_module.so"],
    deps = [
        "@abseil-py//absl/flags",
        "@abseil-py//absl/logging",
        requirement("etils"),
    ],
)

py_test(
    name = "array_record_data_source_test",
    srcs = ["array_record_data_source_test.py"],
    data = [
        ":array_record_module.so",
        "//python/testdata:digits.array_record-00000-of-00002",
        "//python/testdata:digits.array_record-00001-of-00002",
    ],
    imports = [".."],
    deps = [
        ":array_record_data_source",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:flagsaver",
        "@abseil-py//absl/testing:parameterized",
    ],
)
