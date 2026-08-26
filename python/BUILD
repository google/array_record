# Python binding for ArrayRecord

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@pypi//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

pybind_extension(
    name = "array_record_module",
    srcs = ["array_record_module.cc"],
    deps = [
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
    deps = [
        "@abseil-py//absl/testing:absltest",
    ],
)

py_library(
    name = "array_record_data_source",
    srcs = ["array_record_data_source.py"],
    data = [":array_record_module.so"],
    deps = [
        requirement("etils"),
    ],
)

py_test(
    name = "array_record_data_source_test",
    srcs = ["array_record_data_source_test.py"],
    args = ["--test_srcdir=python/testdata"],
    data = [
        ":array_record_module.so",
        "//python/testdata:digits.array_record-00000-of-00002",
        "//python/testdata:digits.array_record-00001-of-00002",
    ],
    deps = [
        ":array_record_data_source",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:flagsaver",
        "@abseil-py//absl/testing:parameterized",
    ],
)

cc_library(
    name = "array_record_data_source_cpp",
    srcs = ["array_record_data_source_cpp.cc"],
    hdrs = ["array_record_data_source_cpp.h"],
    deps = [
        ":read_instructions_lib",
        "//cpp:array_record_reader",
        "//cpp:array_record_writer",
        "//util/task:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_riegeli//riegeli/bytes:file_reader",
    ],
)

cc_library(
    name = "read_instructions_lib",
    srcs = ["read_instructions_lib.cc"],
    hdrs = ["read_instructions_lib.h"],
    deps = [
        "//cpp:array_record_reader",
        "//cpp:common",
        "//cpp:parallel_for",
        "//cpp:thread_pool",
        "//file/base",
        "//third_party/py/grain/google/placer",
        "//third_party/re2",
        "//thread",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_riegeli//riegeli/bytes:file_reader",
    ],
)
