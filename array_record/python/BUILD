# Python binding for ArrayRecord
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@pypi_array_record//:requirements.bzl", "requirement")
load("@rules_python//python:packaging.bzl", "py_wheel")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

pybind_extension(
    name = "array_record_module",
    srcs = ["array_record_module.cc"],
    deps = [
        "//cpp:array_record_reader",
        "//cpp:array_record_writer",
        "//cpp:thread_pool",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_riegeli//riegeli/bytes:fd_reader",
        "@com_google_riegeli//riegeli/bytes:fd_writer",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "array_record_module_test",
    srcs = ["array_record_module_test.py"],
    data = [":array_record_module"],
    deps = [
        "@com_google_absl_py//absl/testing:absltest",
    ],
    imports = ["../.."]
)

py_library(
    name = "array_record_data_source",
    srcs = ["array_record_data_source.py"],
    data = [":array_record_module.so"],
    deps = [
        requirement("etils")
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "init",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "array_record_data_source_test",
    srcs = ["array_record_data_source_test.py"],
    args = ["--test_srcdir=array_record/python/testdata"],
    data = [
        ":array_record_module",
        "//array_record/python/testdata:digits.array_record-00000-of-00002",
        "//array_record/python/testdata:digits.array_record-00001-of-00002",
    ],
    deps = [
        ":init",
        ":array_record_data_source",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:flagsaver",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
    imports = ["../.."]
)
