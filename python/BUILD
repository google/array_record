# Python binding for ArrayRecord

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@pypi//:requirements.bzl", "requirement")
load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")

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

py_test(
    name = "read_my_data_test",
    srcs = ["read_my_data_test.py"],
    data = [
        ":array_record_module.so",
        "//python/testdata:fine_tuning_train_multilabel.array_record",
    ],
    deps = [
        "//third_party/py/requests",
        "//third_party/py/tensorflow",
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

py_binary(
    name = "read_my_data_bin",
    srcs = ["read_my_data_test.py"],
    data = [":array_record_module.so"],
    main = "read_my_data_test.py",
    deps = [
        "//third_party/py/tensorflow",
        "@abseil-py//absl/testing:absltest",
    ],
)
