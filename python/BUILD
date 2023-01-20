# Python binding for ArrayRecord

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

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
)

py_test(
    name = "array_record_module_test",
    srcs = ["array_record_module_test.py"],
    data = [":array_record_module.so"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
    ],
)

py_strict_library(
    name = "array_record_data_source",
    srcs = ["array_record_data_source.py"],
    deps = [
        ":array_record_module",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/etils/epath",
        "//util/task/python:error",
    ],
)

py_test(
    name = "array_record_data_source_test",
    srcs = ["array_record_data_source_test.py"],
    data = [
        "//third_party/py/grain/_src/python/testdata:digits.array_record-00000-of-00002",
        "//third_party/py/grain/_src/python/testdata:digits.array_record-00001-of-00002",
    ],
    deps = [
        ":array_record_data_source",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
    ],
)
