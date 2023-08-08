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
    deps = [
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "array_record_data_source",
    srcs = ["array_record_data_source.py"],
    data = [":array_record_module.so"],
    deps = [
        # Implicit etils (/epath) dependency.
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
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:flagsaver",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)
