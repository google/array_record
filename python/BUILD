# Python binding for ArrayRecord

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

pybind_extension(
    name = "array_record_module",
    srcs = ["array_record_module.cc"],
    deps = [
        "//third_party/absl/status",
        "//third_party/absl/strings",
        "//third_party/array_record/cpp:array_record_reader",
        "//third_party/array_record/cpp:array_record_writer",
        "//third_party/array_record/cpp:thread_pool",
        "//third_party/riegeli/bytes:fd_reader",
        "//third_party/riegeli/bytes:fd_writer",
    ],
)

py3_test(
    name = "array_record_module_test",
    srcs = ["array_record_module_test.py"],
    deps = [
        ":array_record_module",
        "//third_party/py/absl/testing:absltest",
    ],
)
