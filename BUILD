load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@rules_python//python:packaging.bzl", "py_wheel", 'py_package')

package(default_visibility = ["//visibility:public"])

compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)

py_wheel(
    name = "array_record_wheel",
    distribution = "array_record",
    version = "0.6.0",
    platform = select({
        "@platforms//os:macos": "macosx_14_0_arm64",
        "@platforms//os:linux": "manylinux2014_x86_64",
    }),
    deps = [
        "//array_record/python:array_record_data_source",
        "//array_record/python:array_record_module",
        "//array_record/python:init",
        "//array_record/beam:beam",
        "//array_record:package_info",
    ],
)
