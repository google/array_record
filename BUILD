# ArrayRecord is a new file format for IO intensive applications.
# It supports efficient random access and various compression algorithms.

load("@rules_python//python:pip.bzl", "compile_pip_requirements")


package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "setup",
    srcs = ["setup.py"],
)

compile_pip_requirements(
    name = "requirements",
    requirements_in = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)
