# ArrayRecord is a new file format for IO intensive applications.
# It supports efficient random access and various compression algorithms.

load("@python//3.10:defs.bzl", compile_pip_requirements_3_10 = "compile_pip_requirements")
load("@python//3.11:defs.bzl", compile_pip_requirements_3_11 = "compile_pip_requirements")
load("@python//3.12:defs.bzl", compile_pip_requirements_3_12 = "compile_pip_requirements")
load("@python//3.13:defs.bzl", compile_pip_requirements_3_13 = "compile_pip_requirements")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "setup",
    srcs = ["setup.py"],
)

compile_pip_requirements_3_10(
    name = "requirements_3_10",
    requirements_in = "test_requirements.in",
    requirements_txt = "test_requirements_lock_3_10.txt",
)

compile_pip_requirements_3_11(
    name = "requirements_3_11",
    requirements_in = "test_requirements.in",
    requirements_txt = "test_requirements_lock_3_11.txt",
)

compile_pip_requirements_3_12(
    name = "requirements_3_12",
    requirements_in = "test_requirements.in",
    requirements_txt = "test_requirements_lock_3_12.txt",
)

compile_pip_requirements_3_13(
    name = "requirements_3_13",
    requirements_in = "test_requirements.in",
    requirements_txt = "test_requirements_lock_3_13.txt",
)
