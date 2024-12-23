load("@rules_python//python:pip.bzl", "compile_pip_requirements")


py_library(
    name = "setup",
    srcs = ["setup.py"],
)

compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)
