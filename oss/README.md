# Steps to build and publish a new `array_record` release

`array_record` automatically publishes Python wheels to PyPI, cuts a GitHub
Release, and opens a Bazel Central Registry (BCR) pull request via GitHub
Actions (`.github/workflows/publish_release.yml`).

Once you're ready to create a new release:

1. Update `version = "..."` in `MODULE.bazel`. `setup.py` reads the version
   directly from `MODULE.bazel`.

2. Submit the change. When the commit is pushed to `main`, the
   `Build and Publish Release` workflow runs automatically (or can be triggered
   manually from the
   [GitHub Actions page](https://github.com/google/array_record/actions)):
   - Builds and tests Python wheels across all supported OS/Python versions,
   - Publishes the wheels to https://pypi.org/project/array-record/#history,
   - Creates the `v<VERSION>` GitHub Release with `array_record-v<VERSION>.tar.gz`,
   - Opens a pull request to `bazelbuild/bazel-central-registry`.

---

If you want to build a wheel locally in your development environment in the root
folder, run:

```sh
./oss/build_whl.sh
```
to use the current `python3` version. Otherwise, optionally set:
```sh
PYTHON_VERSION=3.11 ./oss/build_whl.sh
```

Wheels are in `all_dist/`.
