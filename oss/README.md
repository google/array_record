# Steps to build and publish a new `array_record` PyPI package

`array_record` supports automatic publishing to PyPI via GitHub Actions.
Once you're ready to create a new release you need to:

1. Update the version number in `setup.py`.

2. Go to [GitHub Actions page](https://github.com/google/array_record/actions),
   select `Build and Publish Release` workflow, and run it. It will spin up a few
   test jobs, and once all of them complete successfully, a `publish-wheel` will start.

3. On completion you should notice a new release on https://pypi.org/project/array-record/#history.

---

If you want to build a wheel locally in your development environment in the root folder, run:

```sh
./oss/build_whl.sh
```
to use the current `python3` version. Otherwise, optionally set:
```sh
PYTHON_VERSION=3.9 ./oss/build_whl.sh
```

Wheels are in `all_dist/`.
