# Steps to build a new array_record pip package

1. Update the version number in setup.py

2. In the root folder, run

  ```
  ./oss/build_whl.sh
  ```
  to use the current `python3` version.  Otherwise, optionally set
  ```
  PYTHON_VERSION=3.9 ./oss/build_whl.sh
  ```

3. Wheels are in `all_dist/`.
