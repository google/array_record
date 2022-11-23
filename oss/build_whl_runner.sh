#!/bin/bash
# run build_whl.sh for different python versions

set -x

for p in 8 9
do
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.${p} 1

  PYTHON_MINOR_VERSION=${p}  oss/build_whl.sh \
    --crosstool_top=@sigbuild-r2.9-python3.${p}_config_cuda//crosstool:toolchain

  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.${p} 0
done

cp /tmp/array_record_pip_pkg/*.whl /tmp/array_record/