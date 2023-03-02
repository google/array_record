# Steps to build a new array_record pip package

1. Update the version number in setup.py

2. In workspace, run

```
cd third_party
./array_record/oss/runner.sh
```

3. Wheels are in `/tmp/array_record/all_dist`.

4. Upload to PyPI:

```
python3 -m pip install --upgrade twine
python3 -m twine upload /tmp/array_record/all_dist/*-any.whl
```

Authenticate with Twine by following https://pypi.org/help/#apitoken.
