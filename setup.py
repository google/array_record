# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup.py file for array_record."""

import pathlib
import re
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

REQUIRED_PACKAGES = [
    'absl-py',
    'etils[epath]',
]

TF_PACKAGE = ['tensorflow>=2.20.0']

BEAM_EXTRAS = [
    'apache-beam[gcp]>=2.53.0',
    'google-cloud-storage>=2.11.0',
] + TF_PACKAGE

TEST_EXTRAS = [
    'jax',
    'grain',
] + TF_PACKAGE


def _get_version() -> str:
  """Reads the package version from MODULE.bazel."""
  base_dir = pathlib.Path(__file__).resolve().parent
  for rel_path in (
      'MODULE.bazel',
      'array_record/MODULE.bazel',
      'oss/MODULE.bazel',
  ):
    module_file = base_dir / rel_path
    if module_file.is_file():
      match = re.search(
          r'module\s*\([^)]*?\bversion\s*=\s*"([^"]+)"',
          module_file.read_text(encoding='utf-8'),
          re.DOTALL,
      )
      if match:
        return match.group(1)
  raise RuntimeError('Could not determine version from MODULE.bazel')


class BinaryDistribution(Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True


setup(
    name='array_record',
    version=_get_version(),
    description='A file format that achieves a new frontier of IO efficiency',
    author='ArrayRecord team',
    author_email='no-reply@google.com',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.so']},
    python_requires='>=3.11',
    install_requires=REQUIRED_PACKAGES,
    extras_require={'beam': BEAM_EXTRAS, 'test': TEST_EXTRAS},
    url='https://github.com/google/array_record',
    license='Apache-2.0',
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
    ],
    zip_safe=False,
    distclass=BinaryDistribution,
)
