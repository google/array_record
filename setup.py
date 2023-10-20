"""Setup.py file for array_record."""

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

REQUIRED_PACKAGES = [
    'absl-py',
    'etils[epath]',
]


class BinaryDistribution(Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True


setup(
    name='array_record',
    version='0.5.0',
    description='A file format that achieves a new frontier of IO efficiency',
    author='ArrayRecord team',
    author_email='no-reply@google.com',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.so']},
    python_requires='>=3.9',
    install_requires=REQUIRED_PACKAGES,
    url='https://github.com/google/array_record',
    license='Apache-2.0',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    zip_safe=False,
    distclass=BinaryDistribution,
)
