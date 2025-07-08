"""Setup.py file for array_record."""

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

REQUIRED_PACKAGES = [
    'absl-py',
    'etils[epath]',
]

BEAM_EXTRAS = [
    'apache-beam[gcp]==2.53.0',
    'google-cloud-storage>=2.11.0',
    'tensorflow>=2.14.0'
]


class BinaryDistribution(Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True


setup(
    name='array_record',
    version='0.7.2',
    description='A file format that achieves a new frontier of IO efficiency',
    author='ArrayRecord team',
    author_email='no-reply@google.com',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.so']},
    python_requires='>=3.10',
    install_requires=REQUIRED_PACKAGES,
    extras_require={'beam': BEAM_EXTRAS},
    url='https://github.com/google/array_record',
    license='Apache-2.0',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    zip_safe=False,
    distclass=BinaryDistribution,
)
