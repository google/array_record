"""Setup.py file for array_record."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='array_record',
    version='0.1.0',
    description=(
        'A file format that achieves a new frontier of IO efficiency'
        ),
    author='ArrayRecord team',
    author_email='no-reply@google.com',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.so']},
    python_requires='>=3.7',
    install_requires=['absl-py'],
    url='https://github.com/google/array_record',
    license='Apache-2.0',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    zip_safe=False,
)
