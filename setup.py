#!/usr/bin/python3
# -*- coding:utf-8 -*-
# run via  python3 setup.py upload

import io, os, sys
from shutil import rmtree
from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'python-cmethods'
DESCRIPTION = 'Collection of bias adjustment procedures for multidimensional climate data'
URL = 'https://github.com/btschwertfeger/Bias-Adjustment-Python'
EMAIL = 'development@b-schwertfeger.de'
AUTHOR = 'Benjamin Thomas Schwertfeger'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = None

# What packages are required for this module to be executed?
REQUIRED = [
   'xarray', 'numpy', 'tqdm', 'netCDF4' # <- always conflicts on install with tqdm on test.pypi..
]

# What packages are optional?
EXTRAS = {
    'working examples notebook': ['matplotlib'],
}

here = os.path.abspath(os.path.dirname(__file__))

# only works if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f'\n{f.read()}'
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
if not VERSION:
    project_slug = 'cmethods'
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    raise ValueError('Version not found!')
    exit()

class UploadCommand(Command):
    '''Support setup.py upload.'''

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        print(f'\033[1m{s}\033[0m')

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(f'{sys.executable} setup.py sdist bdist_wheel --universal')

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')
        sys.exit()

class TestUploadCommand(Command):
    '''Support setup.py test upload.'''

    description = 'Build and test publishing the package.'
    user_options = []

    @staticmethod
    def status(s):
        print(f'\033[1m{s}\033[0m')

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(f'{sys.executable} setup.py sdist bdist_wheel --universal')

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload -r testpypi dist/*')

        sys.exit()

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='GPLv3',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
    ],
    cmdclass={
        'upload': UploadCommand,
        'test': TestUploadCommand,
    },
)
