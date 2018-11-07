import codecs
import os

from setuptools import find_packages, setup

import version

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


# ==============================================================================
# Variables
# ==============================================================================

NAME = "modularity"
VERSION = version.get_version()
DESCRIPTION = "modularity: A repository for finding modules in complex networks."
LONG_DESCRIPTION = read('README.rst')
PACKAGES = find_packages()
AUTHOR = "Ankit N. Khambhati"
AUTHOR_EMAIL = "akhambhati@gmail.com"
DOWNLOAD_URL = 'http://github.com/akhambhati/modularity'
LICENSE = 'MIT'
INSTALL_REQUIRES = ['numpy', 'scipy']

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
)
