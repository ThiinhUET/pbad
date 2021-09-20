# Cython compile instructions

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# to build: python setup.py build_ext --inplace
extensions = [
    Extension('cpatternm',
              ['*.pyx'],
              include_dirs=[numpy.get_include()],
              extra_link_args=['-Wno-unreachable-code'])
]

# setup
setup(
    ext_modules=cythonize(extensions),
)
