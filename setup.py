from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension("sample", sources=["sample.pyx"], include_dirs=['.', numpy.get_include()])
setup(name="sample", ext_modules=cythonize([ext]))
