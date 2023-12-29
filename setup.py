from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules = cythonize("openptv_python/orientation.py")
)
