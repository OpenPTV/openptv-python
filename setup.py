from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules = cythonize("openptv_python/fast_targ_rec.pyx")
)
