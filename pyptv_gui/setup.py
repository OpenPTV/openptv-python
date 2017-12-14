# -*- coding: utf-8 -*-
"""
Copyright (c) 2008-2013, Tel Aviv University
Copyright (c) 2013 - the OpenPTV team
The software is distributed under the terms of MIT-like license
http://opensource.org/licenses/MIT
"""

import os
import sys
from distutils.core import setup

import numpy as np
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

if sys.platform.startswith(("linux", "darwin")):
    output_name = 'ptv1.so'
else:
    output_name = 'ptv1.pyd'

origdir = os.getcwd()
os.chdir('../src_c')

inc_dirs = [np.get_include()]

ext_mods = [
    Extension("ptv1", ["ptv1.pyx", "segmentation.c", "tools.c",
                       "image_processing.c", "jw_ptv.c", "peakfitting.c",
                       "correspondences.c", "epi.c", "multimed.c",
                       "imgcoord.c", "orientation.c", "sortgrid.c",
                       "pointpos.c", "intersect.c", "track.c", "ttools.c", "draw.c",
                       "mousefunction.c", "tracking_run.c"],
              include_dirs=inc_dirs, libraries=['optv'], extra_compile_args=['-O3'],
              pyrex_include_dirs=['.']),
    Extension("tracking_run_py", ["tracking_run_py.pyx", "tracking_run.c"],
              libraries=['optv'], include_dirs=inc_dirs, pyrex_include_dirs=['.'])
]

setup(
    name="ptv1",
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_mods,
    py_modules=['ptv1', ],
)
