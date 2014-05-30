""" OpenPTV-Python is the GUI for the OpenPTV (http://www.openptv.net) liboptv library 
based on Python/Enthought Traits GUI/Numpy/Chaco

.. moduleauthors:: OpenPTV group

"""

__all__ = ["pyptv_batch", "parameters", "parameters_gui"]

import os, sys
sys.path.append(os.path.abspath('../src_c'))
