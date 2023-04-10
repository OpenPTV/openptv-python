import numpy as np

from .calibration import Calibration, ap_52
from .ray_tracing import ray_tracing


def test_ray_tracing():
    # Test Case 1
    x, y = 0, 0
    cal = Calibration()  # create Variable for cal with necessary data
    mm = ap_52()  # create an object for mm with necessary data
    expected_output = np.array(
        [cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0]
    ), np.zeros(3)
    assert np.allclose(ray_tracing(x, y, cal, mm), expected_output)
