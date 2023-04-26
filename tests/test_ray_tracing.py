import numpy as np

from openptv_python.calibration import Calibration
from openptv_python.parameters import ControlPar
from openptv_python.ray_tracing import ray_tracing


def test_ray_tracing():
    """Test ray_tracing function."""
    # Test Case 1
    x, y = 0, 0
    cal = Calibration()  # create Variable for cal with necessary data
    cpar = ControlPar()
    expected_output = np.array(
        [cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0]
    ), np.zeros(3)
    output = ray_tracing(x, y, cal, cpar.mm)
    assert np.allclose(output, expected_output)


if __name__ == "__main__":
    test_ray_tracing()
