"""Test ray_tracing function."""
import unittest

import numpy as np

from openptv_python.calibration import Calibration, Exterior, Glass, Interior, ap_52
from openptv_python.parameters import ControlPar
from openptv_python.parameters import MultimediaPar as mm_np
from openptv_python.ray_tracing import ray_tracing


class TestRayTracing(unittest.TestCase):
    """Test ray_tracing function."""

    def test_ray_tracing(self):
        """Test ray_tracing function."""
        # Test Case 1
        x, y = 0, 0
        cal = Calibration()  # create Variable for cal with necessary data
        cpar = ControlPar()
        expected_output = np.zeros(3), np.zeros(3)
        output = ray_tracing(x, y, cal, cpar.mm)
        assert np.allclose(output, expected_output)

    def test_ray_tracing_bard(self):
        """Tests the ray_tracing() function."""
        # Input values
        x = 100.0
        y = 100.0

        # The exterior parameters
        test_Ex = Exterior(
            0.0,
            0.0,
            100.0,
            0.0,
            0.0,
            0.0,
            ((1.0, 0.2, -0.3), (0.2, 1.0, 0.0), (-0.3, 0.0, 1.0)),
        )

        # The interior parameters
        test_I = Interior(0.0, 0.0, 100.0)

        # The glass parameters
        test_G = Glass(0.0001, 0.00001, 1.0)

        # The addp parameters
        test_addp = ap_52(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        # The calibration parameters
        test_cal = Calibration(test_Ex, test_I, test_G, test_addp)

        # The mm_np parameters
        test_mm = mm_np(3, 1.0, (1.49, 0.0, 0.0), (5.0, 0.0, 0.0), 1.33)

        # Expected output values
        expected_X = [110.406944, 88.325788, 0.988076]
        expected_a = [0.387960, 0.310405, -0.867834]

        # Call the ray_tracing() function
        actual_X, actual_a = ray_tracing(x, y, test_cal, test_mm)

        # Check that the actual output values are equal to the expected output values
        assert np.allclose(actual_X, expected_X)
        assert np.allclose(actual_a, expected_a)


if __name__ == "__main__":
    unittest.main()
