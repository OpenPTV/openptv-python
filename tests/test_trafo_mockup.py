import unittest
from pathlib import Path

import numpy as np

from openptv_python.calibration import Calibration, interior_dtype
from openptv_python.parameters import ControlPar
from openptv_python.trafo import (
    dist_to_flat,
    flat_to_dist,
    metric_to_pixel,
    pixel_to_metric,
)

EPS = 1e-10


class TestFlatToDist(unittest.TestCase):
    """Test the flat_to_dist function."""

    def setUp(self):
        self.cpar = ControlPar()
        self.cpar.imx = 1024
        self.cpar.imy = 1008
        self.cpar.pix_x = 0.010
        self.cpar.pix_y = 0.010

    def test_zero_coordinates(self):
        """Test the case when the input coordinates are zero."""
        cal = Calibration()
        dist_x, dist_y = flat_to_dist(0, 0, cal)
        self.assertEqual(dist_x, 0)
        self.assertEqual(dist_y, 0)

    def test_positive_coordinates(self):
        """Test the case when the input coordinates are positive."""
        cal = Calibration()
        cal.int_par['xh'] = 100
        cal.int_par['yh'] = 50
        dist_x, dist_y = flat_to_dist(50, 25, cal)
        self.assertEqual(dist_x, 150)
        self.assertEqual(dist_y, 75)

    def test_distortion(self):
        """Test the case when the distortion is not zero."""
        cal = Calibration()
        cal.set_added_par(np.array([1e-5, 0, 0, 0, 0, 1, 0], dtype=np.float64))
        dist_x, dist_y = flat_to_dist(100, 200, cal)
        self.assertAlmostEqual(dist_x, 150, places=3)
        self.assertAlmostEqual(dist_y, 300, places=3)

    def test_pixel_to_metric_and_back(self):
        """Test the pixel_to_metric and metric_to_pixel functions."""
        cpar = ControlPar().from_file(Path("tests/testing_folder/control_parameters/control.par"))

        x, y = metric_to_pixel(1, 1, cpar)
        x, y = pixel_to_metric(x, y, cpar)
        self.assertTrue(abs(x - 1) < EPS)
        self.assertTrue(abs(y - 1) < EPS)

    def test_0_0(self):
        """Test the case when the input coordinates are zero."""
        xc = 0.0
        yc = 0.0

        xp, yp = metric_to_pixel(xc, yc, self.cpar)
        xc1, yc1 = pixel_to_metric(xp, yp, self.cpar)

        self.assertTrue(abs(xc1 - xc) < EPS)
        self.assertTrue(abs(yc1 - yc) < EPS)

    def test_1_0(self):
        """Test the case when the input coordinates are (1, 0)."""
        xc = 1.0
        yc = 0.0

        xp, yp = metric_to_pixel(xc, yc, self.cpar)
        xc1, yc1 = pixel_to_metric(xp, yp, self.cpar)

        self.assertTrue(abs(xc1 - xc) < EPS)
        self.assertTrue(abs(yc1 - yc) < EPS)

    def test_0_neg1(self):
        """Test the case when the input coordinates are (0, -1)."""
        xc = 0.0
        yc = -1.0

        xp, yp = metric_to_pixel(xc, yc, self.cpar)
        xc1, yc1 = pixel_to_metric(xp, yp, self.cpar)

        self.assertTrue(abs(xc1 - xc) < EPS)
        self.assertTrue(abs(yc1 - yc) < EPS)

    def dist_flat_round_trip(self):
        """Test the round trip from flat to distorted and back."""
        # /*  Cheks that the order of operations in converting metric flat image to
        #     distorted image coordinates and vice-versa is correct.

        #     Distortion value in this one is kept to a level suitable (marginally)
        #     to an Rmax = 10 mm sensor. The allowed round-trip error is adjusted
        #     accordingly. Note that the higher the distortion parameter, the worse
        #     will the round-trip error be, at least unless we introduce more iteration
        # */

        x = 10.0
        y = 10.0
        iter_eps = 1e-5

        cal = Calibration()
        cal.int_par = np.array( (1.5, 1.5, 60.0), dtype = interior_dtype)
        cal.set_added_par(np.array([0.0005, 0, 0, 0, 0, 1, 0], dtype=np.float64))

        xres, yres = flat_to_dist(x, y, cal)
        xres, yres = dist_to_flat(xres, yres, cal, 0.00001)

        self.assertTrue(abs(xres - x) < iter_eps)
        self.assertTrue(abs(yres - y) < iter_eps)


if __name__ == "__main__":
    unittest.main()
