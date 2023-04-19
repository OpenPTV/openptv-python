import unittest

from openptv_python.calibration import Calibration
from openptv_python.parameters import ControlPar
from openptv_python.trafo import flat_to_dist, metric_to_pixel, pixel_to_metric

EPS = 1e-10


class TestFlatToDist(unittest.TestCase):
    def test_no_calibration(self):
        with self.assertRaises(TypeError):
            flat_to_dist([0, 0], None)

    def test_zero_coordinates(self):
        cal = Calibration()
        print(id(cal))
        dist_x, dist_y = flat_to_dist(0, 0, cal)
        self.assertEqual(dist_x, 0)
        self.assertEqual(dist_y, 0)

    def test_positive_coordinates(self):
        cal = Calibration()
        cal.int_par.xh = 100
        cal.int_par.yh = 50
        dist_x, dist_y = flat_to_dist(50, 25, cal)
        self.assertEqual(dist_x, 150)
        self.assertEqual(dist_y, 75)

    def test_distortion(self):
        cal = Calibration()
        cal.set_added_par([1e-5, 0, 0, 0, 0, 1, 0])
        dist_x, dist_y = flat_to_dist(100, 200, cal)
        self.assertAlmostEqual(dist_x, 150, places=3)
        self.assertAlmostEqual(dist_y, 300, places=3)

    def test_pixel_to_metric_and_back(self):
        cpar = ControlPar()
        cpar.from_file("tests/testing_folder/control_parameters/control.par")

        x, y = metric_to_pixel(1, 1, cpar)
        x, y = pixel_to_metric(x, y, cpar)
        self.assertTrue(abs(x - 1) < EPS)
        self.assertTrue(abs(y - 1) < EPS)

    def test_0_0(self):
        xc = 0.0
        yc = 0.0
        cpar = ControlPar()
        cpar.imx = 1024
        cpar.imy = 1008
        cpar.pix_x = 0.010
        cpar.pix_y = 0.010

        xp, yp = metric_to_pixel(xc, yc, cpar)
        xc1, yc1 = pixel_to_metric(xp, yp, cpar)

        self.assertTrue(abs(xc1 - xc) < EPS)
        self.assertTrue(abs(yc1 - yc) < EPS)

    def test_1_0(self):
        xc = 1.0
        yc = 0.0
        cpar = ControlPar()
        cpar.imx = 1024
        cpar.imy = 1008
        cpar.pix_x = 0.010
        cpar.pix_y = 0.010

        xp, yp = metric_to_pixel(xc, yc, cpar)
        xc1, yc1 = pixel_to_metric(xp, yp, cpar)

        self.assertTrue(abs(xc1 - xc) < EPS)
        self.assertTrue(abs(yc1 - yc) < EPS)

    def test_0_neg1(self):
        xc = 0.0
        yc = -1.0
        cpar = ControlPar()
        cpar.imx = 1024
        cpar.imy = 1008
        cpar.pix_x = 0.010
        cpar.pix_y = 0.010

        xp, yp = metric_to_pixel(xc, yc, cpar)
        xc1, yc1 = pixel_to_metric(xp, yp, cpar)

        self.assertTrue(abs(xc1 - xc) < EPS)
        self.assertTrue(abs(yc1 - yc) < EPS)


if __name__ == "__main__":
    unittest.main()
