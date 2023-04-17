import unittest
from openptv_python.calibration import ap_52, Calibration
from openptv_python.trafo import flat_to_dist

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

if __name__ == "__main__":
    unittest.main()
