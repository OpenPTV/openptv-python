import unittest
from math import isclose
from unittest.mock import Mock

import numpy as np

from openptv_python.calibration import Calibration, Exterior, Glass, Interior
from openptv_python.multimed import back_trans_point, trans_cam_point
from openptv_python.parameters import MultimediaPar


class TestBackTransPoint(unittest.TestCase):
    def setUp(self):
        self.pos_t = np.array([1, 2, 3])
        self.mm = Mock()
        self.mm.d = [4]
        self.G = Mock()
        self.G.vec_x = 0
        self.G.vec_y = 0
        self.G.vec_z = 1
        self.cross_p = np.array([10, 20])
        self.cross_c = np.array([30, 40, 50])

    def test_back_trans_point_returns_numpy_array(self):
        result = back_trans_point(
            self.pos_t, self.mm, self.G, self.cross_p, self.cross_c
        )
        self.assertIsInstance(result, np.ndarray)

    def test_back_trans_point_returns_correct_shape(self):
        result = back_trans_point(
            self.pos_t, self.mm, self.G, self.cross_p, self.cross_c
        )
        self.assertEqual(result.shape, (3,))

    def test_back_trans_point_returns_expected_output(self):
        result = back_trans_point(
            self.pos_t, self.mm, self.G, self.cross_p, self.cross_c
        )
        expected_output = np.array([30.0, 40.0, 49.25])
        np.testing.assert_allclose(result, expected_output, rtol=1e-5)

    def test_back_trans_point(self):
        pos = np.array([100.0, 100.0, 0.0])

        test_Ex = Exterior(
            x0=0.0,
            y0=0.0,
            z0=100.0,
            omega=0.0,
            phi=0.0,
            kappa=0.0,
            dm=np.array([[1.0, 0.2, -0.3], [0.2, 1.0, 0.0], [-0.3, 0.0, 1.0]]),
        )

        correct_Ex_t = Exterior(
            x0=0.0,
            y0=0.0,
            z0=99.0,
            omega=-0.0,
            phi=0.0,
            kappa=0.0,
            dm=np.array([[-0.0, -0.0, -0.0], [-0.0, 0.0, -0.0], [0.0, -0.0, -0.0]]),
        )

        test_I = Interior(0.0, 0.0, 100.0)
        test_G = Glass(0.0001, 0.00001, 1.0)
        test_addp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        Calibration(test_Ex, test_I, test_G, test_addp)

        test_mm = MultimediaPar(
            1, 1.0, np.array([1.49, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), 1.33
        )

        pos_t, cross_p, cross_c = trans_cam_point(
            test_Ex, test_mm, test_G, pos, correct_Ex_t
        )
        pos1 = back_trans_point(pos_t, test_mm, test_G, cross_p, cross_c)

        self.assertTrue(isclose(pos1[0], pos[0], rel_tol=1e-9, abs_tol=1e-9))
        self.assertTrue(isclose(pos1[1], pos[1], rel_tol=1e-9, abs_tol=1e-9))
        self.assertTrue(isclose(pos1[2], pos[2], rel_tol=1e-9, abs_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
