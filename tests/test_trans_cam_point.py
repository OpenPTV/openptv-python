import unittest

import numpy as np

from openptv_python.calibration import Exterior, Glass
from openptv_python.multimed import back_trans_point, trans_cam_point
from openptv_python.parameters import MultimediaPar


class TestTransformFunctions(unittest.TestCase):
    def test_back_trans_point(self):
        pos_t = np.array([1, 2, 3])
        mm = MultimediaPar(d=[4])
        G = Glass(vec_x=5, vec_y=6, vec_z=7)
        cross_p = np.array([8, 9, 10])
        cross_c = np.array([11, 12, 13])

        expected_result = np.array([3.72520005, 4.24471467, 4.76422929])

        result = back_trans_point(pos_t, mm, G, cross_p, cross_c)

        np.testing.assert_allclose(result, expected_result, rtol=1e-7)

    def test_trans_cam_point(self):
        ex = Exterior(x0=1, y0=2, z0=3)
        mm = MultimediaPar(d=[4])
        glass = Glass(vec_x=5, vec_y=6, vec_z=7)
        pos = np.array([8, 9, 10])

        expected_ex_t = Exterior(x0=9.491525423728814, y0=2, z0=4)
        expected_pos_t = np.array([8.77496439, 0.0, 4.48864969])
        expected_cross_p = np.array([8, 9, 10])
        expected_cross_c = np.array([9.49152542, 2.0, 3.0])

        ex_t, pos_t, cross_p, cross_c = trans_cam_point(ex, mm, glass, pos)

        np.testing.assert_allclose(ex_t.x0, expected_ex_t.x0, rtol=1e-7)
        np.testing.assert_allclose(ex_t.y0, expected_ex_t.y0, rtol=1e-7)
        np.testing.assert_allclose(ex_t.z0, expected_ex_t.z0, rtol=1e-7)

        np.testing.assert_allclose(pos_t, expected_pos_t, rtol=1e-7)

        np.testing.assert_allclose(cross_p, expected_cross_p, rtol=1e-7)
        np.testing.assert_allclose(cross_c, expected_cross_c, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
