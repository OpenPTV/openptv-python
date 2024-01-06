import unittest

import numpy as np

from openptv_python.calibration import Exterior
from openptv_python.multimed import back_trans_point, trans_cam_point
from openptv_python.parameters import MultimediaPar
from openptv_python.vec_utils import vec_norm, vec_set


class TestTransformFunctions(unittest.TestCase):
    def test_back_trans_point(self):
        """Test back trans point."""
        pos = np.r_[100.0, 100.0, 0.0]

        test_Ex = Exterior.copy()
        test_Ex.z0 = 100.
        test_Ex.dm=np.array([[1.0, 0.2, -0.3], [0.2, 1.0, 0.0], [-0.3, 0.0, 1.0]])


        correct_Ex_t = Exterior.copy()
        correct_Ex_t.z0 = 99.0

        test_G = np.array((0.0001, 0.00001, 1.0))

        #
        test_mm = MultimediaPar(1, 1.0, [1.49, 0.0, 0.0], [5.0, 0.0, 0.0], 1.33)

        Ex_t = Exterior.copy()

        pos_t, cross_p, cross_c, Ex_t.z0 = trans_cam_point(test_Ex, test_mm, test_G, pos)

        np.allclose(pos, np.r_[141.429134, 0.000000, -0.989000])
        np.allclose(cross_p, np.r_[100.000099, 100.000010, 0.989000])
        np.allclose(cross_c, np.r_[-0.009400, -0.000940, 6.000001])

        pos1 = back_trans_point(pos_t, test_mm, test_G, cross_p, cross_c)  #

        assert np.allclose(Ex_t.z0, correct_Ex_t.z0)
        assert np.allclose(pos, pos1)

    def test_trans_cam_point(self):
        pos = vec_set(100.0, 100.0, 0.0)
        sep_norm = vec_norm(pos)

        test_Ex = Exterior.copy()
        test_Ex.z0 = 100.0
        test_Ex.dm = np.array([[1.0, 0.2, -0.3], [0.2, 1.0, 0.0], [-0.3, 0.0, 1.0]])

        correct_Ex_t = Exterior.copy()
        correct_Ex_t.z0 = 50.0
        correct_Ex_t.dm = np.array([[1.0, 0.2, -0.3], [0.2, 1.0, 0.0], [-0.3, 0.0, 1.0]])


        test_G = np.array((0.0, 0.0, 50.0))
        # ap_52 test_addp = {0., 0., 0., 0., 0., 1., 0.};
        # Calibration test_cal = {test_Ex, test_I, test_G, test_addp};

        test_mm = MultimediaPar(1, 1.0, [1.49, 0.0, 0.0], [5.0, 0.0, 0.0], 1.33)

        Ex_t = Exterior.copy()
        pos_t, cross_p, cross_c, Ex_t.z0 = trans_cam_point(test_Ex, test_mm, test_G, pos)

        self.assertTrue(np.allclose(pos_t, np.r_[sep_norm, 0.0, -test_G[2]]))
        self.assertTrue(np.allclose(cross_p, np.r_[pos[0], pos[1], test_G[2]]))
        self.assertTrue(
            np.allclose(
                cross_c, np.r_[test_Ex.x0, test_Ex.y0, test_G[2] + test_mm.d[0]]
            )
        )

        self.assertAlmostEqual(Ex_t.x0, correct_Ex_t.x0)
        self.assertAlmostEqual(Ex_t.y0, correct_Ex_t.y0)
        self.assertAlmostEqual(Ex_t.z0, correct_Ex_t.z0)


if __name__ == "__main__":
    unittest.main()
