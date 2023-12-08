import unittest

import numpy as np

# from ctypes import c_double
from openptv_python.calibration import mm_lut, read_calibration
from openptv_python.multimed import get_mmf_from_mmlut, init_mmlut
from openptv_python.parameters import read_control_par, read_volume_par
from openptv_python.vec_utils import vec_set


class TestGetMmfMmLUT(unittest.TestCase):
    def setUp(self):
        self.ori_file = "tests/testing_fodder/cal/cam2.tif.ori"
        self.add_file = "tests/testing_fodder/cal/cam2.tif.addpar"
        self.vol_file = "tests/testing_fodder/parameters/criteria.par"
        self.ptv_file = "tests/testing_fodder/parameters/ptv.par"
        self.cal = read_calibration(self.ori_file, self.add_file)
        self.vpar = read_volume_par(self.vol_file)
        self.cpar = read_control_par(self.ptv_file)
        self.correct_mmlut = [
            mm_lut(origin=np.array([0.0, 0.0, -250.00001105]), nr=130, nz=177, rw=2)
            for _ in range(4)
        ]

    def test_get_mmf_mm_lut(self):
        """Test the get_mmf_from_mmlut function."""
        init_mmlut(self.vpar, self.cpar, self.cal)

        pos = vec_set(1.0, 1.0, 1.0)
        mmf = get_mmf_from_mmlut(self.cal, pos)
        self.assertAlmostEqual(mmf, 1.0038208, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
