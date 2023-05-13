import unittest
from pathlib import Path

# from ctypes import c_double
from openptv_python.calibration import mmlut, read_calibration
from openptv_python.multimed import get_mmf_from_mmlut, init_mmlut
from openptv_python.parameters import read_control_par, read_volume_par
from openptv_python.vec_utils import vec_set


class TestGetMmfMmLUT(unittest.TestCase):
    def setUp(self):
        self.ori_file = Path("tests/testing_fodder/cal/cam2.tif.ori")
        self.add_file = Path("tests/testing_fodder/cal/cam2.tif.addpar")
        self.vol_file = Path("tests/testing_fodder/parameters/criteria.par")
        self.ptv_file = Path("tests/testing_fodder/parameters/ptv.par")
        self.cal = read_calibration(self.ori_file, self.add_file, None)
        self.vpar = read_volume_par(self.vol_file)
        self.cpar = read_control_par(self.ptv_file)
        self.correct_mmlut = [
            mmlut(origin=[0.0, 0.0, -250.00001105], nr=130, nz=177, rw=2)
            for _ in range(4)
        ]

    def test_get_mmf_mmLUT(self):
        self.assertTrue(self.ori_file.exists(), f"File {self.ori_file} does not exist")
        self.assertTrue(self.add_file.exists(), f"File {self.add_file} does not exist")
        self.assertIsNotNone(self.cal, "\n ORI or ADDPAR file reading failed \n")

        self.assertTrue(self.vol_file.exists(), f"File {self.vol_file} does not exist")
        self.assertIsNotNone(self.vpar, "\n volume parameter file reading failed \n")

        self.assertTrue(self.ptv_file.exists(), f"File {self.ptv_file} does not exist")
        self.assertIsNotNone(self.cpar, "\n control parameter file reading failed\n ")

        init_mmlut(self.vpar, self.cpar, self.cal)

        pos = vec_set(1.0, 1.0, 1.0)
        mmf = get_mmf_from_mmlut(self.cal, pos)
        self.assertAlmostEqual(mmf, 1.00382, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
