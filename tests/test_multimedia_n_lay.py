import unittest
from pathlib import Path

import numpy as np

from openptv_python.calibration import read_calibration
from openptv_python.multimed import init_mmlut, multimed_nlay, multimed_r_nlay
from openptv_python.parameters import read_control_par, read_volume_par


class TestMultimedRnlay(unittest.TestCase):
    def setUp(self):
        filepath = Path("tests") / "testing_fodder"
        ori_file = filepath / "cal" / "cam1.tif.ori"
        add_file = filepath / "cal" / "cam1.tif.addpar"
        self.cal = read_calibration(ori_file, add_file)
        self.assertIsNotNone(self.cal, "ORI or ADDPAR file reading failed")

        vol_file = filepath / "parameters" / "criteria.par"
        self.vpar = read_volume_par(vol_file)
        self.assertIsNotNone(self.vpar, "volume parameter file reading failed")

        filename = filepath / "parameters" / "ptv.par"
        self.cpar = read_control_par(filename)
        self.assertIsNotNone(self.cpar, "control parameter file reading failed")

        self.cpar.num_cams = 1

    def test_multimed_r_nlay(self):
        """Test the non-recursive version of multimed_r_nlay."""
        pos = np.array([self.cal.ext_par.x0, self.cal.ext_par.y0, 0.0])
        tmp = multimed_r_nlay(self.cal, self.cpar.mm, pos)
        self.assertAlmostEqual(tmp, 1.0)

        self.cal = init_mmlut(self.vpar, self.cpar, self.cal)

        # print("finished with init_mmlut \n")
        # print(self.cal.mmlut.nr, self.cal.mmlut.nz, self.cal.mmlut.rw)

        # Set up input position and expected output values
        pos = np.array([1.23, 1.23, 1.23])

        correct_Xq = 0.74811917
        correct_Yq = 0.75977975

        # radial_shift = multimed_r_nlay (self.cal, self.cpar.mm, pos)
        # print(f"radial shift is {radial_shift}")

        # /* if radial_shift == 1.0, this degenerates to Xq = X, Yq = Y  */
        # Xq = self.cal.ext_par.x0 + (pos[0] - self.cal.ext_par.x0) * radial_shift
        # Yq = self.cal.ext_par.y0 + (pos[1] - self.cal.ext_par.y0) * radial_shift

        # print("\n Xq = %f, Yq = %f \n" % (Xq, Yq));

        # Call function and check output values
        Xq, Yq = multimed_nlay(self.cal, self.cpar.mm, pos)
        self.assertAlmostEqual(Xq, correct_Xq, delta=1e-8)
        self.assertAlmostEqual(Yq, correct_Yq, delta=1e-8)

    def test_multimed_r_nlay_2(self):
        """Test the non-recursive version of multimed_r_nlay."""
        # Set up input position and expected output values
        pos = np.array([1.23, 1.23, 1.23])

        radial_shift = multimed_r_nlay(self.cal, self.cpar.mm, pos)
        # print(f"radial_shift = {radial_shift}")

        self.assertAlmostEqual(radial_shift, 1.0035607, delta=1e-6)
        correct_Xq = 0.8595652692
        correct_Yq = 0.8685290653

        # Call function and check output values
        Xq, Yq = multimed_nlay(self.cal, self.cpar.mm, pos)
        self.assertAlmostEqual(Xq, correct_Xq, delta=1e-6)
        self.assertAlmostEqual(Yq, correct_Yq, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
