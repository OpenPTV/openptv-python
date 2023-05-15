import unittest

from openptv_python.calibration import read_calibration
from openptv_python.multimed import multimed_nlay, multimed_r_nlay
from openptv_python.parameters import read_control_par, read_volume_par


class TestMultimedRnlay(unittest.TestCase):
    def setUp(self):
        ori_file = "tests/testing_fodder/cal/cam1.tif.ori"
        add_file = "tests/testing_fodder/cal/cam1.tif.addpar"
        self.cal = read_calibration(ori_file, add_file, None)
        self.assertIsNotNone(self.cal, "ORI or ADDPAR file reading failed")

        vol_file = "tests/testing_fodder/parameters/criteria.par"
        self.vpar = read_volume_par(vol_file)
        self.assertIsNotNone(self.vpar, "volume parameter file reading failed")

        filename = "tests/testing_fodder/parameters/ptv.par"
        self.cpar = read_control_par(filename)
        self.assertIsNotNone(self.cpar, "control parameter file reading failed")

        self.cpar.num_cams = 1

    # def test_multimed_r_nlay(self):
    #     """Test the non-recursive version of multimed_r_nlay."""
    #     pos = [self.cal.ext_par.x0, self.cal.ext_par.y0, 0.0]
    #     self.assertAlmostEqual(multimed_r_nlay(self.cal, self.cpar.mm, pos), 1.0)

    #     self.cal = init_mmlut(self.vpar, self.cpar, self.cal)

    #     print("finished with init_mmlut \n")
    #     print(self.cal.mmlut.nr, self.cal.mmlut.nz, self.cal.mmlut.rw)

    #     # Set up input position and expected output values
    #     pos = [1.23, 1.23, 1.23]

    #     correct_Xq = 0.74811917
    #     correct_Yq = 0.75977975

    #     # Call function and check output values
    #     Xq, Yq = multimed_nlay(self.cal, self.cpar.mm, pos)
    #     self.assertAlmostEqual(Xq, correct_Xq, delta=1e-8)
    #     self.assertAlmostEqual(Yq, correct_Yq, delta=1e-8)

    def test_multimed_r_nlay_2(self):
        """Test the non-recursive version of multimed_r_nlay."""
        # Set up input position and expected output values
        pos = [1.23, 1.23, 1.23]

        radial_shift = multimed_r_nlay(self.cal, self.cpar.mm, pos)
        print(f"radial_shift = {radial_shift}")

        correct_Xq = 0.74811917
        correct_Yq = 0.75977975

        # Call function and check output values
        Xq, Yq = multimed_nlay(self.cal, self.cpar.mm, pos)
        self.assertAlmostEqual(Xq, correct_Xq, delta=1e-8)
        self.assertAlmostEqual(Yq, correct_Yq, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
