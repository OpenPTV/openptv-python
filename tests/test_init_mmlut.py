import unittest
from pathlib import Path


from openptv_python.calibration import read_calibration
from openptv_python.parameters import read_volume_par, read_control_par
from openptv_python.multimed import init_mmlut
from openptv_python.calibration import mmlut


class TestInitMmLut(unittest.TestCase):
    def test_init_mmLUT(self):
        ori_file = "tests/testing_fodder/cal/cam2.tif.ori"
        add_file = "tests/testing_fodder/cal/cam2.tif.addpar"
        vol_file = "tests/testing_fodder/parameters/criteria.par"
        filename = "tests/testing_fodder/parameters/ptv.par"

        self.assertTrue(Path(ori_file).exists, f"File {ori_file} does not exist")
        self.assertTrue(Path(add_file).exists, f"File {add_file} does not exist")
        cal = read_calibration(ori_file, add_file, None)
        self.assertIsNotNone(cal, "\n ORI or ADDPAR file reading failed \n")

        self.assertTrue(Path(vol_file).exists, f"File {vol_file} does not exist")
        vpar = read_volume_par(vol_file)
        self.assertIsNotNone(vpar, "\n volume parameter file reading failed \n")

        self.assertTrue(Path(filename).exists, f"File {filename} does not exist")
        cpar = read_control_par(filename)
        self.assertIsNotNone(cpar, "\n control parameter file reading failed\n ")

        # test_mmlut = [mmlut() for _ in range(cpar.num_cams)]
        correct_mmlut = [
            mmlut(
                origin=(0.0, 0.0, -250.00001105),
                nr=130,
                nz=177,
                rw=2,
            )
        ]

        # run init_mmLUT for one camera only
        i = 0
        cpar.num_cams = 1

        init_mmlut(vpar, cpar, cal)

        # Data[0] Is the radial shift of a point directly on the glass vector
        self.assertAlmostEqual(cal.mmlut.data[0], 1)

        # Radial shift grows with radius
        self.assertLess(cal.mmlut.data[0], cal.mmlut.data[correct_mmlut[0].nz])
        self.assertLess(
            cal.mmlut.data[correct_mmlut[0].nz],
            cal.mmlut.data[2 * correct_mmlut[0].nz],
        )

        self.assertAlmostEqual(
            cal.mmlut.origin[0], correct_mmlut[i].origin[0], places=8
        )
        self.assertAlmostEqual(
            cal.mmlut.origin[1], correct_mmlut[i].origin[1], places=8
        )
        self.assertAlmostEqual(
            cal.mmlut.origin[2], correct_mmlut[i].origin[2], places=8
        )
        self.assertEqual(cal.mmlut.nr, correct_mmlut[i].nr)
        self.assertEqual(cal.mmlut.nz, correct_mmlut[i].nz)
        self.assertEqual(cal.mmlut.rw, correct_mmlut[i].rw)


if __name__ == '__main__':
    unittest.main()
