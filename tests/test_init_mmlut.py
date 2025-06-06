import unittest
from pathlib import Path

import numpy as np

from openptv_python.calibration import mmlut_dtype, read_calibration
from openptv_python.multimed import init_mmlut
from openptv_python.parameters import read_control_par, read_volume_par


class TestInitMmLut(unittest.TestCase):
    def test_init_mmLUT(self):
        filepath = Path("tests") / "testing_fodder"
        ori_file = filepath / "cal" / "cam2.tif.ori"
        add_file = filepath / "cal" / "cam2.tif.addpar"
        vol_file = filepath / "parameters" / "criteria.par"
        filename = filepath / "parameters" / "ptv.par"

        self.assertTrue(Path(ori_file).exists, f"File {ori_file} does not exist")
        self.assertTrue(Path(add_file).exists, f"File {add_file} does not exist")
        cal = read_calibration(ori_file, add_file)
        self.assertIsNotNone(cal, "\n ORI or ADDPAR file reading failed \n")

        self.assertTrue(Path(vol_file).exists, f"File {vol_file} does not exist")
        vpar = read_volume_par(vol_file)
        self.assertIsNotNone(vpar, "\n volume parameter file reading failed \n")

        self.assertTrue(Path(filename).exists, f"File {filename} does not exist")
        cpar = read_control_par(filename)
        self.assertIsNotNone(cpar, "\n control parameter file reading failed\n ")

        # test_mmlut = [mmlut() for _ in range(cpar.num_cams)]
        correct_mmlut = np.array(
            ((0.0, 0.0, -250.00001105), 130, 177, 2), dtype=mmlut_dtype
        )
        correct_mmlut = correct_mmlut.view(np.recarray)

        # run init_mmLUT for one camera only
        cpar.num_cams = 1

        cal = init_mmlut(vpar, cpar, cal)

        # Data[0] Is the radial shift of a point directly on the glass vector

        data = cal.mmlut_data.flatten()

        self.assertAlmostEqual(data[0], 1)

        # Radial shift grows with radius
        self.assertLess(data[0], data[correct_mmlut.nz])
        self.assertLess(
            data[correct_mmlut.nz],
            data[2 * correct_mmlut.nz],
        )

        self.assertAlmostEqual(cal.mmlut.origin[0], correct_mmlut.origin[0], places=5)
        self.assertAlmostEqual(cal.mmlut.origin[1], correct_mmlut.origin[1], places=5)
        self.assertAlmostEqual(cal.mmlut.origin[2], correct_mmlut.origin[2], places=5)
        self.assertEqual(cal.mmlut.nr, correct_mmlut.nr)
        self.assertEqual(cal.mmlut.nz, correct_mmlut.nz)
        self.assertEqual(cal.mmlut.rw, correct_mmlut.rw)


if __name__ == "__main__":
    unittest.main()
