import unittest

import numpy as np

from openptv_python.calibration import (
    Calibration,
    Exterior,
    Interior,
    ap_52,
)
from openptv_python.multimed import multimed_r_nlay
from openptv_python.parameters import MultimediaPar


class TestMultimedRnlay(unittest.TestCase):
    """Test the multimed_r_nlay function."""

    def test_multimedia_r_nlay(self):
        """Test the multimed_r_nlay function."""
        test_Ex = Exterior.copy()
        test_Ex['z0'] = 100.0

        test_I = Interior.copy()
        test_I.cc = 100.0
        test_G = np.array((0.0001, 0.00001, 1.0))
        test_addp = ap_52.copy() #(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        test_cal = Calibration(
            test_Ex, test_I, test_G, test_addp
        )  # note that mmlut is default

        test_mm = MultimediaPar(1, 1.0, [1.49, 0.0, 0.0], [5.0, 0.0, 0.0], 1.33)
        print(test_mm.nlay, test_mm.n1, test_mm.n2, test_mm.d, test_mm.n3)

        pos = np.array([1.23, 1.23, 1.23])

        out = multimed_r_nlay(test_cal, test_mm, pos)
        print(f"out = {out} ")
        self.assertAlmostEqual(out, 1.01374403, places=6)


if __name__ == "__main__":
    unittest.main()
