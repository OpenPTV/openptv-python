import unittest

import numpy as np

from openptv_python.calibration import (
    Calibration,
    Exterior,
    Interior,
    ap_52,
)
from openptv_python.imgcoord import flat_image_coord
from openptv_python.parameters import MultimediaPar
from openptv_python.vec_utils import vec_set


class TestFlatCenteredCam(unittest.TestCase):
    def test_flat_centered_cam(self):
        # When the image plane is centered on the axis. and the camera looks to
        # a straight angle (e.g. along an axis), the image position can be
        # gleaned from simple geometry.
        pos = vec_set(10, 5, -20)
        cal = Calibration(
            ext_par=np.array(
                (0, 0, 40, 0, 0, 0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                dtype=Exterior.dtype,
            ).view(np.recarray),
            int_par=np.array((0, 0, 10), dtype=Interior.dtype).view(np.recarray),
            glass_par=np.array((0.0, 0.0, 20.0)),
            added_par=ap_52.copy(),  # (0, 0, 0, 0, 0, 1, 0),
        )
        mm = MultimediaPar(  # All in air, simplest case.
            nlay=1, n1=1, n2=[1], n3=1, d=[1]
        )

        # Output variables
        x, y = flat_image_coord(pos, cal, mm)
        self.assertAlmostEqual(x, 10 / 6.0)
        self.assertAlmostEqual(x, 2 * y)


if __name__ == "__main__":
    unittest.main()
