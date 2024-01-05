import unittest

import numpy as np

from openptv_python.calibration import Calibration
from openptv_python.imgcoord import flat_image_coord, img_coord
from openptv_python.parameters import ControlPar, MultimediaPar


class Test_image_coordinates(unittest.TestCase):
    def setUp(self):
        self.control = ControlPar(4)
        self.calibration = Calibration()
        self.control.mm = MultimediaPar(n1=1, n2=[1], n3=1, d=[1])

    def test_img_coord_typecheck(self):
        # with self.assertRaises(TypeError):
        x, y = flat_image_coord(
            np.zeros(3, dtype=float), cal=self.calibration, mm=self.control.mm
        )
        assert x == 0 and y == 0

        with self.assertRaises(ValueError):
            _, _ = flat_image_coord(
                np.empty(
                    2,
                ),
                cal=self.calibration,
                mm=self.control.mm,
            )

        with self.assertRaises(ValueError):
            _, _ = img_coord(
                np.empty((10, 3)),
                cal=self.calibration,
                mm=self.control.mm,
            )

        with self.assertRaises(ValueError):
            _, _ = img_coord(
                np.zeros((10, 2)),
                cal=self.calibration,
                mm=self.control.mm,
            )

    def test_image_coord_regress(self):
        """Test image coordinates for a simple case."""
        self.calibration.set_pos(np.array([0, 0, 40]))
        self.calibration.set_angles([0, 0, 0])
        self.calibration.set_primary_point(np.array([0, 0, 10]))
        self.calibration.set_glass_vec(np.array([0, 0, 20]))
        self.calibration.set_radial_distortion(np.array([0, 0, 0]))
        self.calibration.set_decentering(np.array([0, 0]))
        self.calibration.set_affine_trans(np.array([1, 0]))

        # self.mult = MultimediaPar(n1=1, n2=np.array([1]), n3=1, d=np.array([1]))

        input_pos = np.array([10.0, 5.0, -20.0])  # vec3d

        x = 10.0 / 6.0
        y = x / 2.0

        xp, yp = flat_image_coord(
            orig_pos=input_pos, cal=self.calibration, mm=self.control.mm
        )

        np.testing.assert_array_equal(np.array([xp, yp]), np.array([x, y]))


if __name__ == "__main__":
    unittest.main()
