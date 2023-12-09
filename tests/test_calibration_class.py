"""Test the Calibration class."""
import unittest

import numpy as np

from openptv_python.calibration import (
    Calibration,
    Exterior,
    Glass,
    Interior,
    ap_52,
    mm_lut,
)


class TestCalibrationClass(unittest.TestCase):
    """Test the Calibration class."""

    def setUp(self):
        self.cal = Calibration()

    def test_interior_initialization(self):
        """Test interior parameters initialization."""
        intr = self.cal.int_par
        assert intr.xh == 0.0
        assert intr.yh == 0.0
        assert intr.cc == 0.0

    def test_glass_initialization(self):
        """Test glass parameters initialization."""
        glass = self.cal.glass_par
        assert glass.vec_x == 0.0
        assert glass.vec_y == 0.0
        assert glass.vec_z == 1.0

    def test_ap_52_initialization(self):
        """Test ap_52 parameters initialization."""
        ap = self.cal.added_par
        assert ap.k1 == 0.0
        assert ap.k2 == 0.0
        assert ap.k3 == 0.0
        assert ap.p1 == 0.0
        assert ap.p2 == 0.0
        assert ap.scx == 1.0
        assert ap.she == 0.0

    def test_mmlut_initialization(self):
        """Test mmlut parameters initialization."""
        mml = self.cal.mmlut
        assert np.all(mml.origin == np.zeros(3))
        assert mml.nr == 1
        assert mml.nz == 1
        assert mml.rw == 1
        assert mml.data == np.zeros((1, 1))
        # assert isinstance(mml.data, np.ndarray)
        # assert mml.data.shape == (3,)

    def test_calibration_initialization(self):
        """Test calibration parameters initialization."""
        assert isinstance(self.cal.ext_par, Exterior)
        assert isinstance(self.cal.int_par, Interior)
        assert isinstance(self.cal.glass_par, Glass)
        assert isinstance(self.cal.added_par, ap_52)
        assert isinstance(self.cal.mmlut, mm_lut)

    def test_exterior_initialization(self):
        """Test exterior parameters initialization."""
        ext = self.cal.ext_par
        ext.update_rotation_matrix()
        assert np.allclose(ext.dm, np.identity(3, dtype=np.float64))
        assert ext.omega == 0.0
        assert ext.phi == 0.0
        assert ext.kappa == 0.0
        assert ext.x0 == 0.0
        assert ext.y0 == 0.0
        ext.update_rotation_matrix()
        assert np.allclose(ext.dm, np.identity(3, dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
