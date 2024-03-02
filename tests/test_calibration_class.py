"""Test the Calibration class."""
import unittest

import numpy as np

from openptv_python.calibration import (
    Calibration,
    Exterior,
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
        assert (np.all(self.cal.glass_par == np.array([0.0, 0.0, 1.0])))

    def test_ap_52_initialization(self):
        """Test ap_52 parameters initialization."""
        assert np.array_equal(self.cal.added_par, \
            np.array([0,0,0,0,0,1,0],dtype=np.float64))

    def test_mmlut_initialization(self):
        """Test mmlut parameters initialization."""
        mml = self.cal.mmlut
        assert np.all(mml.origin == np.zeros(3))
        assert mml.nr == 0
        assert mml.nz == 0
        assert mml.rw == 0
        assert mml.data.shape == ()
        # assert isinstance(mml.data, np.ndarray)
        # assert mml.data.shape == (3,)

    def test_calibration_initialization(self):
        """Test calibration parameters initialization."""
        assert self.cal.ext_par.dtype == Exterior.dtype
        assert self.cal.int_par.dtype == Interior.dtype
        assert isinstance(self.cal.glass_par, np.ndarray)
        assert self.cal.added_par.dtype == ap_52.dtype
        assert self.cal.mmlut.dtype == mm_lut.dtype

        assert isinstance(self.cal.added_par, np.ndarray)
        assert isinstance(self.cal.mmlut, np.recarray)

    def test_exterior_initialization(self):
        """Test exterior parameters initialization."""
        self.cal.update_rotation_matrix()
        ext = self.cal.ext_par.view(np.recarray)
        assert np.allclose(ext.dm, np.identity(3, dtype=np.float64))
        assert ext.omega == 0.0
        assert ext.phi == 0.0
        assert ext.kappa == 0.0
        assert ext.x0 == 0.0
        assert ext.y0 == 0.0


if __name__ == "__main__":
    unittest.main()
