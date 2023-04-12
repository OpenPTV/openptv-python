import numpy as np

from openptv_python.calibration import (
    Calibration,
    Exterior,
    Glass,
    Interior,
    ap_52,
    mmlut,
)
from openptv_python.vec_utils import vec3d


def test_exterior_initialization():
    """Test exterior parameters initialization."""
    ext = Calibration().ext_par
    assert np.all(ext.dm == np.zeros(3,3))
    assert ext.omega == 0.0
    assert ext.phi == 0.0
    assert ext.kappa == 0.0
    assert ext.x0 == 0.0
    assert ext.y0 == 0.0
    assert ext.z0 == 0.0


def test_interior_initialization():
    """Test interior parameters initialization."""
    intr = Calibration().int_par
    assert intr.xh == 0.0
    assert intr.yh == 0.0
    assert intr.cc == 0.0


def test_glass_initialization():
    """Test glass parameters initialization."""
    glass = Calibration().glass_par
    assert glass.vec_x == 0.0
    assert glass.vec_y == 0.0
    assert glass.vec_z == 0.0


def test_ap_52_initialization():
    """Test ap_52 parameters initialization."""
    ap = Calibration().added_par
    assert ap.k1 == 0.0
    assert ap.k2 == 0.0
    assert ap.k3 == 0.0
    assert ap.p1 == 0.0
    assert ap.p2 == 0.0
    assert ap.scx == 1.0
    assert ap.she == 0.0


def test_mmlut_initialization():
    """Test mmlut parameters initialization."""
    mml = Calibration().mmlut
    assert np.all(mml.origin == np.zeros(3))
    assert mml.nr == 0
    assert mml.nz == 0
    assert mml.rw == 0
    assert mml.data is None
    # assert isinstance(mml.data, np.ndarray)
    # assert mml.data.shape == (3,)


def test_calibration_initialization():
    """Test calibration parameters initialization."""
    calib = Calibration()
    assert isinstance(calib.ext_par, Exterior)
    assert isinstance(calib.int_par, Interior)
    assert isinstance(calib.glass_par, Glass)
    assert isinstance(calib.added_par, ap_52)
    assert isinstance(calib.mmlut, mmlut)
