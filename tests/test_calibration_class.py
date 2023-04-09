from openptv_python import Calibration
from openptv_python.vec_utils import vec3d


def test_exterior_initialization():
    ext = Calibration.Exterior()
    assert ext.dm == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert ext.omega == 0.0
    assert ext.phi == 0.0
    assert ext.kappa == 0.0
    assert ext.x0 == 0.0
    assert ext.y0 == 0.0
    assert ext.z0 == 0.0


def test_interior_initialization():
    intr = Calibration.Interior()
    assert intr.xh == 0.0
    assert intr.yh == 0.0
    assert intr.cc == 0.0


def test_glass_initialization():
    glass = Calibration.Glass()
    assert glass.vec_x == 0.0
    assert glass.vec_y == 0.0
    assert glass.vec_z == 0.0


def test_ap_52_initialization():
    ap = Calibration.ap_52()
    assert ap.k1 == 0.0
    assert ap.k2 == 0.0
    assert ap.k3 == 0.0
    assert ap.p1 == 0.0
    assert ap.p2 == 0.0
    assert ap.scx == 0.0
    assert ap.she == 0.0


def test_mmlut_initialization():
    mml = Calibration.mmlut()
    assert mml.origin == vec3d
    assert mml.nr == 0
    assert mml.nz == 0
    assert mml.rw == 0
    assert mml.data == []


def test_calibration_initialization():
    calib = Calibration()
    assert isinstance(calib.ext_par, Calibration.Exterior)
    assert isinstance(calib.int_par, Calibration.Interior)
    assert isinstance(calib.glass_par, Calibration.Glass)
    assert isinstance(calib.added_par, Calibration.ap_52)
    assert isinstance(calib.mmlut, Calibration.mmlut)
