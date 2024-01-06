import numpy as np
import pytest

from openptv_python.calibration import Exterior, rotation_matrix


# Define some test fixtures
@pytest.fixture
def exterior():
    """Create an exterior object with all angles set to zero."""
    return Exterior.copy()


# Define some test cases
def test_rotation_matrix_0(exterior):
    """Test when all angles are zero."""
    # Test when all angles are zero
    rotation_matrix(exterior)
    # rotation_matrix(exterior)
    assert np.allclose(exterior[0]['dm'], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))


def test_rotation_matrix_pi_2(exterior):
    """Test when all angles are pi/2."""
    exterior["phi"] = np.pi / 2
    exterior["omega"] = np.pi / 2
    exterior["kappa"] = np.pi / 2
    rotation_matrix(exterior)
    # rotation_matrix(exterior)
    assert np.allclose(exterior[0]["dm"], np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]))


def test_rotation_matrix_pi(exterior):
    """Test when all angles are pi."""
    exterior['phi'] = np.pi
    exterior['omega'] = np.pi
    exterior['kappa'] = np.pi
    rotation_matrix(exterior)
    assert np.allclose(exterior[0]['dm'], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))


def test_opposite_rotation():
    ex = Exterior.copy()
    ex['omega'] = np.pi/2
    rotation_matrix(ex)
    assert np.allclose(ex[0]['dm'], np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))

    ex = Exterior.copy()
    ex['phi'] = np.pi/2
    rotation_matrix(ex)
    assert np.allclose(ex[0]['dm'], np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))

    ex = Exterior.copy()
    ex['kappa'] = np.pi/2
    rotation_matrix(ex)
    assert np.allclose(ex[0]['dm'], np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))


def test_manual_rotation_pi():
    """Test when all angles are pi."""
    Ex = Exterior.copy()
    Ex['phi'] = np.pi
    Ex['omega'] = np.pi
    Ex['kappa'] = np.pi

    rotation_matrix(Ex)

    assert np.allclose(Ex[0]['dm'], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

def test_rotation_matrix():
    """Test rotation_matrix()."""
    ext = np.zeros(1, dtype=Exterior.dtype)
    # set angles from the original test in liboptv (see check_calibration.c)
    ext['omega'] = -0.2383291
    ext['phi'] = 0.2442810
    ext['kappa'] = 0.0552577

    # update rotation matrix
    rotation_matrix(ext) #ext['dm'] is updated

    expected_dm = np.array([
        [0.9688305, -0.0535899, 0.2418587],
    [-0.0033422, 0.9734041, 0.2290704],
    [-0.2477021, -0.2227387, 0.9428845],
    ])

    assert np.allclose(ext[0]['dm'], expected_dm)
