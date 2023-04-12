import numpy as np
import pytest
from openptv_python.calibration import rotation_matrix, Exterior

# Define some test fixtures
@pytest.fixture
def exterior():
    return Exterior(phi=0, omega=0, kappa=0, dm=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

# Define some test cases
def test_rotation_matrix_0(exterior):
    # Test when all angles are zero
    result = rotation_matrix(exterior)
    assert np.all(result.dm == [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def test_rotation_matrix_pi_2(exterior):
    # Test when all angles are pi/2
    exterior.phi = np.pi/2
    exterior.omega = np.pi/2
    exterior.kappa = np.pi/2
    result = rotation_matrix(exterior)
    print(result.dm)
    assert np.allclose(result.dm ,np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]))

def test_rotation_matrix_pi(exterior):
    # Test when all angles are pi
    exterior.phi = np.pi
    exterior.omega = np.pi
    exterior.kappa = np.pi
    result = rotation_matrix(exterior)
    print(result.dm)
    assert np.allclose(result.dm, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    
def test_opposite_rotation():
    ex = Exterior(phi=np.pi, omega=np.pi, kappa=np.pi, dm=[[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    # Rotate the matrix using the rotation_matrix function
    ex_rotated = rotation_matrix(ex)
    assert(np.allclose(ex_rotated.dm, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
    
def test_manual_rotation_pi():
    Ex = Exterior(phi=np.pi, omega=np.pi, kappa=np.pi, dm=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    cp = np.cos(Ex.phi)
    sp = np.sin(Ex.phi)
    co = np.cos(Ex.omega)
    so = np.sin(Ex.omega)
    ck = np.cos(Ex.kappa)
    sk = np.sin(Ex.kappa)

    # Modify the Exterior Ex with the new Dmatrix
    Ex.dm[0][0] = cp * ck
    Ex.dm[0][1] = -cp * sk
    Ex.dm[0][2] = sp
    Ex.dm[1][0] = co * sk + so * sp * ck
    Ex.dm[1][1] = co * ck - so * sp * sk
    Ex.dm[1][2] = -so * cp
    Ex.dm[2][0] = so * sk - co * sp * ck
    Ex.dm[2][1] = so * ck + co * sp * sk
    Ex.dm[2][2] = co * cp

    Ex.dm = np.round(Ex.dm, 7)
    
    assert(np.allclose(Ex.dm, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
    