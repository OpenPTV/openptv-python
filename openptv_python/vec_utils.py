# implemnentation of vec_utils.h

# Implementation detail: yes, we use loops. In this day and age, compilers
# can optimize this away at the cost of code size, so it is up to the build
# system to decide whether to invest in loop peeling etc. Here we write
# the logical structure, and allow optimizing for size as well.

import math

import numpy as np
from numba import njit

# Define the np.ndarray type as an numpy array of 3 floats
# vec3d = np.empty(3, dtype=float)

# and 2 floats
#  = np.empty(2, dtype=float)

@njit
def norm(x: float, y: float, z: float) -> float:
    """Return the norm of a 3D vector given by 3 float components."""
    return vec_norm(vec_set(x, y, z))

@njit
def vec_set(x: float, y: float, z: float) -> np.ndarray:
    """Set the components of a  3D vector from separate doubles."""
    return np.array([x, y, z])


def vec_copy(src: np.ndarray) -> np.ndarray:
    """Copy one 3D vector into another."""
    return src.copy()

@njit
def vec_subt(from_: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Subtract two 3D vectors."""
    return from_ - sub

@njit
def vec_add(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Add two 3D vectors."""
    return vec1 + vec2

@njit
def vec_scalar_mul(vec: np.ndarray, scalar: float) -> np.ndarray:
    """vec_scalar_mul(np.ndarray, scalar) multiplies a vector by a scalar."""
    return vec * scalar

@njit
def vec_diff_norm(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """vec_diff_norm() gives the norm of the difference between two vectors."""
    # return np.linalg.norm(vec1 - vec2)
    vec = vec1 - vec2
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

@njit
def vec_norm(vec: np.ndarray) -> float:
    """vec_norm() gives the norm of a vector."""
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

@njit
def vec_dot(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """vec_dot() gives the dot product of two vectors as lists of floats."""
    # return np.dot(vec1, vec2)
    return float(vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2])

@njit
def vec_cross(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Cross product of two vectors."""
    # return np.cross(vec1, vec2)
    return np.array([vec1[1]*vec2[2] - vec1[2]*vec2[1],
                 vec1[2]*vec2[0] - vec1[0]*vec2[2],
                 vec1[0]*vec2[1] - vec1[1]*vec2[0]])

@njit
def vec_cmp(vec1: np.ndarray, vec2: np.ndarray, tol: float = 1e-6) -> bool:
    """vec_cmp() checks whether two vectors are equal within a tolerance."""
    return np.allclose(vec1, vec2, atol=tol)

@njit
def unit_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to a unit vector."""
    magnitude = vec_norm(vec)
    if magnitude == 0:
        return vec  # Avoid division by zero for zero vectors
    return vec / magnitude

@njit
def vec_init(length=3) -> np.ndarray:
    """Initialize a vector to zero."""
    return np.zeros(length, dtype=float)
