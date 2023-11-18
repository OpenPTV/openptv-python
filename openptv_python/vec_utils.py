# implemnentation of vec_utils.h

# Implementation detail: yes, we use loops. In this day and age, compilers
# can optimize this away at the cost of code size, so it is up to the build
# system to decide whether to invest in loop peeling etc. Here we write
# the logical structure, and allow optimizing for size as well.

import numpy as np

# Define the np.ndarray type as an numpy array of 3 floats
# vec3d = np.empty(3, dtype=float)

# and 2 floats
#  = np.empty(2, dtype=float)


def norm(x: float, y: float, z: float) -> float:
    """Return the norm of a 3D vector given by 3 float components."""
    return float(np.linalg.norm(vec_set(x, y, z)))


def vec_set(x: float, y: float, z: float) -> np.ndarray:
    """Set the components of a  3D vector from separate doubles."""
    return np.r_[x, y, z]


def vec_copy(src: np.ndarray) -> np.ndarray:
    """Copy one 3D vector into another."""
    return src.copy()


def vec_subt(from_: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Subtract two 3D vectors."""
    return from_ - sub


def vec_add(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Add two 3D vectors."""
    return vec1 + vec2


def vec_scalar_mul(vec: np.ndarray, scalar: float) -> np.ndarray:
    """vec_scalar_mul(np.ndarray, scalar) multiplies a vector by a scalar."""
    return vec * scalar


def vec_diff_norm(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """vec_diff_norm() gives the norm of the difference between two vectors."""
    return float(np.linalg.norm(vec1 - vec2))


def vec_norm(vec: np.ndarray) -> float:
    """vec_norm() gives the norm of a vector."""
    return float(np.linalg.norm(vec))


def vec_dot(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """vec_dot() gives the dot product of two vectors as lists of floats."""
    return np.dot(vec1, vec2)


def vec_cross(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Cross product of two vectors."""
    return np.cross(vec1, vec2)


def vec_cmp(vec1: np.ndarray, vec2: np.ndarray, tol: float = 1e-6) -> bool:
    """vec_cmp() checks whether two vectors are equal within a tolerance."""
    return np.allclose(vec1, vec2, atol=tol)


def unit_vector(vec: np.ndarray) -> np.ndarray:
    """Create unit vector as a list of floats."""
    normed = vec_norm(vec)
    if normed == 0:
        normed = 1.0
    out = vec_scalar_mul(vec, 1.0 / normed)
    return out


def vec_init():
    """Initialize a vector to zero."""
    return np.zeros(3, dtype=float)
