# implemnentation of vec_utils.h

# Implementation detail: yes, we use loops. In this day and age, compilers
# can optimize this away at the cost of code size, so it is up to the build
# system to decide whether to invest in loop peeling etc. Here we write
# the logical structure, and allow optimizing for size as well.

import numpy as np

# Define the vec3d type as a list of floats
vec3d = np.zeros(3, dtype=float)


def norm(x: float, y: float, z: float) -> float:
    """Return the norm of a 3D vector given by 3 float components."""
    return np.linalg.norm(vec_set(x, y, z))


def vec_init() -> vec3d:
    """vec_init() initializes all components of a 3D vector to zeros."""
    return vec3d


def vec_set(x: float, y: float, z: float) -> vec3d:
    """Set the components of a  3D vector from separate doubles."""
    return np.array([x, y, z], dtype=float)


def vec_copy(src: vec3d) -> vec3d:
    """Copy one 3D vector into another."""
    return src.copy()


def vec_subt(from_: vec3d, sub: vec3d):
    """Subtract two 3D vectors."""
    return from_ - sub


def vec_add(vec1: vec3d, vec2: vec3d) -> vec3d:
    """Add two 3D vectors."""
    return vec1 + vec2


def vec_scalar_mul(vec: vec3d, scalar: float) -> vec3d:
    """vec_scalar_mul(vec3d, scalar) multiplies a vector by a scalar."""
    return vec * scalar


def vec_diff_norm(vec1: vec3d, vec2: vec3d) -> vec3d:
    """vec_diff_norm() gives the norm of the difference between two vectors."""
    return np.linalg.norm(vec1 - vec2)


def vec_norm(vec: np.ndarray) -> float:
    """vec_norm() gives the norm of a vector."""
    return np.linalg.norm(vec)


def vec_dot(vec1: vec3d, vec2: vec3d) -> vec3d:
    """vec_dot() gives the dot product of two vectors as lists of floats."""
    return vec1.dot(vec2)


def vec_cross(vec1: vec3d, vec2: vec3d) -> vec3d:
    """Cross product of two vectors."""
    return vec1.cross(vec2)


def vec_cmp(vec1: vec3d, vec2: vec3d, tol: float = 1e-6) -> bool:
    """vec_cmp() checks whether two vectors are equal within a tolerance."""
    return np.allclose(vec1, vec2, atol=tol)


def unit_vector(vec: np.ndarray) -> np.ndarray:
    """Create unit vector as a list of floats."""
    normed = vec_norm(vec)
    if normed == 0:
        normed = 1.0
    out = vec_scalar_mul(vec, 1.0 / normed)
    return out
