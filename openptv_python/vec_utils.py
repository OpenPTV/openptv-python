# implemnentation of vec_utils.h

# Implementation detail: yes, we use loops. In this day and age, compilers
# can optimize this away at the cost of code size, so it is up to the build
# system to decide whether to invest in loop peeling etc. Here we write
# the logical structure, and allow optimizing for size as well.

import math
from typing import List

# Define the vec3d type as a list of floats
vec3d = [0.0, 0.0, 0.0]


# Define the nan value
try:
    EMPTY_CELL = math.nan
except ValueError:
    EMPTY_CELL = 0.0 / 0.0


# Define the helper functions
def is_empty(x):
    return math.isnan(x)


def norm(x, y, z):
    return math.sqrt((x) * (x) + (y) * (y) + (z) * (z))


def vec_init(init):
    # vec_init() initializes all components of a 3D vector to NaN.

    for ix in range(3):
        init[ix] = math.nan


def vec_set(dest, x, y, z):
    # vec_set() sets the components of a  3D vector from separate doubles.

    dest[0] = x
    dest[1] = y
    dest[2] = z


def vec_copy(dest, src):
    # vec_copy() copies one 3D vector into another.

    for ix in range(3):
        dest[ix] = src[ix]


def vec_subt(from_, sub, output):
    # vec_subt() subtracts two 3D vectors.

    for ix in range(3):
        output[ix] = from_[ix] - sub[ix]


def vec_add(vec1, vec2, output):
    # vec_add() adds two 3D vectors.

    for ix in range(3):
        output[ix] = vec1[ix] + vec2[ix]


def vec_scalar_mul(vec: List[float], scalar: float) -> List[float]:
    # vec_scalar_mul() multiplies a vector by a scalar.
    output = []
    for ix in range(3):
        output[ix] = scalar * vec[ix]

    return output


def vec_diff_norm(vec1, vec2):
    # Implements the common operation of finding the norm of a difference between
    # two vectors. This happens a lot, so we have an optimized function.

    return math.sqrt(
        (vec1[0] - vec2[0]) ** 2 + (vec1[1] - vec2[1]) ** 2 + (vec1[2] - vec2[2]) ** 2
    )


def vec_norm(vec: List[float]) -> float:
    # vec_norm() calculates the norm of a vector.

    # Just plug into the macro
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def vec_dot(vec1: List[float], vec2: List[float]):
    """vec_dot() gives the dot product of two vectors as lists of floats."""
    sum_ = 0
    for ix in range(3):
        sum_ += vec1[ix] * vec2[ix]
    return sum_


def vec_cross(vec1: List[float], vec2: List[float]) -> List[float]:
    """Cross product of two vectors.

    Args:
    ----
        vec1 (List[float]): first vector
        vec2 (List[float]): second vector

    Returns:
    -------
        List[float]: cross product of vec1 and vec2
    """
    out = [0, 0, 0]
    out[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    out[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    out[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return out


def vec_cmp(vec1, vec2, tol):
    # vec_cmp() checks whether two vectors are equal.

    for ix in range(3):
        if abs(vec1[ix] - vec2[ix]) > tol:
            return False
    return True


def vec_approx_cmp(vec1: List[float], vec2: List[float], eps: float) -> int:
    for i in range(3):
        if abs(vec1[i] - vec2[i]) > eps:
            return 0
    return 1


def unit_vector(vec: List[float]) -> List[float]:
    """Create unit vector as a list of floats."""
    normed = vec_norm(vec)
    if normed == 0:
        normed = 1.0
    out = vec_scalar_mul(vec, 1.0 / normed)
    return out
