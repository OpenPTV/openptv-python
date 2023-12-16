"""Ray tracing."""
from typing import Tuple

import numpy as np
from numba import njit

from .calibration import Calibration
from .lsqadj import matmul
from .parameters import MultimediaPar
from .vec_utils import (
    unit_vector,
    vec_add,
    vec_dot,
    vec_norm,
    vec_scalar_mul,
    vec_subt,
)


def ray_tracing(
    x: float, y: float, cal: Calibration, mm: MultimediaPar
) -> Tuple[np.ndarray, np.ndarray]:
    """Ray tracing.

        /*  ray_tracing () traces the optical ray through the multi-media interface of
        (presently) three layers, typically air - glass - water, and returns the
        position of the ray crossing point and the vector normal to the interface.
        See refs. [1,2].

    Arguments:
    ---------
        double x, y - metric position of a point in the image space
        Calibration *cal - parameters of a specific camera.
        mm_np mm - multi-media information (thickness, index of refraction)

        Output Arguments:
        vec3d X - crossing point position.
        vec3d out - vector pointing normal to the interface.
    */

    Args:
    ----
            x (_type_): _description_
            y (_type_): _description_
            cal (_type_): _description_
            mm (_type_): _description_

    Returns
    -------
            _type_: _description_
    """
    primary_point = np.r_[cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0]
    glass = np.r_[cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z]
    return fast_ray_tracing(
        x,
        y,
        cal.int_par.cc,
        cal.ext_par.dm,
        primary_point,
        glass,
        mm.d[0],
        mm.n1,
        mm.n2[0],
        mm.n3,
    )

@njit
def fast_ray_tracing(
    camera_x: float,
    camera_y: float,
    camera_cc: float,
    distortion_matrix: np.ndarray,
    primary_point: np.ndarray,
    glass_vector: np.ndarray,
    distance_param: float,
    refractive_index_medium1: float,
    refractive_index_medium2: float,
    refractive_index_medium3: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast ray tracing.

    Parameters
    ----------
    - camera_x, camera_y, camera_cc: Camera parameters
    - distortion_matrix: Distortion matrix
    - primary_point: Primary point coordinates
    - glass_vector: Glass vector
    - distance_param: Distance parameter
    - refractive_index_medium1, refractive_index_medium2, refractive_index_medium3: Refractive indices

    Returns
    -------
    - Tuple containing the resulting point X and output direction vector
    """
    # Initial ray direction in global coordinate system
    initial_ray_direction = np.array([camera_x, camera_y, -1 * camera_cc])
    initial_ray_direction = unit_vector(initial_ray_direction)
    transformed_direction = np.empty(3, dtype=float)
    matmul(transformed_direction, distortion_matrix, initial_ray_direction, 3, 3, 1, 3, 3)

    glass_direction = unit_vector(glass_vector)
    c_param = vec_norm(glass_vector) + distance_param

    # Project start ray on glass vector to find n1/n2 interface.
    dist_cam_glass = vec_dot(glass_direction, primary_point) - c_param
    dot_product_start_dir = float(vec_dot(glass_direction, transformed_direction))

    # avoid division by zero
    if dot_product_start_dir == 0.0:
        dot_product_start_dir = 1.0
    d1 = -dist_cam_glass / dot_product_start_dir

    transformed_direction_scaled = vec_scalar_mul(transformed_direction, d1)
    Xb = vec_add(primary_point, transformed_direction_scaled)

    # Break down ray into glass-normal and glass-parallel components. */
    n = vec_dot(transformed_direction, glass_direction)
    transformed_direction_parallel = vec_scalar_mul(glass_direction, n)

    transformed_direction_perpendicular = vec_subt(transformed_direction, transformed_direction_parallel)
    bp = unit_vector(transformed_direction_perpendicular)

    # Transform to direction inside glass, using Snell's law
    p = np.sqrt(1 - n * n) * refractive_index_medium1 / refractive_index_medium2
    # glass parallel
    n = -np.sqrt(1 - p * p)
    # glass normal

    # Propagation length in glass parallel to glass vector */
    transformed_direction_parallel_scaled = vec_scalar_mul(bp, p)
    transformed_direction_perpendicular_scaled = vec_scalar_mul(glass_direction, n)
    a2 = vec_add(transformed_direction_parallel_scaled, transformed_direction_perpendicular_scaled)

    abs_dot_product = np.abs(vec_dot(glass_direction, a2))

    # avoid division by zero
    if abs_dot_product == 0:
        abs_dot_product = 1.0
    d2 = distance_param / abs_dot_product

    # point on the horizontal plane between n2,n3
    a2_scaled = vec_scalar_mul(a2, d2)
    X = vec_add(Xb, a2_scaled)

    # Again, direction in next medium
    n = vec_dot(a2, glass_direction)
    transformed_direction_perpendicular_scaled = vec_subt(a2, transformed_direction_perpendicular_scaled)
    bp = unit_vector(transformed_direction_perpendicular_scaled)

    p = np.sqrt(1 - n * n)
    p = p * refractive_index_medium2 / refractive_index_medium3
    n = -np.sqrt(1 - p * p)

    transformed_direction_parallel_scaled = vec_scalar_mul(bp, p)
    transformed_direction_perpendicular_scaled = vec_scalar_mul(glass_direction, n)
    out = vec_add(transformed_direction_parallel_scaled, transformed_direction_perpendicular_scaled)

    return X, out

# def fast_ray_tracing(
#     x,
#     y,
#     cc,
#     dm,
#     primary_point,
#     glass,
#     d,
#     n1,
#     n2,
#     n3,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#     """ Fast ray tracing.

#     Parameters:
#     - x, y, cc: Camera parameters
#     - dm: Distortion matrix
#     - primary_point: Primary point coordinates
#     - glass: Glass vector
#     - d: Distance parameter
#     - n1, n2, n3: Refractive indices

#     Returns:
#     - Tuple containing the resulting point X and output direction vector
#     """
#     # Initial ray direction in global coordinate system
#     tmp1 = np.array([x, y, -1 * cc])
#     tmp1 = unit_vector(tmp1)
#     start_dir = np.empty(3, dtype=float)
#     matmul(start_dir, dm, tmp1, 3, 3, 1, 3, 3)


#     glass_dir = unit_vector(glass)
#     c = vec_norm(glass) + d

#     # Project start ray on glass vector to find n1/n2 interface.
#     dist_cam_glass = vec_dot(glass_dir, primary_point) - c
#     tmp1 = float(vec_dot(glass_dir, start_dir))

#     # avoid division by zero
#     if tmp1 == 0.0:
#         tmp1 = 1.0
#     d1 = -dist_cam_glass / tmp1

#     tmp1 = vec_scalar_mul(start_dir, d1)
#     Xb = vec_add(primary_point, tmp1)

#     # Break down ray into glass-normal and glass-parallel components. */
#     n = vec_dot(start_dir, glass_dir)
#     tmp1 = vec_scalar_mul(glass_dir, n)

#     tmp2 = vec_subt(start_dir, tmp1)
#     bp = unit_vector(tmp2)

#     # Transform to direction inside glass, using Snell's law
#     p = np.sqrt(1 - n * n) * n1 / n2
#     # glass parallel
#     n = -np.sqrt(1 - p * p)
#     # glass normal

#     # Propagation length in glass parallel to glass vector */
#     tmp1 = vec_scalar_mul(bp, p)
#     tmp2 = vec_scalar_mul(glass_dir, n)
#     a2 = vec_add(tmp1, tmp2)

#     tmp1 = np.abs(vec_dot(glass_dir, a2))

#     # avoid division by zero
#     if tmp1 == 0:
#         tmp1 = 1.0
#     d2 = d / tmp1

#     #   point on the horizontal plane between n2,n3  */
#     tmp1 = vec_scalar_mul(a2, d2)
#     X = vec_add(Xb, tmp1)

#     # Again, direction in next medium */
#     n = vec_dot(a2, glass_dir)
#     tmp2 = vec_subt(a2, tmp2)
#     bp = unit_vector(tmp2)

#     p = np.sqrt(1 - n * n)
#     p = p * n2 / n3
#     n = -np.sqrt(1 - p * p)

#     tmp1 = vec_scalar_mul(bp, p)
#     tmp2 = vec_scalar_mul(glass_dir, n)
#     out = vec_add(tmp1, tmp2)

#     return X, out
