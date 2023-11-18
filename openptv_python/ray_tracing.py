"""Ray tracing."""
from typing import Tuple

import numpy as np

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
    # Initial ray direction in global coordinate system
    tmp1 = np.r_[x, y, -1 * cal.int_par.cc]
    tmp1 = unit_vector(tmp1)
    start_dir = np.empty(3, dtype=float)
    matmul(start_dir, cal.ext_par.dm, tmp1, 3, 3, 1, 3, 3)

    primary_point = np.r_[cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0]

    tmp1 = np.r_[cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z]
    glass_dir = unit_vector(tmp1)
    c = vec_norm(tmp1) + mm.d[0]

    # Project start ray on glass vector to find n1/n2 interface.
    dist_cam_glass = vec_dot(glass_dir, primary_point) - c
    tmp1 = vec_dot(glass_dir, start_dir)

    # avoid division by zero
    if tmp1 == 0:
        tmp1 = 1.0
    d1 = -dist_cam_glass / tmp1

    tmp1 = vec_scalar_mul(start_dir, d1)
    Xb = vec_add(primary_point, tmp1)

    # Break down ray into glass-normal and glass-parallel components. */
    n = vec_dot(start_dir, glass_dir)
    tmp1 = vec_scalar_mul(glass_dir, n)

    tmp2 = vec_subt(start_dir, tmp1)
    bp = unit_vector(tmp2)

    # Transform to direction inside glass, using Snell's law
    p = np.sqrt(1 - n * n) * mm.n1 / mm.n2[0]
    # glass parallel
    n = -np.sqrt(1 - p * p)
    # glass normal

    # Propagation length in glass parallel to glass vector */
    tmp1 = vec_scalar_mul(bp, p)
    tmp2 = vec_scalar_mul(glass_dir, n)
    a2 = vec_add(tmp1, tmp2)

    tmp1 = np.abs(vec_dot(glass_dir, a2))

    # avoid division by zero
    if tmp1 == 0:
        tmp1 = 1.0
    d2 = mm.d[0] / tmp1

    #   point on the horizontal plane between n2,n3  */
    tmp1 = vec_scalar_mul(a2, d2)
    X = vec_add(Xb, tmp1)

    # Again, direction in next medium */
    n = vec_dot(a2, glass_dir)
    tmp2 = vec_subt(a2, tmp2)
    bp = unit_vector(tmp2)

    p = np.sqrt(1 - n * n)
    p = p * mm.n2[0] / mm.n3
    n = -np.sqrt(1 - p * p)

    tmp1 = vec_scalar_mul(bp, p)
    tmp2 = vec_scalar_mul(glass_dir, n)
    out = vec_add(tmp1, tmp2)

    return X, out
