"""Ray tracing."""
import numpy as np

from openptv_python.calibration import Calibration, mmlut
from openptv_python.vec_utils import unit_vector


def ray_tracing(
    x: float, y: float, cal: Calibration, mm: mmlut
) -> tuple(np.ndarray, np.ndarray):
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

    Returns:
    -------
            _type_: _description_
    """
    d1, d2, c, dist_cam_glass, n, p = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    start_dir = np.zeros(3)
    primary_point = np.array([cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0])
    glass_dir = np.zeros(3)
    bp = np.zeros(3)
    tmp1 = np.zeros(3)
    tmp2 = np.zeros(3)
    Xb = np.zeros(3)
    a2 = np.zeros(3)
    X = np.zeros(3)
    out = np.zeros(3)

    # Initial ray direction in global coordinate system
    start_dir = np.dot(cal.ext_par.dm, unit_vector(np.array([x, y, -cal.int_par.cc])))

    # Project start ray on glass vector to find n1/n2 interface.
    tmp1 = unit_vector(
        np.array([cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z])
    )
    glass_dir = tmp1.copy()
    c = np.linalg.norm(tmp1) + mm.d[0]
    dist_cam_glass = np.dot(glass_dir, primary_point) - c
    d1 = -dist_cam_glass / np.dot(glass_dir, start_dir)
    tmp1 = start_dir * d1
    Xb = primary_point + tmp1

    # Break down ray into glass-normal and glass-parallel components.
    n = np.dot(start_dir, glass_dir)
    tmp1 = glass_dir * n
    bp = start_dir - tmp1
    bp = bp / np.linalg.norm(bp)

    # Transform to direction inside glass, using Snell's law.
    p = np.sqrt(1 - n * n) * mm.n1 / mm.n2[0]  # glass parallel
    n = -np.sqrt(1 - p * p)  # glass normal

    # Propagation length in glass parallel to glass vector.
    tmp1 = bp * p
    tmp2 = glass_dir * n
    a2 = tmp1 + tmp2
    d2 = mm.d[0] / abs(np.dot(glass_dir, a2))

    # Point on the horizontal plane between n2,n3.
    tmp1 = a2 * d2
    X = Xb + tmp1

    # Again, direction in next medium.
    n = np.dot(a2, glass_dir)
    tmp2 = a2 - glass_dir * n
    bp = tmp2 / np.linalg.norm(tmp2)

    p = np.sqrt(1 - n * n)
    p = p * mm.n2[0] / mm.n3
    n = -np.sqrt(1 - p * p)

    tmp1 = bp * p
    tmp2 = glass_dir * n
    out = tmp1 + tmp2

    return X, out
