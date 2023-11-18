"""Image coordinates."""


from typing import Tuple

import numpy as np

from .calibration import Calibration
from .multimed import back_trans_point, multimed_nlay, trans_cam_point
from .parameters import MultimediaPar
from .trafo import flat_to_dist
from .vec_utils import vec_set


def flat_image_coord(
    orig_pos: np.ndarray, cal: Calibration, mm: MultimediaPar
) -> Tuple[float, float]:
    """Flat image coordinate.

    Args:
    ----
        orig_pos (_type_): _description_
        cal (_type_): _description_
        mm (_type_): _description_

    Returns
    -------
        _type_: _description_
    """
    cal_t = Calibration()
    cal_t.mmlut = cal.mmlut

    # This block calculate 3D position in an imaginary air-filled space,
    # i.e. where the point will have been seen in the absence of refractive
    # layers between it and the camera.
    pos_t, cross_p, cross_c = trans_cam_point(
        cal.ext_par, mm, cal.glass_par, orig_pos, cal_t.ext_par
    )

    # print(f"pos_t {pos_t}")
    # print(f"cross_p {cross_p}")
    # print(f"cros_c {cross_c}")

    x_t, y_t = multimed_nlay(cal_t, mm, pos_t)
    # print(f"x_t {x_t}, y_t {y_t}")

    pos_t = vec_set(x_t, y_t, pos_t[2])
    pos = back_trans_point(pos_t, mm, cal.glass_par, cross_p, cross_c)

    # print(f"pos {pos}")

    deno = (
        cal.ext_par.dm[0][2] * (pos[0] - cal.ext_par.x0)
        + cal.ext_par.dm[1][2] * (pos[1] - cal.ext_par.y0)
        + cal.ext_par.dm[2][2] * (pos[2] - cal.ext_par.z0)
    )

    # print(f"deno {deno}")

    if deno == 0:
        deno = 1

    x = (
        -cal.int_par.cc
        * (
            cal.ext_par.dm[0][0] * (pos[0] - cal.ext_par.x0)
            + cal.ext_par.dm[1][0] * (pos[1] - cal.ext_par.y0)
            + cal.ext_par.dm[2][0] * (pos[2] - cal.ext_par.z0)
        )
        / deno
    )

    y = (
        -cal.int_par.cc
        * (
            cal.ext_par.dm[0][1] * (pos[0] - cal.ext_par.x0)
            + cal.ext_par.dm[1][1] * (pos[1] - cal.ext_par.y0)
            + cal.ext_par.dm[2][1] * (pos[2] - cal.ext_par.z0)
        )
        / deno
    )

    # print(f"x {x}, y {y}")

    return x, y


def flat_image_coordinates(
    orig_pos: np.ndarray, cal: Calibration, mm: MultimediaPar
) -> np.ndarray:
    """Flat image coordinates in array mode."""
    out = np.empty((orig_pos.shape[0], 2))

    for i, row in enumerate(orig_pos):
        out[i, 0], out[i, 1] = flat_image_coord(row, cal, mm)

    return out


def img_coord(
    pos: np.ndarray, cal: Calibration, mm: MultimediaPar
) -> Tuple[float, float]:
    """Image coordinate."""
    # Estimate metric coordinates in image space using flat_image_coord()
    if pos.shape[0] != 3:
        raise ValueError("pos must be a 3D vector")

    x, y = flat_image_coord(pos, cal, mm)
    # print(f"flat_image_coord: x = {x}, y = {y}")

    # Distort the metric coordinates using the Brown distortion model
    x, y = flat_to_dist(x, y, cal)

    # print("f flat_to_dist: x = {x}, y = {y}")

    return x, y


def image_coordinates(
    orig_pos: np.ndarray, cal: Calibration, mm: MultimediaPar
) -> np.ndarray:
    """Image coordinates in array mode."""
    out = np.empty((orig_pos.shape[0], 2), dtype=float)

    for i, row in enumerate(orig_pos):
        out[i, 0], out[i, 1] = img_coord(row, cal, mm)

    return out
