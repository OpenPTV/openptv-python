"""Image coordinates."""


from typing import Tuple

import numpy as np

from .calibration import Calibration
from .multimed import back_trans_point, multimed_nlay, trans_cam_point
from .parameters import MultimediaPar
from .trafo import flat_to_dist


def flat_image_coord(
    orig_pos: np.ndarray, cal: Calibration, mm: MultimediaPar
) -> Tuple[np.float64, np.float64]:
    """Flat image coordinate.

    Args:
    ----
        orig_pos (np.ndarray): 3D position
        cal (Calibration): camera calibration
        mm (Multimedia): multimedia parameters

    Returns
    -------
        _type_: _description_
    """
    if orig_pos.shape != (3,):
        raise ValueError("orig_pos must be a 3D vector")

    cal_t = Calibration(mmlut = cal.mmlut)

    # This block calculate 3D position in an imaginary air-filled space,
    # i.e. where the point will have been seen in the absence of refractive
    # layers between it and the camera.
    pos_t, cross_p, cross_c, cal_t.ext_par['z0'] = trans_cam_point(
        cal.ext_par, mm, cal.glass_par, orig_pos
    )

    # print(f"pos_t {pos_t}")
    # print(f"cross_p {cross_p}")
    # print(f"cros_c {cross_c}")

    x_t, y_t = multimed_nlay(cal_t, mm, pos_t)
    # print(f"x_t {x_t}, y_t {y_t}")

    pos_t = np.r_[x_t, y_t, pos_t[2]]
    pos = back_trans_point(pos_t, mm, cal.glass_par, cross_p, cross_c)

    dm = cal.ext_par['dm']
    origin = np.r_[cal.ext_par['x0'], cal.ext_par['y0'], cal.ext_par['z0']]

    deno = np.dot(dm[:,2], (pos - origin))
    if deno == 0:
        deno = 1

    x = (-cal.int_par['cc'] * np.dot(dm[:, 0], (pos - origin)) / deno)
    y = (-cal.int_par['cc'] * np.dot(dm[:, 1], (pos - origin)) / deno)

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
) -> Tuple[np.float64, np.float64]:
    """Estimate metric coordinates in image space (mm)."""
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
    out = np.empty((orig_pos.shape[0], 2), dtype=np.float64)

    for i, row in enumerate(orig_pos):
        out[i, 0], out[i, 1] = img_coord(row, cal, mm)

    return out
