"""Module for coordinate transformations."""
from typing import Tuple

import numpy as np

from .calibration import Calibration, ap_52
from .parameters import ControlPar


def pixel_to_metric(
    x_pixel: float, y_pixel: float, parameters: ControlPar
) -> Tuple[float, float]:
    """Convert pixel coordinates to metric coordinates.

    Arguments:
    ---------
    x_metric, y_metric (float): output metric coordinates.
    x_pixel, y_pixel (float): input pixel coordinates.
    parameters (ControlPar): control structure holding image and pixel sizes.
    """
    x_metric = (x_pixel - float(parameters.imx) / 2.0) * parameters.pix_x
    y_metric = (float(parameters.imy) / 2.0 - y_pixel) * parameters.pix_y

    return (x_metric, y_metric)


def arr_pixel_to_metric(pixel: np.ndarray, parameters: ControlPar) -> np.ndarray:
    """Convert pixel coordinates to metric coordinates.

    Arguments:
    ---------
    metric (np.ndarray): output metric coordinates.
    pixel (np.ndarray): input pixel coordinates.
    parameters (ControlPar): control structure holding image and pixel sizes.
    """
    pixel = np.atleast_2d(np.array(pixel))
    metric = np.empty_like(pixel)
    metric[:, 0] = (pixel[:, 0] - float(parameters.imx) / 2.0) * parameters.pix_x
    metric[:, 1] = (float(parameters.imy) / 2.0 - pixel[:, 1]) * parameters.pix_y

    return metric


def metric_to_pixel(
    x_metric: float, y_metric: float, parameters: ControlPar
) -> Tuple[float, float]:
    """Convert metric coordinates to pixel coordinates.

    Arguments:
    ---------
    x_metric, y_metric (float): input metric coordinates.
    parameters (ControlPar): control structure holding image and pixel sizes.

    Returns:
    -------
    x_pixel, y_pixel (float): output pixel coordinates.
    """
    x_pixel = (x_metric / parameters.pix_x) + (float(parameters.imx) / 2.0)
    y_pixel = (float(parameters.imy) / 2.0) - (y_metric / parameters.pix_y)

    return x_pixel, y_pixel


def arr_metric_to_pixel(metric: np.ndarray, parameters: ControlPar) -> np.ndarray:
    """Convert an array of metric coordinates to pixel coordinates.

    Arguments:
    ---------
    metric (np.ndarray): input array of metric coordinates.
    parameters (ControlPar): control structure holding image and pixel sizes.

    Returns:
    -------
    pixel (np.ndarray): output array of pixel coordinates.
    """
    pixel = np.zeros_like(metric)
    pixel[:, 0] = (metric[:, 0] / parameters.pix_x) + (float(parameters.imx) / 2.0)
    pixel[:, 1] = (float(parameters.imy) / 2.0) - (metric[:, 1] / parameters.pix_y)

    return pixel


def distort_brown_affine(
    x: np.float64, y: np.float64, ap: ap_52
) -> Tuple[float, float]:
    """Distort a point using the Brown affine model."""
    if x == 0 and y == 0:
        return 0, 0

    r = np.sqrt(x**2 + y**2)

    x += (
        x * (ap.k1 * r**2 + ap.k2 * r**4 + ap.k3 * r**6)
        + ap.p1 * (r**2 + 2 * x**2)
        + 2 * ap.p2 * x * y
    )
    y += (
        y * (ap.k1 * r**2 + ap.k2 * r**4 + ap.k3 * r**6)
        + ap.p2 * (r**2 + 2 * y**2)
        + 2 * ap.p1 * x * y
    )
    x1 = ap.scx * x - np.sin(ap.she) * y
    y1 = np.cos(ap.she) * y

    return x1, y1


def correct_brown_affine(x, y, ap, tol=1e-5):
    """Correct a distorted point using the Brown affine model."""
    r, rq, xq, yq = 0.0, 0.0, x, y
    itnum = 0

    if x == 0 and y == 0:
        return xq, yq

    rq = np.sqrt(x**2 + y**2)

    while True:
        r = rq
        xq = (
            (x + yq * np.sin(ap.she)) / ap.scx
            - xq * (ap.k1 * r**2 + ap.k2 * r**4 + ap.k3 * r**6)
            - ap.p1 * (r**2 + 2 * xq**2)
            - 2 * ap.p2 * xq * yq
        )

        yq = (
            y / np.cos(ap.she)
            - yq * (ap.k1 * r**2 + ap.k2 * r**4 + ap.k3 * r**6)
            - ap.p2 * (r**2 + 2 * yq**2)
            - 2 * ap.p1 * xq * yq
        )

        rq = np.sqrt(xq**2 + yq**2)

        if rq > 1.2 * r:
            rq = 0.5 * r

        itnum += 1

        if itnum >= 201 or abs(rq - r) / r <= tol:
            break

    r = rq
    x1 = (
        (x + yq * np.sin(ap.she)) / ap.scx
        - xq * (ap.k1 * r**2 + ap.k2 * r**4 + ap.k3 * r**6)
        - ap.p1 * (r**2 + 2 * xq**2)
        - 2 * ap.p2 * xq * yq
    )

    y1 = (
        y / np.cos(ap.she)
        - yq * (ap.k1 * r**2 + ap.k2 * r**4 + ap.k3 * r**6)
        - ap.p2 * (r**2 + 2 * yq**2)
        - 2 * ap.p1 * xq * yq
    )

    return x1, y1


# def correct_brown_affine(x, y, ap, tol=1e-5):
#     """Correct a distorted point using the Brown affine model."""
#     xq, yq = x, y
#     rq = np.sqrt(x**2 + y**2)

#     def f(r, xq, yq):
#         xq = (
#             (x + yq * np.sin(ap.she)) / ap.scx
#             - xq * (ap.k1 * r**2 + ap.k2 * r**4 + ap.k3 * r**6)
#             - ap.p1 * (r**2 + 2 * xq**2)
#             - 2 * ap.p2 * xq * yq
#         )
#         yq = (
#             y / np.cos(ap.she)
#             - yq * (ap.k1 * r**2 + ap.k2 * r**4 + ap.k3 * r**6)
#             - ap.p2 * (r**2 + 2 * yq**2)
#             - 2 * ap.p1 * xq * yq
#         )
#         rq = np.sqrt(xq**2 + yq**2)
#         return rq - r

#     r = root_scalar(f, args=(xq, yq), bracket=[0, 2 * rq], maxiter=200, xtol=tol).root

#     xq = (
#         (x + yq * np.sin(ap.she)) / ap.scx
#         - xq * (ap.k1 * r**2 + ap.k2 * r**4 + ap.k3 * r**6)
#         - ap.p1 * (r**2 + 2 * xq**2)
#         - 2 * ap.p2 * xq * yq
#     )
#     yq = (
#         y / np.cos(ap.she)
#         - yq * (ap.k1 * r**2 + ap.k2 * r**4 + ap.k3 * r**6)
#         - ap.p2 * (r**2 + 2 * yq**2)
#         - 2 * ap.p1 * xq * yq
#     )

#     x1 = (
#         (x + yq * np.sin(ap.she)) / ap.scx
#         - xq * (ap.k1 * r**2 + ap.k2 * r**4 + ap.k3 * r**6)
#         - ap.p1 * (r**2 + 2 * xq**2)
#         - 2 * ap.p2 * xq * yq
#     )
#     y1 = (
#         y / np.cos(ap.she)
#         - yq * (ap.k1 * r**2 + ap.k2 * r**4 + ap.k3 * r**6)
#         - ap.p2 * (r**2 + 2 * yq**2)
#         - 2 * ap.p1 * xq * yq
#     )

#     return x1, y1


def flat_to_dist(flat_x: float, flat_y: float, cal: Calibration) -> Tuple[float, float]:
    """Convert flat-image coordinates to real-image coordinates.

    Make coordinates relative to sensor center rather than primary point
    image coordinates, because distortion formula assumes it, [1] p.180.
    """
    flat_x += cal.int_par.xh
    flat_y += cal.int_par.yh

    dist_x, dist_y = distort_brown_affine(flat_x, flat_y, cal.added_par)
    return dist_x, dist_y


def dist_to_flat(dist_x: float, dist_y: float, cal: Calibration, tol: float = 1e-5):
    """Convert real-image coordinates to flat-image coordinates."""
    flat_x, flat_y = correct_brown_affine(dist_x, dist_y, cal.added_par, tol)
    flat_x -= cal.int_par.xh
    flat_y -= cal.int_par.yh
    return flat_x, flat_y
