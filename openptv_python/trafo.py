"""Module for coordinate transformations."""
from typing import Tuple

import numpy as np
from numba import float64, int32, njit

from .calibration import Calibration
from .parameters import ControlPar


def pixel_to_metric(
    x_pixel: float, y_pixel: float, parameters: ControlPar
) -> Tuple[np.float64, np.float64]:
    """Convert pixel coordinates to metric coordinates.

    Arguments:
    ---------
    x_metric, y_metric (float): output metric coordinates.
    x_pixel, y_pixel (float): input pixel coordinates.
    parameters (ControlPar): control structure holding image and pixel sizes.
    """
    return fast_pixel_to_metric(
        x_pixel,
        y_pixel,
        parameters.imx,
        parameters.imy,
        parameters.pix_x,
        parameters.pix_y
    )


@njit
def fast_pixel_to_metric(x_pixel, y_pixel, imx, imy, pix_x, pix_y) -> Tuple[np.float64, np.float64]:
    """Convert pixel coordinates to metric coordinates."""
    x_metric = (x_pixel - float(imx) / 2.0) * pix_x
    y_metric = (float(imy) / 2.0 - y_pixel) * pix_y

    return (x_metric, y_metric)

@njit(float64[:,:](int32[:,:],int32,int32,float64,float64))
def arr_pixel_to_metric(pixel: np.ndarray,
                        imx: np.int32,
                        imy: np.int32,
                        pix_x: np.float64,
                        pix_y: np.float64) -> np.ndarray:
    """Convert pixel coordinates to metric coordinates.

    Arguments:
    ---------
    imx (float): image width in pixels.
    imy (float): image height in pixels.
    pix_x (float): pixel size in x-direction.
    pix_y (float): pixel size in y-direction.

    Returns
    -------
    metric (np.ndarray): output metric coordinates.

    """
    metric = np.empty_like(pixel, dtype=np.float64)
    metric[:, 0] = (pixel[:, 0] - imx / 2.0) * pix_x
    metric[:, 1] = (imy / 2.0 - pixel[:, 1]) * pix_y

    return metric


def metric_to_pixel(
    x_metric: np.float64, y_metric: np.float64, parameters: ControlPar
) -> Tuple[np.float64, np.float64]:
    """Convert metric coordinates to pixel coordinates.

    Arguments:
    ---------
    x_metric, y_metric (float): input metric coordinates.
    parameters (ControlPar): control structure holding image and pixel sizes.

    Returns
    -------
    x_pixel, y_pixel (float): output pixel coordinates.
    """
    return fast_metric_to_pixel(
        x_metric,
        y_metric,
        parameters.imx,
        parameters.imy,
        parameters.pix_x,
        parameters.pix_y
    )


@njit
def fast_metric_to_pixel(
    x_metric: np.float64,
    y_metric: np.float64,
    imx: np.int32,
    imy: np.int32,
    pix_x: np.float64,
    pix_y: np.float64
) -> Tuple[np.float64, np.float64]:
    """Convert metric coordinates to pixel coordinates."""
    x_pixel = (x_metric / pix_x) + (imx / 2.0)
    y_pixel = (imy / 2.0) - (y_metric / pix_y)

    return x_pixel, y_pixel


def arr_metric_to_pixel(metric: np.ndarray,
                        parameters: ControlPar) -> np.ndarray:
    """Convert an array of metric coordinates to pixel coordinates.

    Arguments:
    ---------
    metric (np.ndarray): input array of metric coordinates.
    parameters (ControlPar): control structure holding image and pixel sizes.

    Returns
    -------
    pixel (np.ndarray): output array of pixel coordinates.
    """
    metric = np.atleast_2d(metric)

    return fast_arr_metric_to_pixel(
        metric,
        parameters.imx,
        parameters.imy,
        parameters.pix_x,
        parameters.pix_y
    )


@njit(float64[:,:](float64[:,:],int32,int32,float64,float64))
def fast_arr_metric_to_pixel(
    metric: np.ndarray,
    imx: np.int32,
    imy: np.int32,
    pix_x: np.float64,
    pix_y: np.float64,
) -> np.ndarray:
    """Convert an array of metric coordinates to pixel coordinates."""
    pixel = np.zeros_like(metric)
    pixel[:, 0] = (metric[:, 0] / pix_x) + (imx / 2.0)
    pixel[:, 1] = (imy / 2.0) - (metric[:, 1] / pix_y)

    return pixel

@njit(fastmath=True, cache=True, nogil=True)
def distort_brown_affine(x: np.float64,
                         y: np.float64,
                         ap: np.ndarray,
                         ) -> Tuple[np.float64, np.float64]:
    """Distort a point using the Brown affine model.

    ap: ap52_dtype: np.ndarray with fields k1, k2, k3, p1, p2, scx, she
        presented here as a np.array

    """
    if x == 0 and y == 0:
        return np.float64(0.0), np.float64(0.0)

    r = np.sqrt(x**2 + y**2)

    x += (
        x * (ap[0] * r**2 + ap[1] * r**4 + ap[2] * r**6)
        + ap[3] * (r**2 + 2 * x**2)
        + 2 * ap[4] * x * y
    )
    y += (
        y * (ap[0] * r**2 + ap[1] * r**4 + ap[2] * r**6)
        + ap[4] * (r**2 + 2 * y**2)
        + 2 * ap[3] * x * y
    )

    x1 = ap[5] * x - np.sin(ap[6]) * y
    y1 = np.cos(ap[6]) * y

    return x1, y1

# @njit(fastmath=True, cache=True, nogil=True)
def correct_brown_affine(
    x: np.float64,
    y: np.float64,
    ap: np.ndarray,
    tol: float = 1e-5,
    ) -> Tuple[np.float64, np.float64]:
    """Correct a distorted point using the Brown affine model."""
    r, rq, xq, yq = 0.0, 0.0, x, y
    itnum = 0

    if x == 0 and y == 0:
        return xq, yq

    rq = np.sqrt(x**2 + y**2)
    two_p1 = 2 * ap[3]
    two_p2 = 2 * ap[4]
    cos_she = np.cos(ap[6])
    sin_she = np.sin(ap[6])

    while True:
        r = rq
        common_term = (ap[0] * r**2 + ap[1] * r**4 + ap[2] * r**6)
        xq_common = xq * common_term
        yq_common = yq * common_term

        xq = (
            (x + yq * sin_she) / ap[5]
            - xq_common
            - ap[3] * (r**2 + 2 * xq**2)
            - two_p2 * xq * yq
        )

        yq = (
            y / cos_she
            - yq_common
            - ap[4] * (r**2 + 2 * yq**2)
            - two_p1 * xq * yq
        )

        rq = np.sqrt(xq**2 + yq**2)

        if rq > 1.2 * r:
            rq = 0.5 * r

        itnum += 1

        if itnum >= 201 or np.abs(rq - r) <= tol * r:
            break

    r = rq
    x1 = (
        (x + yq * sin_she) / ap[5]
        - xq * common_term
        - ap[3] * (r**2 + 2 * xq**2)
        - two_p2 * xq * yq
    )

    y1 = (
        y / cos_she
        - yq * common_term
        - ap[4] * (r**2 + 2 * yq**2)
        - two_p1 * xq * yq
    )

    return x1, y1

def flat_to_dist(
    flat_x: np.float64,
    flat_y: np.float64,
    cal: Calibration,
    ) -> Tuple[np.float64, np.float64]:
    """Convert flat-image coordinates to real-image coordinates.

    Make coordinates relative to sensor center rather than primary point
    image coordinates, because distortion formula assumes it, [1] p.180.
    """
    # print(f"flat_x {flat_x}, flat_y {flat_y}")
    # print(f"cal.int {cal.int_par['xh']}, {cal.int_par['yh']}")
    flat_x += cal.int_par['xh'] # type: ignore
    flat_y += cal.int_par['yh'] # type: ignore

    # print(f"flat_x {flat_x}, flat_y {flat_y}")

    dist_x, dist_y = distort_brown_affine(flat_x, flat_y, cal.added_par)
    # print(f"dist_x {dist_x}, dist_y {dist_y}")

    return dist_x, dist_y


def dist_to_flat(dist_x: np.float64, dist_y: np.float64, cal: Calibration, tol: float = 1e-5):
    """Convert real-image coordinates to flat-image coordinates."""
    flat_x, flat_y = correct_brown_affine(dist_x, dist_y, cal.added_par, tol)
    flat_x -= cal.int_par['xh']
    flat_y -= cal.int_par['yh']
    return flat_x, flat_y
