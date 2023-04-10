"""Module for coordinate transformations."""
import math

from .calibration import Calibration, ap_52
from .parameters import control_par


def pixel_to_metric(
    x_pixel: float, y_pixel: float, parameters: control_par
) -> tuple(float, float):
    """Convert pixel coordinates to metric coordinates.

    Arguments:
    ---------
    x_metric, y_metric (float): output metric coordinates.
    x_pixel, y_pixel (float): input pixel coordinates.
    parameters (control_par): control structure holding image and pixel sizes.
    """
    x_metric = (x_pixel - float(parameters.imx) / 2.0) * parameters.pix_x
    y_metric = (float(parameters.imy) / 2.0 - y_pixel) * parameters.pix_y

    return x_metric, y_metric


def metric_to_pixel(
    x_metric: float, y_metric: float, parameters: control_par
) -> tuple(float, float):
    """Convert metric coordinates to pixel coordinates.

    Arguments:
    ---------
    x_pixel, y_pixel (float): input pixel coordinates.
    x_metric, y_metric (float): output metric coordinates.
    parameters (control_par): control structure holding image and pixel sizes.
    y_remap_mode (int): for use with interlaced cameras. Pass 0 for normal use,
        1 for odd lines and 2 for even lines.

    """
    x_pixel = (x_metric / parameters.pix_x) + (float(parameters.imx) / 2.0)
    y_pixel = (float(parameters.imy) / 2.0) - (y_metric / parameters.pix_y)

    return x_pixel, y_pixel


def distort_brown_affine(x: float, y: float, ap: ap_52) -> tuple(float, float):
    r = math.sqrt(x * x + y * y)
    if r != 0:
        x += (
            x * (ap.k1 * r * r + ap.k2 * r * r * r * r + ap.k3 * r * r * r * r * r * r)
            + ap.p1 * (r * r + 2 * x * x)
            + 2 * ap.p2 * x * y
        )
        y += (
            y * (ap.k1 * r * r + ap.k2 * r * r * r * r + ap.k3 * r * r * r * r * r * r)
            + ap.p2 * (r * r + 2 * y * y)
            + 2 * ap.p1 * x * y
        )
        x1 = ap.scx * x - math.sin(ap.she) * y
        y1 = math.cos(ap.she) * y

    return x1, y1


def correct_brown_affine(
    x: float, y: float, ap: ap_52, tol: float = 1e5
) -> tuple(float, float):
    """Solve the inverse problem iteratively.

        of what flat-image coordinate yielded the given distorted
        coordinates.

    Arguments:
    ---------
        double x, y - input metric shifted real-image coordinates.
        ap_52 ap - distortion parameters used in the distorting step.
        double *x1, *y1 - output metric shifted flat-image coordinates. Still needs
            unshifting to get pinhole-equivalent coordinates.
        tol - stop if the relative improvement in position between iterations is
            less than this value.
    */

    Args:
    ----
            x (_type_): _description_
            y (_type_): _description_
            ap (_type_): _description_

    Returns:
    -------
            _type_: _description_

    """
    r, rq, xq, yq = 0.0, 0.0, 0.0, 0.0
    itnum = 0

    if x == 0 and y == 0:
        return x, y

    # Initial guess for the flat point is the distorted point, assuming
    # distortion is small.
    rq = math.sqrt(x * x + y * y)
    xq, yq = x, y

    while True:
        r = rq
        xq = (
            (x + yq * math.sin(ap.she)) / ap.scx
            - xq
            * (ap.k1 * r * r + ap.k2 * r * r * r * r + ap.k3 * r * r * r * r * r * r)
            - ap.p1 * (r * r + 2 * xq * xq)
            - 2 * ap.p2 * xq * yq
        )
        yq = (
            y / math.cos(ap.she)
            - yq
            * (ap.k1 * r * r + ap.k2 * r * r * r * r + ap.k3 * r * r * r * r * r * r)
            - ap.p2 * (r * r + 2 * yq * yq)
            - 2 * ap.p1 * xq * yq
        )
        rq = math.sqrt(xq * xq + yq * yq)

        # Limit divergent iteration.
        if rq > 1.2 * r:
            rq = 0.5 * r

        itnum += 1
        if itnum >= 201 or abs(rq - r) / r <= tol:
            break

    # Final step uses the iteratively-found R and x, y to apply the exact
    # correction, equivalent to one more iteration.
    r = rq
    x1 = (
        (x + yq * math.sin(ap.she)) / ap.scx
        - xq * (ap.k1 * r * r + ap.k2 * r * r * r * r + ap.k3 * r * r * r * r * r * r)
        - ap.p1 * (r * r + 2 * xq * xq)
        - 2 * ap.p2 * xq * yq
    )
    y1 = (
        y / math.cos(ap.she)
        - yq * (ap.k1 * r * r + ap.k2 * r * r * r * r + ap.k3 * r * r * r * r * r * r)
        - ap.p2 * (r * r + 2 * yq * yq)
        - 2 * ap.p1 * xq * yq
    )

    return x1, y1


def flat_to_dist(flat_x: float, flat_y: float, cal: Calibration) -> tuple(float, float):
    """Convert flat-image coordinates to real-image coordinates.

    Make coordinates relative to sensor center rather than primary point
    image coordinates, because distortion formula assumes it, [1] p.180.
    """
    flat_x += cal.int_par.xh
    flat_y += cal.int_par.yh

    dist_x, dist_y = distort_brown_affine(flat_x, flat_y, cal.added_par)
    return dist_x, dist_y


def dist_to_flat(dist_x: float, dist_y: float, cal: Calibration, tol: float):
    """Attempt to restore metric flat-image positions from metric real-image coordinates.

    This is an inverse problem so some error is to be expected, but for small enough
    distortions it's bearable.
    """
    flat_x = dist_x
    flat_y = dist_y
    r = math.sqrt(flat_x**2 + flat_y**2)

    itnum = 0
    while True:
        xq = (
            (flat_x + flat_y * math.sin(cal.added_par.she)) / cal.added_par.scx
            - flat_x
            * (
                cal.added_par.k1 * r**2
                + cal.added_par.k2 * r**4
                + cal.added_par.k3 * r**6
            )
            - cal.added_par.p1 * (r**2 + 2 * flat_x**2)
            - 2 * cal.added_par.p2 * flat_x * flat_y
        )
        yq = (
            dist_y / math.cos(cal.added_par.she)
            - flat_y
            * (
                cal.added_par.k1 * r**2
                + cal.added_par.k2 * r**4
                + cal.added_par.k3 * r**6
            )
            - cal.added_par.p2 * (r**2 + 2 * flat_y**2)
            - 2 * cal.added_par.p1 * flat_x * flat_y
        )
        rq = math.sqrt(xq**2 + yq**2)

        # Limit divergent iteration
        if rq > 1.2 * r:
            rq = 0.5 * r

        itnum += 1

        # Check if we can stop iterating
        if itnum >= 201 or abs(rq - r) / r <= tol:
            break

        r = rq
        flat_x = xq
        flat_y = yq

    flat_x -= cal.int_par.xh
    flat_y -= cal.int_par.yh
    return flat_x, flat_y
