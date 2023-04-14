from typing import Tuple

import numpy as np

from openptv_python.calibration import Calibration
from openptv_python.parameters import ControlPar

from .constants import POS_INF, PT_UNUSED
from .imgcoord import img_coord
from .tracking_frame_buf import TargetArray
from .trafo import metric_to_pixel


def sortgrid(
    cal: Calibration,
    cpar: ControlPar,
    nfix: int,
    fix: TargetArray,
    eps: float,
    pix: TargetArray,
) -> TargetArray:
    """Sorts the grid points according to the image coordinates."""
    sorted_pix = TargetArray(nfix)

    for i in range(nfix):
        sorted_pix[i].pnr = -999

    for i in range(nfix):
        xp, yp = img_coord(fix[i], cal, cpar.mm)

        calib_point = metric_to_pixel(xp, yp, cpar)

        if (
            (calib_point[0] > -eps)
            and (calib_point[1] > -eps)
            and (calib_point[0] < cpar.imx + eps)
            and (calib_point[1] < cpar.imy + eps)
        ):
            j = nearest_neighbour_pix(pix, calib_point[0], calib_point[1], eps)

            if j != -999:
                sorted_pix[i] = pix[j]
                sorted_pix[i].pnr = i

    return sorted_pix


def nearest_neighbour_pix(pix: TargetArray, x: float, y: float, eps: float):
    """Find the nearest neighbour pixel to the given point.

    Args:
    ----
        pix: array of point objects of size (num,), where each point object has x and y attributes.
        num: number of points in pix.
        x: x-coordinate of the point to find the nearest neighbour for.
        y: y-coordinate of the point to find the nearest neighbour for.
        eps: radius of the search region around the point.

    Returns:
    -------
        pnr: index of the nearest neighbour pixel in pix. If no pixel is
             found within the search region, PT_UNUSED is returned.
    """
    pnr = PT_UNUSED
    dmin = POS_INF
    xmin, xmax, ymin, ymax = x - eps, x + eps, y - eps, y + eps

    y_within_bounds = (pix.y > ymin) & (pix.y < ymax)
    x_within_bounds = (pix.x > xmin) & (pix.x < xmax)
    indices_within_bounds = np.nonzero(x_within_bounds & y_within_bounds)[0]

    for j in indices_within_bounds:
        d = np.sqrt((x - pix[j].x) ** 2 + (y - pix[j].y) ** 2)
        if d < dmin:
            dmin = d
            pnr = j

    return pnr


def read_sortgrid_par(filename) -> int:
    """Read the parameters for the sortgrid function from a file."""
    try:
        with open(filename, "r", encoding="utf-8") as fpp:
            eps = int(fpp.readline())
    except IOError:
        # handle error
        eps = None

    return eps


def read_calblock(filename: str) -> Tuple[np.ndarray, int]:
    """
    Read the calibration block file into the structure of 3D positions and pointers.

    Args:
    ----
    - filename (str): path to the text file containing the calibration points.

    Returns:
    -------
    - ndarray: an array of 3D positions and integer identification pointers of the calibration
      target points in the calibration file.
    - int: number of valid calibration points. If reading failed for any reason, returns None.
    """
    try:
        with open(filename, "r") as f:
            data = np.loadtxt(f, usecols=(1, 2, 3))
    except FileNotFoundError:
        print(f"Can't open calibration block file: {filename}")
        return None, None
    except ValueError:
        print(f"Empty or badly formatted file: {filename}")
        return None, None

    num_points = data.shape[0]
    ident_pointers = np.arange(num_points)
    cal_data = np.column_stack((data, ident_pointers))

    return cal_data, num_points
