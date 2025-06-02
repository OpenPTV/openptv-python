from pathlib import Path
from typing import List

import numpy as np

from .calibration import Calibration
from .constants import POS_INF, PT_UNUSED, SORTGRID_EPS
from .epi import Coord3d_dtype
from .imgcoord import img_coord
from .parameters import ControlPar
from .tracking_frame_buf import Target
from .trafo import metric_to_pixel


def sortgrid(
    cal: Calibration,
    cpar: ControlPar,
    nfix: int,
    fix: np.ndarray,
    eps: float,
    pix: List[Target],
) -> List[Target]:
    """Sorts the grid points according to the image coordinates.

    /* sortgrid () is sorting detected target points by back-projection. Three dimensional
    positions of the dots on the calibration target are provided with the known IDs. The
    points are back-projected onto the image space and the nearest neighbour dots
    identified by image processing routines are selected and sorted according to the
    pre-defined IDs. The one to one correspondence provides a data on which the
    calibration process is based on. The nearest neighbour search is a primitive
    minimum distance search within a pre-defined radius (default = 10) read from the
    `sortgrid.par` parameter file (radius is given in pixels).

    Arguments:
    ---------
    Calibration: calibration parameters
    Control: control parameters
    nfix is the integer number of points in the calibration text file
    vec3d fix[] structure 3d positions and integer identification pointers of
    the calibration target points in the calibration file
    num is the number of detected (by image processing) dots on the calibration image

    Output:
    target sorted_pix[] is the array of targets or detected dots that have an ID (pnr),
    pixel position, size of the dot, sum of grey values and another identification (tnr)
    the pnr pointer is the row number of the dot in the calibration block file

    """
    sorted_pix = [Target() for _ in range(nfix)]

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


def nearest_neighbour_pix(pix: List[Target], x: float, y: float, eps: float):
    """Find the nearest neighbour pixel to the given point.

    Args:
    ----
        pix: array of point objects of size (num,), where each point object has x and y attributes.
        num: number of points in pix.
        x: x-coordinate of the point to find the nearest neighbour for.
        y: y-coordinate of the point to find the nearest neighbour for.
        eps: radius of the search region around the point.

    Returns
    -------
        pnr: index of the nearest neighbour pixel in pix. If no pixel is
             found within the search region, PT_UNUSED is returned.
    """
    pnr = PT_UNUSED
    dmin = POS_INF
    xmin, xmax, ymin, ymax = x - eps, x + eps, y - eps, y + eps

    for count, t in enumerate(pix):
        if ymin < t.y < ymax and xmin < t.x < xmax:
            d = np.sqrt((x - t.x) ** 2 + (y - t.y) ** 2)
            if d < dmin:
                dmin = d
                pnr = count

    return pnr

    # indices = np.where((pix['y'] > ymin) & (pix['y'] < ymax) & (pix['x'] > xmin) & (pix['x'] < xmax))[0]

    # if indices.size > 0:
    #     dists = np.sqrt((pix['x'][indices] - x) ** 2 + (pix['y'][indices] - y) ** 2)
    #     min_index = np.argmin(dists)
    #     pnr = indices[min_index]

    # return pnr


def read_sortgrid_par(filename) -> int:
    """Read the parameters for the sortgrid function from a file."""
    try:
        with open(filename, "r", encoding="utf-8") as fpp:
            eps = int(fpp.readline())
    except IOError:
        print(f"Can't open sortgrid parameter file: {filename} using default value")
        # handle error
        eps = SORTGRID_EPS

    return eps


def read_calblock(filename: Path) -> np.recarray:  # List[Coord3d]:
    """
    Read the calibration block file into the structure of 3D positions and pointers.

    Args:
    ----
    - filename (str): path to the text file containing the calibration points.

    Returns
    -------
    - List of Coord3d: 3D positions and integer identification pointers of the calibration
      target points in the calibration file of class Coord3d. if fails, returns None
    """
    coords = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                values = line.strip().split()
                pnr = int(values[0])
                x = float(values[1])
                y = float(values[2])
                z = float(values[3])
                # coord = Coord3d(pnr, x, y, z)
                coord = np.array([(pnr, x, y, z)], dtype=Coord3d_dtype)
                coords.append(coord)
    except FileNotFoundError:
        print(f"Can't open calibration block file: {filename}")
        return np.recarray(0, dtype=Coord3d_dtype)
    except ValueError:
        print(f"Empty or badly formatted file: {filename}")
        return np.recarray(0, dtype=Coord3d_dtype)

    return np.array(coords).view(np.recarray)
