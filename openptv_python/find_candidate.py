import math
from typing import List

import numpy as np
from numba import float64, int32, njit

from .calibration import Calibration
from .constants import MAXCAND
from .epi import Candidate
from .parameters import ControlPar, VolumePar
from .trafo import correct_brown_affine


def find_candidate(
    crd: np.ndarray,
    pix: np.ndarray,
    num: int,
    xa: np.float64,
    ya: np.float64,
    xb: np.float64,
    yb: np.float64,
    n: np.int32,
    nx: np.int32,
    ny: np.int32,
    sumg: np.int32,
    vpar: VolumePar,
    cpar: ControlPar,
    cal: Calibration,
) -> np.ndarray:
    """Search in the image space of the image all the candidates around the epipolar line.

    originating from another camera. It is a binary search in an x-sorted coord-set,
    exploits shape information of the particles.

    Args:
    ----
        crd: A list of `coord_2d` objects. Each object corresponds to the corrected
                coordinates of a target in an image.
        pix: An array of `target` objects. Each object corresponds to the target information
                (size, grey value, etc.) of a target.
        num: The number of targets in the image.
        xa: The x-coordinate of the start point of the epipolar line.
        ya: The y-coordinate of the start point of the epipolar line.
        xb: The x-coordinate of the end point of the epipolar line.
        yb: The y-coordinate of the end point of the epipolar line., &xmax, &ymax
        n: The total size of a typical target.
        nx: The x-size of a typical target.
        ny: The y-size of a typical target.
        sumg: The sum of the grey values of a typical target.
        cand: A list of `candidate` objects. Each object corresponds to a candidate target.
        vpar: A `volume_par` object.
        cpar: A `control_par` object.
        cal: A `Calibration` object.

    Returns
    -------
        cand: list of candidates, or empty list if nothing is found
    """
    cand: List[np.ndarray] = []
    # cand: np.ndarray = np.ndarray((0,), dtype=Candidate_dtype)

    # The image space is the image plane of the camera. The image space is
    # given in millimeters of sensor size and the origin is in the center of the sensor.

    xmin = (-1) * cpar.pix_x * cpar.imx / 2
    xmax = cpar.pix_x * cpar.imx / 2
    ymin = (-1) * cpar.pix_y * cpar.imy / 2
    ymax = cpar.pix_y * cpar.imy / 2
    xmin -= cal.int_par['xh']
    ymin -= cal.int_par['yh']
    xmax -= cal.int_par['xh']
    ymax -= cal.int_par['yh']

    xmin, ymin = correct_brown_affine(np.float64(xmin), np.float64(ymin), cal.added_par)
    xmax, ymax = correct_brown_affine(np.float64(xmax), np.float64(ymax), cal.added_par)

    # line equation: y = m*x + b
    if xa == xb:  # the line is a point or a vertical line in this camera
        xb += 1e-10

    m = (yb - ya) / (xb - xa)
    b = ya - m * xa

    if xa > xb:
        xa, xb = xb, xa

    if ya > yb:
        ya, yb = yb, ya

    # If epipolar line out of sensor area, give up.
    if xb <= xmin or xa >= xmax or yb <= ymin or ya >= ymax:
        # return np.array(cand)
        return Candidate

    j0 = find_start_point(crd, num, xa, vpar)

    # print(f"j0: {j0}")

    for j in range(j0, num):
        # Since the list is x-sorted, an out of x-bound candidate is after the
        # last possible candidate, so stop.
        if crd[j]['x'] > xb + vpar.eps0:
            out = np.array(cand).flatten()
            return out  # type: ignore

        # Candidate should at the very least be in the epipolar search window
        # to be considered.
        if crd[j]['y'] <= ya - vpar.eps0 or crd[j]['y'] >= yb + vpar.eps0:
            continue
        if crd[j]['x'] <= xa - vpar.eps0 or crd[j]['x'] >= xb + vpar.eps0:
            continue

        # Only take candidates within a predefined distance from epipolar line.
        d = math.fabs((crd[j]['y'] - m * crd[j]['x'] - b) / math.sqrt(m * m + 1))
        if d >= vpar.eps0:
            continue

        p2 = crd[j]['pnr']

        # print(f"p2 = {p2}, d = {d}, eps0 = {vpar.eps0}")

        # Quality of each parameter is a ratio of the values of the
        # size n, nx, ny and sum of grey values sumg.
        qn = quality_ratio(n, pix[p2]['n'])
        qnx = quality_ratio(nx, pix[p2]['nx'])
        qny = quality_ratio(ny, pix[p2]['ny'])
        qsumg = quality_ratio(sumg, pix[p2]['sumg'])

        # Enforce minimum quality values and maximum candidates.
        if qn < vpar.cn or qnx < vpar.cnx or qny < vpar.cny or qsumg <= vpar.csumg:
            continue
        if len(cand) >= MAXCAND:
            print(f"Increase maximum number of candidates {len(cand)}\n")
            out = np.array(cand).flatten()
            return out # type: ignore

        # Empirical correlation coefficient from shape and brightness
        # parameters.
        corr = 4 * qsumg + 2 * qn + qnx + qny

        # Prefer matches with brighter targets.
        corr *= sumg + pix[p2]['sumg']

        # cand.append(Candidate(pnr=j, tol=d, corr=corr))
        cand.append(np.array([(j, d, corr)], dtype=Candidate.dtype)) # type: ignore



    out = np.array(cand).flatten()
        # print(f"appended: {cand[-1]}")

    return out # type: ignore


# def quality_ratio(a, b):
#     """ Return the ratio of the smaller to the larger of the two numbers."""
#     return a / b if a < b else b / a

# @njit(fastmath=True)
# @njit
def quality_ratio(a: np.int32, b: np.int32) -> np.float64:
    """
    Return the ratio of the smaller to the larger of the two integers.

    If either input is zero, return zero.
    """
    if a == 0 or b == 0:
        return np.float64(0.0)
    return np.min(np.r_[a, b]) / np.max(np.r_[a, b])

def find_start_point(crd: np.ndarray, num: int, xa: float, vpar: VolumePar) -> int:
    """Find the start point of the candidate search.

    Args:
    ----
        crd: A list of `coord_2d` objects. Each object corresponds to the
                corrected coordinates of a target in an image.
        num: The number of targets in the image.
        xa: The x-coordinate of the start point of the epipolar line.
        vpar: A `volume_par` object.

    Returns
    -------
        The start point of the candidate search.
    """
    # x = np.array([_['x'] for _ in crd])
    out = find_start_point_binary(crd['x'], num, xa, vpar.eps0)
    return out

@njit(int32(float64[:], int32, float64, float64))
def find_start_point_binary(x: np.ndarray, num: int, xa: float, eps0: float) -> int:
    # num = len(x)
    j0 = num // 2
    dj = num // 4

    while dj > 1:
        if x[j0] < (xa - eps0):
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    # print(f"j0 before: {j0}")

    # Due to truncation error we might shift to smaller x.
    start_point = j0 - 12  # why in C it's -12 ? -12 is a magic number
    if start_point < 0:
        start_point = 0

    return start_point
