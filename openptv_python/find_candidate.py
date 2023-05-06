from openptv_python.constants import MAXCAND
from .epi import Candidate, Coord2d
from .tracking_frame_buf import Target
from .trafo import correct_brown_affine
from .constants import MAXCAND
from .calibration import Calibration
from .parameters import ControlPar, VolumePar
from typing import List
import numpy as np


def find_candidate(crd: List[Coord2d], pix: List[Target], 
                   num: int, xa: float, ya: float, xb: float, 
                   yb: float, n: int, nx: int, ny: int, 
                   sumg: int, cand: List[Candidate], 
                   vpar: VolumePar, cpar: ControlPar, cal: Calibration):
    """find_candidate() is searching in the image space of the image all the 
    candidates around the epipolar line originating from another camera. It is 
    a binary search in an x-sorted coord-set, exploits shape information of the
    particles.

    Args:
        crd: A list of `coord_2d` objects. Each object corresponds to the corrected coordinates of a target in an image.
        pix: A list of `target` objects. Each object corresponds to the target information (size, grey value, etc.) of a target.
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

    Returns:
        The number of candidates found.
    """

    count = 0
    xmin = (-1) * cpar.pix_x * cpar.imx / 2
    xmax = cpar.pix_x * cpar.imx / 2
    ymin = (-1) * cpar.pix_y * cpar.imy / 2
    ymax = cpar.pix_y * cpar.imy / 2
    xmin -= cal.int_par.xh
    ymin -= cal.int_par.yh
    xmax -= cal.int_par.xh
    ymax -= cal.int_par.yh

    xmin, ymin = correct_brown_affine(xmin, ymin, cal.added_par)
    xmax, ymax = correct_brown_affine(xmax, ymax, cal.added_par)

    # line equation: y = m*x + b
    if xa == xb:
        xb += 1e-10

    m = (yb - ya) / (xb - xa)
    b = ya - m * xa

    if xa > xb:
        temp = xa
        xa = xb
        xb = temp
    if ya > yb:
        temp = ya
        ya = yb
        yb = temp

    # If epipolar line out of sensor area, give up.
    if xb <= xmin or xa >= xmax or yb <= ymin or ya >= ymax:
        return -1

    # Binary search for start point of candidate search.
    for j0 in range(num // 2, -1, -1):
        if crd[j0].x < (xa - vpar.eps0):
            j0 -= 1
        else:
            break

    # Due to truncation error we might shift to smaller x.
    j0 -= 12
    if j0 < 0:
        j0 = 0

    for j in range(j0, num):
        # Since the list is x-sorted, an out of x-bound candidate is after the
        # last possible candidate, so stop.
        if crd[j].x > xb + vpar.eps0:
            return count

        # Candidate should at the very least be in the epipolar search window
        # to be considered.
        if crd[j].y <= ya - vpar.eps0 or crd[j].y >= yb + vpar.eps0:
            continue
        if crd[j].x <= xa - vpar.eps0 or crd[j].x >= xb + vpar.eps0:
            continue

        # Only take candidates within a predefined distance from epipolar line.
        d = np.abs((crd[j].y - m * crd[j].x - b) / np.sqrt(m * m + 1))
        if d >= vpar.eps0:
            continue

        p2 = crd[j].pnr

        # Quality of each parameter is a ratio of the values of the
        # size n, nx, ny and sum of grey values sumg.
        qn = quality_ratio(n, pix[p2].n)
        qnx = quality_ratio(nx, pix[p2].nx)
        qny = quality_ratio(ny, pix[p2].ny)
        qsumg = quality_ratio(sumg, pix[p2].sumg)

        # Enforce minimum quality values and maximum candidates.
        if qn < vpar.cn or qnx < vpar.cnx or qny < vpar.cny or qsumg <= vpar.csumg:
            continue
        if count >= MAXCAND:
            print(f"More candidates than {MAXCAND}: {count}\n")
            return count

        # Empirical correlation coefficient from shape and brightness
        # parameters.
        corr = (4 * qsumg + 2 * qn + qnx + qny)

        # Prefer matches with brighter targets.
        corr *= (sumg + pix[p2].sumg)

        cand[count].pnr = j
        cand[count].tol = d
        cand[count].corr = corr
        count += 1

    return count


def quality_ratio(a, b):
    """ Return the ratio of the smaller to the larger of the two numbers."""
    return a / b if a < b else b / a
