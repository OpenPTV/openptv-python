"""Epipolar geometry."""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .calibration import Calibration
from .constants import PT_UNUSED
from .imgcoord import flat_image_coord, img_coord
from .multimed import move_along_ray
from .parameters import ControlPar, MultimediaPar, VolumePar
from .ray_tracing import ray_tracing
from .trafo import dist_to_flat, metric_to_pixel, pixel_to_metric


@dataclass
class Candidate:
    """Candidate point in the second image."""

    pnr: int = field(default_factory=int)
    tol: float = field(default_factory=float)
    corr: float = field(default_factory=float)


@dataclass
class Coord2d:
    """2D coordinates in the image plane."""

    pnr: int = field(default=PT_UNUSED)
    x: float = field(default_factory=float)
    y: float = field(default_factory=float)


def sort_coord2d_x(crd: List[Coord2d]) -> List[Coord2d]:
    """Quicksort for coordinates by x ."""
    return sorted(crd, key=lambda p: p.x)


def sort_coord2d_y(crd: List[Coord2d]) -> List[Coord2d]:
    """Sort coordinates by y."""
    return sorted(crd, key=lambda p: p.y)


@dataclass
class Coord3d:
    """3D coordinates in the object space."""

    pnr: int = field(default=PT_UNUSED)
    x: float = field(default=0.0)
    y: float = field(default=0.0)
    z: float = field(default=0.0)


def epi_mm(xl, yl, cal1, cal2, mmp, vpar):
    """Return the end points of the epipolar line in the "second" camera.

    /*  epi_mm() takes a point in images space of one camera, positions of this
    and another camera and returns the epipolar line (in millimeter units)
    that corresponds to the point of interest in the another camera space.

    Arguments:
    ---------
    double xl, yl - position of the point on the origin camera's image space,
        in [mm].
    Calibration *cal1 - position of the origin camera
    Calibration *cal2 - position of camera on which the line is projected.
    mm_np *mmp - pointer to multimedia model of the experiment.
    volume_par *vpar - limits the search in 3D for the epipolar line

    Output:
    xmin,ymin and xmax,ymax - end points of the epipolar line in the "second"
        camera
    */

    Args:
    ----
        xl (_type_): _description_
        yl (_type_): _description_
        cal1 (_type_): _description_
        cal2 (_type_): _description_
        mmp (_type_): _description_
        vpar (_type_): _description_

    Returns:
    -------
        _type_: _description_
    """
    Zmin, Zmax = 0, 0
    pos, v, X = [0, 0, 0], [0, 0, 0], [0, 0, 0]

    pos, v = ray_tracing(xl, yl, cal1, mmp)

    # calculate min and max depth for position (valid only for one setup)
    Zmin = vpar.Zmin_lay[0] + (pos[0] - vpar.X_lay[0]) * (
        vpar.Zmin_lay[1] - vpar.Zmin_lay[0]
    ) / (vpar.X_lay[1] - vpar.X_lay[0])

    Zmax = vpar.Zmax_lay[0] + (pos[0] - vpar.X_lay[0]) * (
        vpar.Zmax_lay[1] - vpar.Zmax_lay[0]
    ) / (vpar.X_lay[1] - vpar.X_lay[0])

    X = move_along_ray(Zmin, pos, v)
    xmin, ymin = flat_image_coord(X, cal2, mmp)

    X = move_along_ray(Zmax, pos, v)
    xmax, ymax = flat_image_coord(X, cal2, mmp)

    return xmin, ymin, xmax, ymax


def epi_mm_2D(
    xl: float, yl: float, cal1: Calibration, mmp: MultimediaPar, vpar: VolumePar
) -> np.ndarray:
    """Return the position of the point in the 3D space.

        /*  epi_mm_2D() is a very degenerate case of the epipolar geometry use.
        It is valuable only for the case of a single camera with multi-media.
        It takes a point in images space of one (single) camera, positions of this
        camera and returns the position (in millimeter units) inside the 3D space
        that corresponds to the provided point of interest, limited in the middle of
        the 3D space, half-way between Zmin and Zmax. In purely 2D experiment, with
        an infinitely small light sheet thickness or on a flat surface, this will
        mean the point ray traced through the multi-media into the 3D space.

    Arguments:
    ---------
        double xl, yl - position of the point in the camera image space [mm].
        Calibration *cal1 - position of the camera
        mm_np *mmp - pointer to multimedia model of the experiment.
        volume_par *vpar - limits the search in 3D for the epipolar line.

        Output:
        vec3d out - 3D position of the point in the mid-plane between Zmin and
            Zmax, which are estimated using volume limits provided in vpar.
    */

    Args:
    ----
            xl (_type_): _description_
            yl (_type_): _description_
            cal1 (_type_): _description_
            mmp (_type_): _description_
            vpar (_type_): _description_
            out (_type_): _description_
    """
    pos, v = ray_tracing(xl, yl, cal1, mmp)

    Zmin = vpar.Zmin_lay[0] + (pos[0] - vpar.X_lay[0]) * (
        vpar.Zmin_lay[1] - vpar.Zmin_lay[0]
    ) / (vpar.X_lay[1] - vpar.X_lay[0])
    Zmax = vpar.Zmax_lay[0] + (pos[0] - vpar.X_lay[0]) * (
        vpar.Zmax_lay[1] - vpar.Zmax_lay[0]
    ) / (vpar.X_lay[1] - vpar.X_lay[0])

    out = move_along_ray(0.5 * (Zmin + Zmax), pos, v)
    return out


# def find_candidate(
#     crd: List[Coord2d], pix, num, xa, ya, xb, yb, n, nx, ny, sumg, cand, vpar, cpar, cal
# ):
#     """Find the candidate in the image space of the image all the candidates around the epipolar line.

#     originating from another camera. It is a binary search in an x-sorted coord-set,
#     exploits shape information of the particles.

#     /*  find_candidate() is searching in the image space of the image all the
#         candidates around the epipolar line originating from another camera. It is
#         a binary search in an x-sorted coord-set, exploits shape information of the
#         particles.

#     Arguments:
#     ---------
#         Coord2d *crd - points to an array of detected-points position information.
#             the points must be in flat-image (brown/affine corrected) coordinates
#             and sorted by their x coordinate, i.e. ``crd[i].x <= crd[i + 1].x``.
#         target *pix - array of target information (size, grey value, etc.)
#             structures. pix[j] describes the target corresponding to
#             (crd[...].pnr == j).
#         int num - number of particles in the image.
#         double xa, xb, ya, yb - end points of the epipolar line [mm].
#         int n, nx, ny - total, and per dimension pixel size of a typical target,
#             used to evaluate the quality of each candidate by comparing to typical.
#         int sumg - same, for the grey value.

#         Outputs:
#         candidate cand[] - array of candidate properties. The .pnr property of cand
#             points to an index in the x-sorted corrected detections array
#             (``crd``).

#         Extra configuration Arguments:
#         volume_par *vpar - observed volume dimensions.
#         control_par *cpar - general scene data s.a. image size.
#         Calibration *cal - position and other parameters on the camera seeing
#             the candidates.

#     Returns:
#     -------
#         int count - the number of selected candidates, length of cand array.
#             Negative if epipolar line out of sensor array.
#     */

#     Args:
#     ----
#             crd (_type_): _description_
#             pix (_type_): _description_
#             num (_type_): _description_
#             xa (_type_): _description_
#             ya (_type_): _description_
#             xb (_type_): _description_
#             yb (_type_): _description_
#             n (_type_): _description_
#             nx (_type_): _description_
#             ny (_type_): _description_
#             sumg (_type_): _description_
#             cand (_type_): _description_
#             vpar (_type_): _description_
#             cpar (_type_): _description_
#             cal (_type_): _description_

#     Returns:
#     -------
#             _type_: _description_
#     """
#     j = 0
#     j0 = 0
#     dj = 0
#     p2 = 0
#     count = 0
#     tol_band_width = vpar.eps0

#     # define sensor format for search interrupt
#     xmin = (-1) * cpar.pix_x * cpar.imx / 2
#     xmax = cpar.pix_x * cpar.imx / 2
#     ymin = (-1) * cpar.pix_y * cpar.imy / 2
#     ymax = cpar.pix_y * cpar.imy / 2
#     xmin -= cal.int_par.xh
#     ymin -= cal.int_par.yh
#     xmax -= cal.int_par.xh
#     ymax -= cal.int_par.yh
#     xmin, ymin = correct_brown_affine(xmin, ymin, cal.added_par)
#     xmax, ymax = correct_brown_affine(xmax, ymax, cal.added_par)

#     # line equation: y = m*x + b
#     if xa == xb:  # the line is a point or a vertical line in this camera
#         xb += 1e-10  # if we use xa += 1e-10, we always switch later

#     # equation of a line
#     m = (yb - ya) / (xb - xa)
#     b = ya - m * xa
#     if xa > xb:
#         temp = xa
#         xa = xb
#         xb = temp

#     if ya > yb:
#         temp = ya
#         ya = yb
#         yb = temp

#     # If epipolar line out of sensor area, give up.
#     if xb <= xmin or xa >= xmax or yb <= ymin or ya >= ymax:
#         return -1

#     # binary search for start point of candidate search
#     j0 = num // 2
#     dj = num // 4
#     while dj > 1:
#         if crd[j0].x < xa - tol_band_width:
#             j0 += dj
#         else:
#             j0 -= dj
#         dj //= 2

#     # due to truncation error we might shift to smaller x
#     j0 -= 12
#     if j0 < 0:
#         j0 = 0

#     # candidate search
#     for j in range(j0, num):
#         # Since the list is x-sorted, an out of x-bound candidate is after the
#         # last possible candidate, so stop.
#         if crd[j].x > xb + tol_band_width:
#             return count

#         # Candidate should at the very least be in the epipolar search window
#         # to be considred.
#         if crd[j].y <= ya - tol_band_width or crd[j].y >= yb + tol_band_width:
#             continue
#         if crd[j].x <= xa - tol_band_width or crd[j].x >= xb + tol_band_width:
#             continue

#         # Only take candidates within a predefined distance from epipolar line.
#         d = abs((crd[j].y - m * crd[j].x - b) / math.sqrt(m * m + 1))
#         if d >= tol_band_width:
#             continue

#         p2 = crd[j].pnr

#         # quality of each parameter is a ratio of the values of the size n, nx, ny
#         # and sum of grey values sumg
#         qn = quality_ratio(n, pix[p2].n)
#         qnx = quality_ratio(nx, pix[p2].nx)
#         qny = quality_ratio(ny, pix[p2].ny)
#         qsumg = quality_ratio(sumg, pix[p2].sumg)

#         # Enforce minimum quality values and maximum candidates
#         if qn < vpar.cn or qnx < vpar.cnx or qny < vpar.cny or qsumg <= vpar.csumg:
#             continue
#         if count >= MAXCAND:
#             print(f"More candidates than (maxcand): {count}")
#             return count

#         # empirical correlation coefficient from shape and brightness parameters
#         corr = 4 * qsumg + 2 * qn + qnx + qny

#         # prefer matches with brighter targets
#         corr *= float(sumg + pix[p2].sumg)

#         cand[count].pnr = j
#         cand[count].tol = d
#         cand[count].corr = corr
#         count += 1

#     return count


def epipolar_curve(
    image_point,
    origin_cam: Calibration,
    project_cam: Calibration,
    num_points: int,
    cparam: ControlPar,
    vparam: VolumePar,
) -> np.ndarray:
    """
    Get the points lying on the epipolar line from one camera to the other, on.

    the edges of the observed volume. Gives pixel coordinates.

    Assumes the same volume applies to all cameras.

    Arguments:
    ---------
    image_point - the 2D point on the image
        plane of the camera seeing the point. Distorted pixel coordinates.
    Calibration origin_cam - current position and other parameters of the
        camera seeing the point.
    Calibration project_cam - current position and other parameters of the
        cameraon which the line is projected.
    int num_points - the number of points to generate along the line. Minimum
        is 2 for both endpoints.
    ControlParams cparam - an object holding general control parameters.
    VolumeParams vparam - an object holding observed volume size parameters.

    Returns:
    -------
    line_points - (num_points,2) array with projection camera image coordinates
        of points lying on the ray stretching from the minimal Z coordinate of
        the observed volume to the maximal Z thereof, and connecting the camera
        with the image point on the origin camera.
    """
    # cdef:
    #     np.ndarray[ndim=2, dtype=np.float64_t] line_points
    #     vec3d vertex, direct, pos
    #     int pt_ix
    #     double Z
    #     double *x
    #     double *y
    #     double img_pt[2]

    line_points = np.empty((num_points, 2))

    # Move from distorted pixel coordinates to straight metric coordinates.
    x, y = pixel_to_metric(image_point[0], image_point[1], cparam)
    x, y = dist_to_flat(x, y, origin_cam, 0.00001)

    vertex, direct = ray_tracing(x, y, origin_cam, cparam.mm)

    for pt_ix, Z in enumerate(
        np.linspace(vparam.Zmin_lay[0], vparam.Zmax_lay[0], num_points)
    ):
        # x = line_points[pt_ix], 0)
        # y = <double *>np.PyArray_GETPTR2(line_points, pt_ix, 1)

        pos = move_along_ray(Z, vertex, direct)
        x, y = img_coord(pos, project_cam, cparam.mm)
        line_points[pt_ix, 0], line_points[pt_ix, 1] = metric_to_pixel(x, y, cparam)

    return line_points
