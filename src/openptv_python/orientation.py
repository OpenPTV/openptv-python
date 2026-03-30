"""Functions for the orientation of the camera."""

from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import scipy
from numba import njit, prange

from openptv_python.constants import COORD_UNUSED

from .calibration import Calibration
from ._native_compat import get_multimedia_par as _get_multimedia_par
from .constants import CONVERGENCE, IDT, NPAR, NUM_ITER, POS_INF
from .epi import epi_mm_2D
from .imgcoord import image_coordinates, img_coord

# from .lsqadj import ata, atl, matinv, matmul
from .parameters import ControlPar, MultimediaPar, OrientPar, VolumePar
from .ray_tracing import fast_ray_tracing, ray_tracing
from .sortgrid import sortgrid
from .tracking_frame_buf import Target
from .trafo import (
    arr_metric_to_pixel,
    correct_brown_affine,
    dist_to_flat,
    metric_to_pixel,
    pixel_to_metric,
)
from .vec_utils import unit_vector, vec_norm, vec_set


def is_singular(matrix):
    rank = np.linalg.matrix_rank(matrix)
    return rank < matrix.shape[0]


@njit
def skew_midpoint(
    vert1: np.ndarray, direct1: np.ndarray, vert2: np.ndarray, direct2: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Find the midpoint of the line segment that is the shortest distance."""
    perp_both = np.cross(direct1, direct2)
    scale = np.dot(perp_both, perp_both)

    sp_diff = vert2 - vert1

    temp = np.cross(sp_diff, direct2)
    on1 = vert1 + direct1 * np.dot(perp_both, temp) / scale

    temp = np.cross(sp_diff, direct1)
    on2 = vert2 + direct2 * np.dot(perp_both, temp) / scale

    scale = np.linalg.norm(on1 - on2)

    res = (on1 + on2) * 0.5
    return float(scale), res


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _multi_cam_point_positions_numba(
    targets: np.ndarray,
    distortion_matrices: np.ndarray,
    primary_points: np.ndarray,
    glass_vectors: np.ndarray,
    ccs: np.ndarray,
    distance_param: float,
    refractive_index_medium1: float,
    refractive_index_medium2: float,
    refractive_index_medium3: float,
    coord_unused: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate multi-camera point positions using compiled per-point loops."""
    num_targets = targets.shape[0]
    num_cams = targets.shape[1]
    res = np.empty((num_targets, 3), dtype=np.float64)
    rcm = np.empty(num_targets, dtype=np.float64)

    for pt in prange(num_targets):
        vertices = np.zeros((num_cams, 3), dtype=np.float64)
        directs = np.zeros((num_cams, 3), dtype=np.float64)
        point_tot = np.zeros(3, dtype=np.float64)
        num_used_pairs = 0
        dtot = 0.0

        for cam in range(num_cams):
            if targets[pt, cam, 0] == coord_unused:
                continue

            camera = np.empty(3, dtype=np.float64)
            camera[0] = targets[pt, cam, 0]
            camera[1] = targets[pt, cam, 1]
            camera[2] = -ccs[cam]
            vertices[cam], directs[cam] = fast_ray_tracing(
                camera,
                distortion_matrices[cam],
                primary_points[cam],
                glass_vectors[cam],
                distance_param,
                refractive_index_medium1,
                refractive_index_medium2,
                refractive_index_medium3,
            )

        for cam in range(num_cams):
            if targets[pt, cam, 0] == coord_unused:
                continue

            for pair in range(cam + 1, num_cams):
                if targets[pt, pair, 0] == coord_unused:
                    continue

                num_used_pairs += 1
                tmp, point = skew_midpoint(
                    vertices[cam],
                    directs[cam],
                    vertices[pair],
                    directs[pair],
                )
                dtot += tmp
                point_tot += point

        if num_used_pairs == 0:
            rcm[pt] = np.nan
            res[pt, :] = np.nan
        else:
            rcm[pt] = dtot / num_used_pairs
            res[pt, :] = point_tot / num_used_pairs

    return res, rcm


def point_position(
    targets: np.ndarray,
    num_cams: int,
    multimed_pars: MultimediaPar,
    cals: List[Calibration],
) -> Tuple[float, np.ndarray]:
    """
    Calculate an average 3D position implied by the rays.

    sent toward it from cameras through the image projections of the point.

    Arguments:
    ---------
    targets - for each camera, the 2D metric, flat, centred coordinates
        of the identified point projection.
    multimed_pars - multimedia parameters struct for ray tracing through
        several layers.
    cals - each camera's calibration object.

    Returns
    -------
    A tuple containing the ray convergence measure (an average of skew ray distance across all ray pairs)
    and the average 3D position vector.
    """
    # loop counters
    num_used_pairs = 0
    dtot = 0.0
    point_tot = np.array([0.0, 0.0, 0.0])

    vertices = np.zeros((num_cams, 3))
    directs = np.zeros((num_cams, 3))
    point = np.zeros(3)

    # Shoot rays from all cameras.
    for cam in range(num_cams):
        if targets[cam][0] != COORD_UNUSED:
            vertices[cam], directs[cam] = ray_tracing(
                targets[cam][0], targets[cam][1], cals[cam], multimed_pars
            )

    # Check intersection distance for each pair of rays and find position
    for cam in range(num_cams):
        if targets[cam][0] == COORD_UNUSED:
            continue

        for pair in range(cam + 1, num_cams):
            if targets[pair][0] == COORD_UNUSED:
                continue

            num_used_pairs += 1
            tmp, point = skew_midpoint(
                vertices[cam], directs[cam], vertices[pair], directs[pair]
            )
            dtot += tmp
            point_tot += point

    res = point_tot / num_used_pairs
    dtot /= num_used_pairs

    return float(dtot), res.astype(float)


def weighted_dumbbell_precision(
    targets: np.ndarray,
    multimed_pars: MultimediaPar,
    cals: List[Calibration],
    db_length: float,
    db_weight: float,
) -> float:
    """Calculate the weighted dumbbell precision of the current orientation."""
    res = [np.empty((3,)), np.empty((3,))]
    dtot = 0.0
    len_err_tot: float = 0.0

    num_targs = targets.shape[0]
    num_cams = targets.shape[1]

    for pt in range(num_targs):
        tmp, res[pt % 2] = point_position(targets[pt], num_cams, multimed_pars, cals)
        dtot += tmp

        if pt % 2 == 1:
            dist = float(np.linalg.norm(res[0] - res[1]))
            len_err_tot += 1.0 - float(
                db_length / dist if dist > db_length else dist / db_length
            )

    return float(dtot / num_targs + db_weight * len_err_tot / (0.5 * num_targs))


def num_deriv_exterior(
    cal: Calibration, cpar: ControlPar, dpos: float, dang: float, pos: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the partial numerical derivative of image coordinates of a.

    given 3D position, over each of the 6 exterior orientation parameters (3
    position parameters, 3 rotation angles).

    Arguments:
    ---------
    cal (Calibration): camera calibration object
    cpar (control_par): control parameters
    dpos (float): the step size for numerical differentiation for the metric variables
    dang (float): the step size for numerical differentiation for the angle variables.
    pos (vec3d): the current 3D position represented on the image.

    Returns
    -------
    Tuple of two lists: (x_ders, y_ders) respectively the derivatives of the x and y
    image coordinates as function of each of the orientation parameters.
    """
    var = ["x0", "y0", "z0", "omega", "phi", "kappa"]

    x_ders = np.zeros(len(var))
    y_ders = np.zeros(len(var))

    steps = [dpos, dpos, dpos, dang, dang, dang]

    # print(f"exterior = {cal.ext_par}")
    cal.update_rotation_matrix()
    xs, ys = img_coord(pos, cal, _get_multimedia_par(cpar))
    # print(f"  xs = {xs}, ys = {ys}")

    for pd in range(6):
        cal.increment_attribute(var[pd], steps[pd])
        # print(f"exterior = {cal.ext_par}")
        if pd > 2:
            cal.update_rotation_matrix()

        xpd, ypd = img_coord(pos, cal, _get_multimedia_par(cpar))
        # print(f" xpd = {xpd}, ypd = {ypd}")
        x_ders[pd] = (xpd - xs) / steps[pd]
        y_ders[pd] = (ypd - ys) / steps[pd]

        # print(f"   x_ders[{pd}] = {x_ders[pd]}, y_ders[{pd}] = {y_ders[pd]}")

        cal.increment_attribute(var[pd], -steps[pd])
        # print(f"exterior = {cal.ext_par}")

    cal.update_rotation_matrix()

    return (x_ders, y_ders)


def orient(
    cal: Calibration,
    cpar: ControlPar,
    nfix: int,
    fix: np.ndarray,
    pix: List[Target],
    flags: OrientPar,
    sigmabeta: np.ndarray,
    dm: float = 0.000001,
    drad: float = 0.000001,
) -> Optional[np.ndarray]:
    """Calculate orientation of the camera, updating its calibration.

    structure using the definitions and algorithms well described in [1].

    Arguments:
    ---------
    cal_in : Calibration object
        camera calibration object
    cpar : control_par object
        control parameters
    nfix : int
        number of 3D known points
    fix : np.array of shape (nfix, 3)
        each of nfix items is one 3D position of known point on
        the calibration object.
    pix : np.array of shape (nfix,)
        image coordinates corresponding to each point in ``fix``.
        can be obtained from the set of detected 2D points using
        sortgrid(). The points which are associated with fix[] have real
        pointer (.pnr attribute), others have -999.
    flags : OrientPar object
        structure of all the flags of the parameters to be (un)changed, read
        from orient.par parameter file using read_orient_par(), defaults
        are zeros except for x_scale which is by default 1.
    sigmabeta : ndarray of shape (20,)
        array of deviations for each of the interior and exterior parameters
        and glass interface vector (19 in total).

    Output:
    cal_in : Calibration object
        if the orientation routine converged, this structure is updated,
        otherwise, returned untouched. The routine works on a copy of the
        calibration structure, cal.
    sigmabeta : ndarray of shape (20,)
        array of deviations for each of the interior and exterior parameters
        and glass interface vector (19 in total).
    resi : ndarray of shape (maxsize,)
        On success, a pointer to an array of residuals. For each observation
        point i = 0..n-1, residual 2*i is the Gauss-Markof residual for the x
        coordinate and residual 2*i + 1 is for the y. Then come 10 cells with
        the delta between initial guess and final solution for internal and
        distortion parameters, which are also part of the G-M model and
        described in it. On failure returns None.

    Returns
    -------
    resi : ndarray of shape (maxsize,) or None
        On success, a pointer to an array of residuals. For each observation
        point i = 0..n-1, residual 2*i is the Gauss-Markof residual for the x
        coordinate and residual 2*i + 1 is for the y. Then come 10 cells with
        the delta between initial guess and final solution for internal and
        distortion parameters, which are also part of the G-M model and
        described in it. On failure returns None.
    """
    maxsize = nfix * 2 + IDT

    # dm: float = 0.000001
    # drad: float = 0.0000001

    # P, y, yh, Xbeta, resi are arrays of double
    P = np.ones(maxsize, dtype=float)
    y = np.zeros(maxsize, dtype=float)
    yh = np.zeros(maxsize, dtype=float)
    # Xbeta = np.zeros(maxsize, dtype=float)
    # resi = np.zeros(maxsize, dtype=float)

    # # X and Xh are arrays of double arrays
    X = np.zeros((maxsize, NPAR), dtype=float)
    Xh = np.zeros((maxsize, NPAR), dtype=float)
    beta = np.zeros(NPAR, dtype=float)
    n_obs = 0

    # sigmabeta = np.zeros(NPAR,)

    if flags.interfflag:
        numbers = 18
    else:
        numbers = 16

    glass_dir = cal.glass_par
    nGl = vec_norm(glass_dir)

    e1_x = 2 * cal.glass_par[2] - 3 * cal.glass_par[0]
    e1_y = 3 * cal.glass_par[0] - 1 * cal.glass_par[2]
    e1_z = 1 * cal.glass_par[1] - 2 * cal.glass_par[1]
    tmp_vec = vec_set(e1_x, e1_y, e1_z)
    e1 = unit_vector(tmp_vec)

    e2_x = e1_y * cal.glass_par[2] - e1_z * cal.glass_par[0]
    e2_y = e1_z * cal.glass_par[0] - e1_x * cal.glass_par[2]
    e2_z = e1_x * cal.glass_par[1] - e1_y * cal.glass_par[1]
    tmp_vec = vec_set(e2_x, e2_y, e2_z)
    e2 = unit_vector(tmp_vec)

    # al = 0
    # be = 0
    # ga = 0

    # init identities
    ident = [
        cal.int_par.cc,
        cal.int_par.xh,
        cal.int_par.yh,
        cal.added_par[0],
        cal.added_par[1],
        cal.added_par[2],
        cal.added_par[3],
        cal.added_par[4],
        cal.added_par[5],
        cal.added_par[6],
    ]

    # backup for changing back and forth
    safety_x = cal.glass_par[0]
    safety_y = cal.glass_par[1]
    safety_z = cal.glass_par[2]

    itnum = 0
    stopflag = False
    n = 0
    while not (stopflag or itnum >= NUM_ITER):
        itnum += 1
        n = 0
        for i in range(nfix):
            if pix[i].pnr != i:  # we need to check this point here
                continue

            if flags.useflag == 1 and i % 2 == 0:
                continue
            elif flags.useflag == 2 and i % 2 != 0:
                continue
            elif flags.useflag == 3 and i % 3 == 0:
                continue

            # get metric flat-image coordinates of the detected point
            xc, yc = pixel_to_metric(pix[i].x, pix[i].y, cpar)
            xc, yc = correct_brown_affine(xc, yc, cal.added_par)

            # Projected 2D position on sensor of corresponding known point
            cal.update_rotation_matrix()
            xp, yp = img_coord(fix[i], cal, _get_multimedia_par(cpar))

            # derivatives of distortion parameters
            r = np.sqrt(xp * xp + yp * yp)

            X[n][7] = cal.added_par[5]  # cal.added_par[5]
            X[n + 1][7] = np.sin(cal.added_par[6])  # np.sin(cal.added_par[6])

            X[n][8] = 0
            X[n + 1][8] = 1

            X[n][9] = cal.added_par[5] * xp * r * r
            X[n + 1][9] = yp * r * r

            X[n][10] = cal.added_par[5] * xp * pow(r, 4)
            X[n + 1][10] = yp * pow(r, 4)

            X[n][11] = cal.added_par[5] * xp * pow(r, 6)
            X[n + 1][11] = yp * pow(r, 6)

            X[n][12] = cal.added_par[5] * (2 * xp * xp + r * r)
            X[n + 1][12] = 2 * xp * yp

            X[n][13] = 2 * cal.added_par[5] * xp * yp
            X[n + 1][13] = 2 * yp * yp + r * r

            qq = cal.added_par[0] * r * r
            qq += cal.added_par[1] * pow(r, 4)
            qq += cal.added_par[2] * pow(r, 6)
            qq += 1
            X[n][14] = (
                xp * qq
                + cal.added_par[3] * (r * r + 2 * xp * xp)
                + 2 * cal.added_par[4] * xp * yp
            )
            X[n + 1][14] = 0

            X[n][15] = -np.cos(cal.added_par[6]) * yp
            X[n + 1][15] = -np.sin(cal.added_par[6]) * yp

            # numeric derivatives of projection coordinates over external parameters,
            # 3D position and the angles
            X[n][:6], X[n + 1][:6] = num_deriv_exterior(cal, cpar, dm, drad, fix[i])

            # Num. deriv. of projection coords over sensor distance from PP
            cal.int_par.cc += dm
            cal.update_rotation_matrix()
            xpd, ypd = img_coord(fix[i], cal, _get_multimedia_par(cpar))
            X[n][6] = (xpd - xp) / dm
            X[n + 1][6] = (ypd - yp) / dm
            # for i in range(len(fix)):
            #     dm = 0.0001
            #     xp, yp = 0.0, 0.0
            #     xc, yc = fix[i][0], fix[i][1]
            #     al, be, ga = cal.alpha, cal.beta, cal.gamma
            #     safety_x, safety_y, safety_z = cal.glass_par[0], cal.glass_par[1], cal.glass_par[2]
            #     nGl = cal.glass_par.n / cal.air_par.n

            cal.int_par.cc -= dm

            # al += dm
            cal.glass_par[0] += e1[0] * nGl * dm
            cal.glass_par[1] += e1[1] * nGl * dm
            cal.glass_par[2] += e1[2] * nGl * dm
            xpd, ypd = img_coord(fix[i], cal, _get_multimedia_par(cpar))
            X[n][16] = (xpd - xp) / dm
            X[n + 1][16] = (ypd - yp) / dm
            # al -= dm
            cal.glass_par[0] = safety_x
            cal.glass_par[1] = safety_y
            cal.glass_par[2] = safety_z

            # be += dm
            cal.glass_par[0] += e2[0] * nGl * dm
            cal.glass_par[1] += e2[1] * nGl * dm
            cal.glass_par[2] += e2[2] * nGl * dm
            xpd, ypd = img_coord(fix[i], cal, _get_multimedia_par(cpar))
            X[n][17] = (xpd - xp) / dm
            X[n + 1][17] = (ypd - yp) / dm
            # be -= dm
            cal.glass_par[0] = safety_x
            cal.glass_par[1] = safety_y
            cal.glass_par[2] = safety_z

            # ga += dm
            cal.glass_par[0] += cal.glass_par[0] * nGl * dm
            cal.glass_par[1] += cal.glass_par[1] * nGl * dm
            cal.glass_par[2] += cal.glass_par[2] * nGl * dm
            xpd, ypd = img_coord(fix[i], cal, _get_multimedia_par(cpar))
            X[n][18] = (xpd - xp) / dm
            X[n + 1][18] = (ypd - yp) / dm
            # ga -= dm
            cal.glass_par[0] = safety_x
            cal.glass_par[1] = safety_y
            cal.glass_par[2] = safety_z

            y[n] = xc - xp
            y[n + 1] = yc - yp

            n += 2
            # end of while loop

        # outside of the for loop
        n_obs = n
        # identities
        for i in range(IDT):
            X[n_obs + i][6 + i] = 1

        y[n_obs + 0] = ident[0] - cal.int_par.cc
        y[n_obs + 1] = ident[1] - cal.int_par.xh
        y[n_obs + 2] = ident[2] - cal.int_par.yh
        y[n_obs + 3] = ident[3] - cal.added_par[0]
        y[n_obs + 4] = ident[4] - cal.added_par[1]
        y[n_obs + 5] = ident[5] - cal.added_par[2]
        y[n_obs + 6] = ident[6] - cal.added_par[3]
        y[n_obs + 7] = ident[7] - cal.added_par[4]
        y[n_obs + 8] = ident[8] - cal.added_par[5]
        y[n_obs + 9] = ident[9] - cal.added_par[6]

        # weights
        for i in range(n_obs):
            P[i] = 1

        P[n_obs + 0] = POS_INF if not flags.ccflag else 1
        P[n_obs + 1] = POS_INF if not flags.xhflag else 1
        P[n_obs + 2] = POS_INF if not flags.yhflag else 1
        P[n_obs + 3] = POS_INF if not flags.k1flag else 1
        P[n_obs + 4] = POS_INF if not flags.k2flag else 1
        P[n_obs + 5] = POS_INF if not flags.k3flag else 1
        P[n_obs + 6] = POS_INF if not flags.p1flag else 1
        P[n_obs + 7] = POS_INF if not flags.p2flag else 1
        P[n_obs + 8] = POS_INF if not flags.scxflag else 1
        P[n_obs + 9] = POS_INF if not flags.sheflag else 1

        n_obs += IDT
        sumP = 0
        for i in range(n_obs):  # homogenize
            p = np.sqrt(P[i])
            for j in range(NPAR):
                Xh[i][j] = p * X[i][j]

            yh[i] = p * y[i]
            sumP += P[i]

        # Gauss Markoff Model - least square adjustment of redundant information
        # contained both in the spatial intersection and the resection
        # see [1], eq. 23

        # beta, residuals, rank, singular_values = np.linalg.lstsq(
        #     Xh[:, :numbers], yh, rcond=None
        # )

        beta, residuals, rank, singular_values = scipy.linalg.lstsq(
            Xh[:, :numbers], yh, lapack_driver="gelsy"
        )

        # Interpret the results
        # print(
        #     f"Coefficients (beta): {beta} \n \
        #         Residuals: {residuals} \n \
        #         singular_values: {singular_values} \n \
        #         rank: {rank} \n \
        #     "
        # )

        # stopflag
        stopflag = True
        for i in range(numbers):
            if abs(beta[i]) > CONVERGENCE:
                stopflag = False

        # check flags and update values
        if not flags.ccflag:
            beta[6] = 0.0
        if not flags.xhflag:
            beta[7] = 0.0
        if not flags.yhflag:
            beta[8] = 0.0
        if not flags.k1flag:
            beta[9] = 0.0
        if not flags.k2flag:
            beta[10] = 0.0
        if not flags.k3flag:
            beta[11] = 0.0
        if not flags.p1flag:
            beta[12] = 0.0
        if not flags.p2flag:
            beta[13] = 0.0
        if not flags.scxflag:
            beta[14] = 0.0
        if not flags.sheflag:
            beta[15] = 0.0

        cal.ext_par.x0 += beta[0]
        cal.ext_par.y0 += beta[1]
        cal.ext_par.z0 += beta[2]
        cal.ext_par.omega += beta[3]
        cal.ext_par.phi += beta[4]
        cal.ext_par.kappa += beta[5]
        cal.int_par.cc += beta[6]
        cal.int_par.xh += beta[7]
        cal.int_par.yh += beta[8]
        cal.added_par[0] += beta[9]
        cal.added_par[1] += beta[10]
        cal.added_par[2] += beta[11]
        cal.added_par[3] += beta[12]
        cal.added_par[4] += beta[13]
        cal.added_par[5] += beta[14]
        cal.added_par[6] += beta[15]

        if flags.interfflag:
            cal.glass_par[0] += e1[0] * nGl * beta[16]
            cal.glass_par[1] += e1[1] * nGl * beta[16]
            cal.glass_par[2] += e1[2] * nGl * beta[16]
            cal.glass_par[0] += e2[0] * nGl * beta[17]
            cal.glass_par[1] += e2[1] * nGl * beta[17]
            cal.glass_par[2] += e2[2] * nGl * beta[17]

    # def compute_residuals(X, y, beta, n_obs, numbers, NPAR, XPX, P, cal, cal_in, stopflag):
    # Xbeta = np.zeros((n_obs, 1))
    # resi = np.zeros((n_obs, 1))
    # sigmabeta = np.zeros((NPAR + 1, 1))
    # omega = 0

    # # Matrix multiplication
    # Xbeta = np.dot(X, beta)

    Xbeta = np.dot(X[:, :numbers], beta)

    resi = Xbeta - y
    omega = np.sum(resi * P * resi)
    sigmabeta[NPAR] = np.sqrt(omega / (n_obs - numbers))

    # if np.any(np.isnan(sigmabeta)):
    #     pdb.set_trace()

    # if np.any(np.isnan(X)):
    #     pdb.set_trace()

    XTX = np.dot(X[:, :numbers].T, X[:, :numbers])
    if is_singular(XTX):
        XPX = np.linalg.pinv(XTX)
    else:
        XPX = np.linalg.inv(XTX)

    # XPX = np.linalg.inv()

    # def invert_singular_matrix(m):
    # a, b = m.shape
    # if a != b:
    #     raise ValueError("Only square matrices are invertible.")
    # identity_matrix = np.eye(a, a)
    # return np.linalg.lstsq(m, identity_matrix)[0]

    # import pdb; pdb.set_trace()
    # for i in range(numbers):
    #     # print(f"{i=}, {np.sqrt(XPX[i][i]) = }")
    #     sigmabeta[i] = sigmabeta[NPAR] * np.sqrt(XPX[i][i])

    sigmabeta[:numbers] = sigmabeta[NPAR] * np.sqrt(np.diag(XPX))

    if stopflag:
        cal.update_rotation_matrix()
        return resi
    else:
        return None


def raw_orient(
    cal: Calibration,
    cpar: ControlPar,
    nfix: int,
    fix: np.ndarray,
    pix: List[Target],
    dm: float = 1e-4,
    drad: float = 1e-4,
) -> bool:
    """Calculate orientation of the camera, updating its calibration."""
    # the original C file says nfix is typically 4, but why X is 10 x 6 and not 8?

    # X = np.zeros((10, 6))
    # y = np.zeros(10)

    X = np.zeros((nfix * 2, 6))
    y = np.zeros(nfix * 2)

    # XPX = np.zeros((6, 6))
    # XPy = np.zeros(6)
    beta = np.zeros(6)
    pos = np.zeros(3)

    # cal.added_par[0] = 0
    # cal.added_par[1] = 0
    # cal.added_par[2] = 0
    # cal.added_par[3] = 0
    # cal.added_par[4] = 0
    # cal.added_par[5] = 1
    # cal.added_par[6] = 0
    cal.added_par = np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float64)
    itnum = 0
    stopflag = False

    while stopflag == 0 and itnum < 20:
        itnum += 1
        n = 0
        for i in range(nfix):
            xc, yc = pixel_to_metric(pix[i].x, pix[i].y, cpar)
            pos = fix[i]
            cal.update_rotation_matrix()
            xp, yp = img_coord(pos, cal, _get_multimedia_par(cpar))
            X[n], X[n + 1] = num_deriv_exterior(cal, cpar, dm, drad, pos)
            y[n] = xc - xp
            y[n + 1] = yc - yp
            n += 2

        # ChatGPT suggested to replace the following 4 lines
        # that performs the Gauss-Markoff model with the following
        # numpy based solution

        # ata(X, XPX, n, 6, 6)
        # matinv(XPX, 6, 6)
        # atl(XPy, X, y, n, 6, 6)
        # matmul(beta, XPX, XPy, 6, 6, 1, 6, 6)

        # Solve the linear system
        beta, residuals, rank, singular_values = scipy.linalg.lstsq(
            X, y, lapack_driver="gelsy"
        )  # , rcond=None)

        # Interpret the results
        # print("Coefficients (beta):", beta)
        # print("Residuals:", residuals)
        # print("rank:", rank)
        # print("singular_values:", singular_values)

        stopflag = True
        for i in range(6):
            if abs(beta[i]) > 0.1:
                stopflag = False

        cal.ext_par.x0 += beta[0]
        cal.ext_par.y0 += beta[1]
        cal.ext_par.z0 += beta[2]
        cal.ext_par.omega += beta[3]
        cal.ext_par.phi += beta[4]
        cal.ext_par.kappa += beta[5]

    if stopflag:
        cal.update_rotation_matrix()

    return stopflag


def read_man_ori_fix(calblock_filename, man_ori_filename, cam):
    """Read the manual orientation file."""
    fix4 = np.zeros((4, 3))
    fix = None
    num_fix = 0
    num_match = 0

    with open(man_ori_filename, "r", encoding="utf-8") as fpp:
        for i in range(cam):
            fpp.readline()
        nr = [int(x) for x in fpp.readline().split()]

    # read the id and positions of the fixed points, assign the pre-defined to fix4
    fix, num_fix = read_calblock(calblock_filename)
    if num_fix < 4:
        print(f"Too few points or incompatible file: {calblock_filename}")
        return None

    for pnr in range(num_fix):
        for i in range(4):
            if pnr == nr[i] - 1:
                fix4[i] = fix[pnr]
                num_match += 1
                break
        if num_match >= num_fix:
            break

    return fix4 if num_match == 4 else None


def read_calblock(filename):
    """Read the calibration block file."""
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        num_fix = int(lines[0])
        fix = np.zeros((num_fix, 3))
        for i, line in enumerate(lines[1:]):
            parts = line.split()
            fix[i] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    return fix, num_fix


def dumbbell_target_func(
    targets: np.ndarray,
    cparam: ControlPar,
    cals: List[Calibration],
    db_length: float,
    db_weight: float,
):
    """
    Wrap the epipolar convergence test.

    Arguments:
    ---------
    np.ndarray[ndim=3, dtype=pos_t] targets - (num_targets, num_cams, 2) array,
        containing the metric coordinates of each target on the image plane of
        each camera. Cameras must be in the same order for all targets.
    ControlParams cparam - needed for the parameters of the tank through which
        we see the targets.
    cals - a sequence of Calibration objects for each of the cameras, in the
        camera order of ``targets``.
    db_length - distance between two dumbbell targets.
    db_weight - weight of relative dumbbell size error in target function.
    """
    return weighted_dumbbell_precision(
        targets,
        cparam.mm,
        cals,
        db_length,
        db_weight,
    )


def external_calibration(
    cal: Calibration, ref_pts: np.ndarray, img_pts: np.ndarray, cparam: ControlPar
) -> bool:
    """
    Update the external calibration with results of raw orientation.

    the iterative process that adjust the initial guess' external parameters
    (position and angle of cameras) without internal or distortions.

    Arguments:
    ---------
    Calibration cal - position and other parameters of the camera.
    np.ndarray[ndim=2, dtype=pos_t] ref_pts - an (n,3) array, the 3D known
        positions of the select 2D points found on the image.
    np.ndarray[ndim=2, dtype=pos_t] img_pts - a selection of pixel coordinates
        of image points whose 3D position is known.
    ControlParams cparam - an object holding general control parameters.

    Returns
    -------
    True if iteration succeeded, false otherwise.
    """
    # cdef:
    #     target *targs
    #     vec3d *ref_coord

    # ref_pts = np.ascontiguousarray(ref_pts)
    # ref_coord = ref_pts.data

    # Convert pixel coords to metric coords:
    # targs = <target *>calloc(len(img_pts), sizeof(target))
    targs = [Target() for _ in img_pts]

    for ptx, pt in enumerate(img_pts):
        targs[ptx].x = pt[0]
        targs[ptx].y = pt[1]

    success = raw_orient(cal, cparam, len(ref_pts), ref_pts, targs)

    # free(targs);
    del targs

    return True if success else False


def full_calibration(
    cal: Calibration,
    ref_pts: np.ndarray,
    img_pts: np.ndarray,
    cparam: ControlPar,
    orient_par: OrientPar,
    dm: float = 1e-6,
    drad: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a full calibration, affecting all calibration structs.

    Arguments:
    ---------
    Calibration cal - current position and other parameters of the camera. Will
        be overwritten with new calibration if iteration succeeded, otherwise
        remains untouched.
    np.ndarray[ndim=2, dtype=pos_t] ref_pts - an (n,3) array, the 3D known
        positions of the select 2D points found on the image.
    TargetArray img_pts - detected points to match to known 3D positions.
        Must be sorted by matching ref point (as done by function
        ``match_detection_to_ref()``.
    ControlParams cparam - an object holding general control parameters.
    flags - a list whose members are the names of possible distortion
        parameters. Only parameter names present in the list will be used.
        Passing an empty list should be functionally equivalent to a raw
        calibration, though the code paths taken in C are different.

        The recognized flags are:
            'cc', 'xh', 'yh' - sensor position.
            'k1', 'k2', 'k3' - radial distortion.
            'p1', 'p2' - decentering
            'scale', 'shear' - affine transforms.

        This is what the underlying library uses a struct for, but come on.

    Returns
    -------
    ret - (r,2) array, the residuals in the x and y direction for r points used
        in orientation.
    used - r-length array, indices into target array of targets used.
    err_est - error estimation per calibration DOF. We

    Raises
    ------
    ValueError if iteration did not converge.
    """
    err_est = np.empty((NPAR + 1), dtype=np.float64)

    if isinstance(img_pts, np.ndarray):
        # convert numpy array to list of Target objects
        targs = [Target() for _ in img_pts]

        for ptx, pt in enumerate(img_pts):
            targs[ptx].x = pt[0]
            targs[ptx].y = pt[1]
            targs[ptx].pnr = ptx
    else:
        targs = img_pts

    residuals = orient(
        cal, cparam, len(ref_pts), ref_pts, targs, orient_par, err_est, dm=dm, drad=drad
    )

    # free(orip)

    if residuals is None:
        # free(residuals)
        print(f"Residuals = {residuals}")
        raise ValueError("Orientation iteration failed, need better setup.")

    ret = np.empty((len(img_pts), 2))
    used = np.empty(len(img_pts), dtype=np.int_)

    for ix, img_pt in enumerate(targs):
        ret[ix] = (residuals[2 * ix], residuals[2 * ix + 1])
        used[ix] = img_pt.pnr

    # free(residuals)
    return ret, used, err_est


def match_detection_to_ref(
    cal: Calibration,
    ref_pts: np.ndarray,
    img_pts: List[Target],
    cparam: ControlPar,
    eps: int = 25,
) -> List[Target]:
    """
    Create a TargetArray where the targets are those for which a point in the.

    projected reference is close enough to be considered a match, ordered by
    the order of corresponding references, with "empty targets" for detection
    points that have no match.

    Each target's pnr attribute is set to the index of the target in the array,
    which is also the index of the associated reference point in ref_pts.

    Arguments:
    ---------
    Calibration cal - position and other parameters of the camera.
    np.ndarray[ndim=2, dtype=pos_t] ref_pts - an (n,3) array, the 3D known
        positions of the selected 2D points found on the image.
    TargetArray img_pts - detected points to match to known 3D positions.
    ControlParams cparam - an object holding general control parameters.
    int eps - pixel radius of neighbourhood around detection to search for
        closest projection.

    Returns
    -------
    TargetArray holding the sorted targets.
    """
    # if len(img_pts) < len(ref_pts):
    #     # raise ValueError('Must have at least as many targets as ref. points.')
    #     print("Must have at least as many targets as ref. points.")
    #     pass

    # cdef:
    #     vec3d *ref_coord
    #     target *sorted_targs
    #     TargetArray t = TargetArray()

    # t = TargetArray(len(ref_pts))

    sorted_targs = sortgrid(cal, cparam, len(ref_pts), ref_pts, eps, img_pts)

    # t.set(sorted_targs)
    return sorted_targs


def point_positions(
    targets: np.ndarray,
    mm_par: MultimediaPar,
    cals: List[Calibration],
    vparam: VolumePar,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the 3D positions of the points given by their 2D projections.

    using one of the options:
    - for a single camera, uses single_cam_point_positions()
    - for multiple cameras, uses multi_cam_point_positions().

    Arguments:
    ---------
    np.ndarray[ndim=3, dtype=pos_t] targets - (num_targets, num_cams, 2) array,
        containing the metric coordinates of each target on the image plane of
        each camera. Cameras must be in the same order for all targets.
    MultimediaPar - needed for the parameters of the tank through which
        we see the targets.
    cals - a sequence of Calibration objects for each of the cameras, in the
        camera order of ``targets``.
    VolumeParams vparam - an object holding observed volume size parameters, needed
        for the single camera case only.

    Returns
    -------
    res - (n,3) array for n points represented by their targets.
    rcm - n-length array, the Ray Convergence Measure for eachpoint for multi camera
        option, or zeros for a single camera option
    """
    # cdef:
    #     np.ndarray[ndim=2, dtype=pos_t] res
    #     np.ndarray[ndim=1, dtype=pos_t] rcm

    if len(cals) == 1:
        res, rcm = single_cam_point_positions(targets, mm_par, cals, vparam)
    elif len(cals) > 1:
        res, rcm = multi_cam_point_positions(targets, mm_par, cals)
    else:
        raise ValueError("wrong number of cameras in point_positions")

    return res, rcm


def single_cam_point_positions(
    targets: np.ndarray,
    mm_par: MultimediaPar,
    cals: List[Calibration],
    vparam: VolumePar,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the 3D positions of the points from a single camera using.

    the 2D target positions given in metric coordinates.

    """
    # cdef:
    #     np.ndarray[ndim=2, dtype=pos_t] res
    #     np.ndarray[ndim=1, dtype=pos_t] rcm
    #     np.ndarray[ndim=2, dtype=pos_t] targ
    #     calibration ** calib = cal_list2arr(cals)
    #     int cam, num_cams

    # So we can address targets.data directly instead of get_ptr stuff:

    num_targets = targets.shape[0]
    # num_cams = targets.shape[1]
    res = np.empty((num_targets, 3))
    rcm = np.zeros(num_targets)

    for pt in range(num_targets):
        targ = targets[pt]
        res[pt, :] = epi_mm_2D(targ[0][0], targ[0][1], cals[0], mm_par, vparam)

    return res, rcm


def multi_cam_point_positions(
    targets: np.ndarray, mm_par: MultimediaPar, cals: List[Calibration]
):
    """
    Calculate the 3D positions of the points given by their 2D projections.

    Arguments:
    ---------
    np.ndarray[ndim=3, dtype=pos_t] targets - (num_targets, num_cams, 2) array,
        containing the metric coordinates of each target on the image plane of
        each camera. Cameras must be in the same order for all targets.
    ControlParams cparam - needed for the parameters of the tank through which
        we see the targets.
    cals - a sequence of Calibration objects for each of the cameras, in the
        camera order of ``targets``.

    Returns
    -------
    res - (n,3) array for n points represented by their targets.
    rcm - n-length array, the Ray Convergence Measure for eachpoint.
    """
    # cdef:
    #     np.ndarray[ndim=2, dtype=pos_t] res
    #     np.ndarray[ndim=1, dtype=pos_t] rcm
    #     np.ndarray[ndim=2, dtype=pos_t] targ
    #     calibration ** calib = cal_list2arr(cals)
    #     int cam, num_cams

    # So we can address targets.data directly instead of get_ptr stuff:

    distortion_matrices = np.ascontiguousarray(
        np.stack([cal.ext_par.dm for cal in cals]).astype(np.float64)
    )
    primary_points = np.ascontiguousarray(
        np.stack(
            [np.array([cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0]) for cal in cals]
        ).astype(np.float64)
    )
    glass_vectors = np.ascontiguousarray(
        np.stack([cal.glass_par for cal in cals]).astype(np.float64)
    )
    ccs = np.ascontiguousarray(
        np.array([cal.int_par.cc for cal in cals], dtype=np.float64)
    )

    return _multi_cam_point_positions_numba(
        np.ascontiguousarray(targets, dtype=np.float64),
        distortion_matrices,
        primary_points,
        glass_vectors,
        ccs,
        float(mm_par.d[0]),
        float(mm_par.n1),
        float(mm_par.n2[0]),
        float(mm_par.n3),
        float(COORD_UNUSED),
    )


def _clone_calibration(cal: Calibration) -> Calibration:
    """Create an isolated calibration copy for optimization updates."""
    return Calibration(
        ext_par=cal.ext_par.copy(),
        int_par=cal.int_par.copy(),
        glass_par=cal.glass_par.copy(),
        added_par=cal.added_par.copy(),
        mmlut=cal.mmlut,
        mmlut_data=cal.mmlut_data,
    )


def _bundle_optional_parameter_names(orient_par: OrientPar) -> List[str]:
    """Return optional calibration parameter names enabled for optimization."""
    names = []
    mapping = (
        ("ccflag", "cc"),
        ("xhflag", "xh"),
        ("yhflag", "yh"),
        ("k1flag", "k1"),
        ("k2flag", "k2"),
        ("k3flag", "k3"),
        ("p1flag", "p1"),
        ("p2flag", "p2"),
        ("scxflag", "scx"),
        ("sheflag", "she"),
    )
    for flag_name, param_name in mapping:
        if getattr(orient_par, flag_name):
            names.append(param_name)
    return names


def _get_optional_parameter(cal: Calibration, name: str) -> float:
    """Read one optional calibration parameter from a camera model."""
    if name == "cc":
        return float(cal.int_par.cc)
    if name == "xh":
        return float(cal.int_par.xh)
    if name == "yh":
        return float(cal.int_par.yh)

    added_map = {
        "k1": 0,
        "k2": 1,
        "k3": 2,
        "p1": 3,
        "p2": 4,
        "scx": 5,
        "she": 6,
    }
    if name not in added_map:
        raise ValueError(f"Unknown optional camera parameter: {name}")
    return float(cal.added_par[added_map[name]])


def _set_optional_parameter(cal: Calibration, name: str, value: float) -> None:
    """Write one optional calibration parameter into a camera model."""
    if name == "cc":
        cal.int_par.cc = value
        return
    if name == "xh":
        cal.int_par.xh = value
        return
    if name == "yh":
        cal.int_par.yh = value
        return

    added_map = {
        "k1": 0,
        "k2": 1,
        "k3": 2,
        "p1": 3,
        "p2": 4,
        "scx": 5,
        "she": 6,
    }
    if name not in added_map:
        raise ValueError(f"Unknown optional camera parameter: {name}")
    cal.added_par[added_map[name]] = value


def _glass_basis(glass_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build a stable tangent basis around the current glass vector."""
    norm = float(np.linalg.norm(glass_vec))
    if norm == 0.0:
        raise ValueError("Glass vector norm must be non-zero")

    normal = glass_vec / norm
    helper = (
        np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    e1 = np.cross(normal, helper)
    e1_norm = float(np.linalg.norm(e1))
    if e1_norm == 0.0:
        helper = np.array([0.0, 0.0, 1.0])
        e1 = np.cross(normal, helper)
        e1_norm = float(np.linalg.norm(e1))
    e1 /= e1_norm
    e2 = np.cross(normal, e1)
    e2 /= float(np.linalg.norm(e2))
    return e1, e2, norm


def _camera_parameter_block(
    cal: Calibration, optional_names: List[str], include_interface: bool
) -> np.ndarray:
    """Pack one camera block for bundle adjustment."""
    values = [
        float(cal.ext_par.x0),
        float(cal.ext_par.y0),
        float(cal.ext_par.z0),
        float(cal.ext_par.omega),
        float(cal.ext_par.phi),
        float(cal.ext_par.kappa),
    ]
    values.extend(_get_optional_parameter(cal, name) for name in optional_names)
    if include_interface:
        values.extend([0.0, 0.0])
    return np.asarray(values, dtype=np.float64)


def _apply_camera_parameter_block(
    cal: Calibration,
    block: np.ndarray,
    optional_names: List[str],
    include_interface: bool,
    base_glass: Optional[np.ndarray],
) -> None:
    """Apply one bundle-adjustment camera block to a calibration object."""
    cal.set_pos(block[:3])
    cal.set_angles(block[3:6])

    offset = 6
    for name in optional_names:
        _set_optional_parameter(cal, name, float(block[offset]))
        offset += 1

    if include_interface:
        if base_glass is None:
            raise ValueError("Base glass vector is required when interfflag is enabled")
        e1, e2, scale = _glass_basis(base_glass)
        cal.glass_par = base_glass + scale * (
            block[offset] * e1 + block[offset + 1] * e2
        )


def metric_observations_from_pixels(
    observed_pixels: np.ndarray,
    cals: List[Calibration],
    cpar: ControlPar,
) -> np.ndarray:
    """Convert pixel observations to flat metric coordinates for ray intersection."""
    num_points, num_cams, _ = observed_pixels.shape
    metric_obs = np.full((num_points, num_cams, 2), COORD_UNUSED, dtype=np.float64)

    for pt in range(num_points):
        for cam in range(num_cams):
            obs = observed_pixels[pt, cam]
            if not np.all(np.isfinite(obs)):
                continue
            x_metric, y_metric = pixel_to_metric(float(obs[0]), float(obs[1]), cpar)
            metric_obs[pt, cam] = dist_to_flat(x_metric, y_metric, cals[cam])

    return metric_obs


def initialize_bundle_adjustment_points(
    observed_pixels: np.ndarray,
    cals: List[Calibration],
    cpar: ControlPar,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate 3D starting points from pixel observations and current cameras."""
    metric_obs = metric_observations_from_pixels(observed_pixels, cals, cpar)
    num_points, num_cams, _ = metric_obs.shape
    points = np.empty((num_points, 3), dtype=np.float64)
    ray_convergence = np.empty(num_points, dtype=np.float64)

    for pt in range(num_points):
        if np.count_nonzero(metric_obs[pt, :, 0] != COORD_UNUSED) < 2:
            raise ValueError(
                "Each point must be observed by at least two cameras for bundle adjustment"
            )
        ray_convergence[pt], points[pt] = point_position(
            metric_obs[pt], num_cams, _get_multimedia_par(cpar), cals
        )

    return points, ray_convergence


def reprojection_errors(
    observed_pixels: np.ndarray,
    points_3d: np.ndarray,
    cals: List[Calibration],
    cpar: ControlPar,
) -> np.ndarray:
    """Return per-observation reprojection errors in pixels."""
    if observed_pixels.ndim != 3 or observed_pixels.shape[2] != 2:
        raise ValueError("observed_pixels must have shape (num_points, num_cams, 2)")
    if points_3d.shape != (observed_pixels.shape[0], 3):
        raise ValueError("points_3d must have shape (num_points, 3)")
    if observed_pixels.shape[1] != len(cals):
        raise ValueError(
            "Number of cameras in observations and calibrations must match"
        )

    residuals = np.full_like(observed_pixels, np.nan, dtype=np.float64)
    for pt in range(observed_pixels.shape[0]):
        for cam in range(observed_pixels.shape[1]):
            obs = observed_pixels[pt, cam]
            if not np.all(np.isfinite(obs)):
                continue
            x_metric, y_metric = img_coord(points_3d[pt], cals[cam], _get_multimedia_par(cpar))
            proj = metric_to_pixel(x_metric, y_metric, cpar)
            residuals[pt, cam] = np.asarray(proj, dtype=np.float64) - obs

    return residuals


def reprojection_rms(
    observed_pixels: np.ndarray,
    points_3d: np.ndarray,
    cals: List[Calibration],
    cpar: ControlPar,
) -> float:
    """Return the global RMS reprojection error in pixels."""
    residuals = reprojection_errors(observed_pixels, points_3d, cals, cpar)
    valid = np.isfinite(residuals)
    if not np.any(valid):
        raise ValueError("No valid observations available for reprojection RMS")
    return float(np.sqrt(np.mean(np.square(residuals[valid]))))


def reprojection_rms_per_camera(
    observed_pixels: np.ndarray,
    points_3d: np.ndarray,
    cals: List[Calibration],
    cpar: ControlPar,
) -> np.ndarray:
    """Return per-camera RMS reprojection errors in pixels."""
    residuals = reprojection_errors(observed_pixels, points_3d, cals, cpar)
    per_camera = np.full(observed_pixels.shape[1], np.nan, dtype=np.float64)
    for cam in range(observed_pixels.shape[1]):
        valid = np.isfinite(residuals[:, cam, :])
        if np.any(valid):
            per_camera[cam] = float(
                np.sqrt(np.mean(np.square(residuals[:, cam, :][valid])))
            )
    return per_camera


def mean_ray_convergence(
    observed_pixels: np.ndarray,
    cals: List[Calibration],
    cpar: ControlPar,
) -> float:
    """Return the mean ray convergence from triangulating observed pixels."""
    _, ray_convergence = initialize_bundle_adjustment_points(
        observed_pixels, cals, cpar
    )
    return float(np.mean(ray_convergence))


def _expand_parameter_limits(
    parameter_names: List[str],
    limits: Optional[Dict[str, Tuple[float, float]]],
    base_blocks: List[np.ndarray],
    repeat: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand per-parameter bounds for each optimized camera block."""
    lower = []
    upper = []

    if repeat <= 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    for block_index in range(repeat):
        for param_index, name in enumerate(parameter_names):
            if limits is None or name not in limits:
                lower.append(-np.inf)
                upper.append(np.inf)
                continue
            low_delta, high_delta = limits[name]
            base_value = base_blocks[block_index][param_index]
            lower_bound = base_value + low_delta
            upper_bound = base_value + high_delta
            if (
                np.isfinite(lower_bound)
                and np.isfinite(upper_bound)
                and lower_bound == upper_bound
            ):
                epsilon = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(lower_bound))
                lower_bound -= epsilon
                upper_bound += epsilon
            lower.append(lower_bound)
            upper.append(upper_bound)

    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _bundle_adjustment_x_scale(
    x_scale: Optional[float | Sequence[float] | Dict[str, float]],
    parameter_names: List[str],
    optimized_cam_indices: List[int],
    optimize_points: bool,
    num_points: int,
    total_parameters: int,
) -> Optional[float | np.ndarray]:
    """Normalize least_squares x_scale inputs to the packed BA parameter vector."""
    if x_scale is None:
        return None

    if np.isscalar(x_scale):
        value = float(cast(float, x_scale))
        if value <= 0:
            raise ValueError("x_scale must be strictly positive")
        return value

    if isinstance(x_scale, dict):
        values: List[float] = []
        for _cam in optimized_cam_indices:
            for name in parameter_names:
                value = float(x_scale.get(name, 1.0))
                if value <= 0:
                    raise ValueError("x_scale entries must be strictly positive")
                values.append(value)

        if optimize_points:
            point_scale = float(x_scale.get("points", x_scale.get("point", 1.0)))
            if point_scale <= 0:
                raise ValueError("point x_scale must be strictly positive")
            values.extend([point_scale] * (num_points * 3))

        return np.asarray(values, dtype=np.float64)

    sequence_values = np.asarray(x_scale, dtype=np.float64)
    if sequence_values.ndim != 1:
        raise ValueError("x_scale sequence must be one-dimensional")
    if sequence_values.size != total_parameters:
        raise ValueError(
            f"x_scale sequence must have length {total_parameters}, got {sequence_values.size}"
        )
    if np.any(sequence_values <= 0):
        raise ValueError("x_scale entries must be strictly positive")
    return sequence_values.copy()


def _bundle_adjustment_jacobian_sparsity(
    obs_mask: np.ndarray,
    num_cams: int,
    optimized_cam_indices: List[int],
    camera_block_size: int,
    point_offset: int,
    optimize_points: bool,
    active_priors: List[Tuple[int, int, float, float]],
    active_known_points: List[Tuple[int, np.ndarray, np.ndarray]],
) -> scipy.sparse.csr_matrix:
    """Build the Jacobian sparsity pattern for finite-difference bundle adjustment."""
    num_observation_residuals = int(np.count_nonzero(obs_mask) * 2)
    num_prior_residuals = len(active_priors)
    num_known_point_residuals = len(active_known_points) * 3
    total_residuals = (
        num_observation_residuals + num_prior_residuals + num_known_point_residuals
    )
    total_parameters = point_offset + (obs_mask.shape[0] * 3 if optimize_points else 0)

    sparsity = scipy.sparse.lil_matrix(
        (total_residuals, total_parameters),
        dtype=np.int8,
    )
    optimized_cam_lookup = {
        cam: cam_index for cam_index, cam in enumerate(optimized_cam_indices)
    }

    row = 0
    for point_index in range(obs_mask.shape[0]):
        point_start = point_offset + point_index * 3
        for cam in range(num_cams):
            if not obs_mask[point_index, cam]:
                continue

            cam_index = optimized_cam_lookup.get(cam)
            if cam_index is not None and camera_block_size > 0:
                cam_start = cam_index * camera_block_size
                sparsity[row : row + 2, cam_start : cam_start + camera_block_size] = 1

            if optimize_points:
                sparsity[row : row + 2, point_start : point_start + 3] = 1

            row += 2

    for cam_index, param_index, _prior_sigma, _base_value in active_priors:
        cam_start = cam_index * camera_block_size
        sparsity[row, cam_start + param_index] = 1
        row += 1

    if optimize_points:
        for point_index, _target, _point_sigma in active_known_points:
            point_start = point_offset + point_index * 3
            sparsity[row : row + 3, point_start : point_start + 3] = 1
            row += 3

    return sparsity.tocsr()


def _normalize_known_point_constraints(
    num_points: int,
    known_points: Optional[Dict[int, np.ndarray]],
    known_point_sigmas: Optional[float | np.ndarray],
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Normalize known-point priors into indexed target and sigma vectors."""
    if known_points is None:
        return []

    sigma_source: float | np.ndarray
    sigma_source = 1.0 if known_point_sigmas is None else known_point_sigmas
    sigma_array = np.asarray(sigma_source, dtype=np.float64)

    constraints = []
    for point_index in sorted(known_points):
        if point_index < 0 or point_index >= num_points:
            raise ValueError("known_points contains an out-of-range point index")

        target = np.asarray(known_points[point_index], dtype=np.float64)
        if target.shape != (3,):
            raise ValueError("Each known_points entry must have shape (3,)")

        if sigma_array.ndim == 0:
            sigma = np.full(3, float(sigma_array), dtype=np.float64)
        elif sigma_array.shape == (3,):
            sigma = sigma_array.copy()
        elif sigma_array.shape == (num_points, 3):
            sigma = sigma_array[point_index].copy()
        else:
            raise ValueError(
                "known_point_sigmas must be a scalar, shape (3,), or shape (num_points, 3)"
            )

        if np.any(sigma <= 0):
            raise ValueError("known_point_sigmas must be strictly positive")

        constraints.append((point_index, target, sigma))

    return constraints


def multi_camera_bundle_adjustment(
    observed_pixels: np.ndarray,
    cals: List[Calibration],
    cpar: ControlPar,
    orient_par: OrientPar,
    point_init: Optional[np.ndarray] = None,
    fix_first_camera: bool = True,
    fixed_camera_indices: Optional[List[int]] = None,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    method: str = "trf",
    prior_sigmas: Optional[Dict[str, float]] = None,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    max_nfev: Optional[int] = None,
    optimize_extrinsics: bool = True,
    optimize_points: bool = True,
    known_points: Optional[Dict[int, np.ndarray]] = None,
    known_point_sigmas: Optional[float | np.ndarray] = None,
    x_scale: Optional[float | Sequence[float] | Dict[str, float]] = None,
    ftol: Optional[float] = None,
    xtol: Optional[float] = None,
    gtol: Optional[float] = None,
) -> Tuple[List[Calibration], np.ndarray, scipy.optimize.OptimizeResult]:
    """Jointly refine multi-camera calibration and 3D points by reprojection error."""
    observed_pixels = np.asarray(observed_pixels, dtype=np.float64)
    if observed_pixels.ndim != 3 or observed_pixels.shape[2] != 2:
        raise ValueError("observed_pixels must have shape (num_points, num_cams, 2)")
    if len(cals) < 2:
        raise ValueError("Bundle adjustment requires at least two cameras")
    if observed_pixels.shape[1] != len(cals):
        raise ValueError(
            "Number of cameras in observations and calibrations must match"
        )

    num_points, num_cams, _ = observed_pixels.shape
    if num_points == 0:
        raise ValueError("At least one 3D point is required for bundle adjustment")

    obs_counts = np.count_nonzero(np.all(np.isfinite(observed_pixels), axis=2), axis=1)
    if np.any(obs_counts < 2):
        raise ValueError("Each point must be observed by at least two cameras")

    base_cals = [_clone_calibration(cal) for cal in cals]
    optional_names = _bundle_optional_parameter_names(orient_par)
    include_interface = bool(orient_par.interfflag)
    prior_sigmas = {} if prior_sigmas is None else prior_sigmas

    if fixed_camera_indices is None:
        fixed_camera_indices = [0] if fix_first_camera else []

    fixed_camera_indices = sorted(set(fixed_camera_indices))
    if any(cam < 0 or cam >= num_cams for cam in fixed_camera_indices):
        raise ValueError("fixed_camera_indices contains an out-of-range camera index")

    if optimize_extrinsics:
        optimized_cam_indices = [
            cam for cam in range(num_cams) if cam not in fixed_camera_indices
        ]
    else:
        optimized_cam_indices = list(range(num_cams))

    if (
        not optimize_extrinsics
        and not optional_names
        and not include_interface
        and not optimize_points
    ):
        raise ValueError("No camera parameters are enabled for optimization")

    if not optimize_points and not optimized_cam_indices:
        raise ValueError(
            "Bundle adjustment must optimize points or at least one camera"
        )
    if known_points is not None and not optimize_points:
        raise ValueError(
            "known_points constraints require optimize_points=True in the current implementation"
        )

    # Refining both 3D points and camera poses from image observations has a similarity
    # gauge. One fixed camera removes global translation/rotation, but scale still drifts
    # unless another camera is fixed or translation priors are applied.
    has_translation_priors = all(
        prior_sigmas.get(name, 0) > 0 for name in ("x0", "y0", "z0")
    )
    if (
        optimize_points
        and optimized_cam_indices
        and optimize_extrinsics
        and len(fixed_camera_indices) < 2
        and not has_translation_priors
    ):
        raise ValueError(
            "Bundle adjustment with free 3D points and only one fixed camera is scale-ambiguous. "
            "Fix at least two cameras via fixed_camera_indices or provide translation priors "
            "for x0, y0, and z0."
        )

    if point_init is None:
        points0, _ = initialize_bundle_adjustment_points(
            observed_pixels, base_cals, cpar
        )
    else:
        points0 = np.asarray(point_init, dtype=np.float64)
        if points0.shape != (num_points, 3):
            raise ValueError("point_init must have shape (num_points, 3)")

    base_camera_blocks = []
    base_glass_vectors: Dict[int, np.ndarray] = {}
    parameter_names = []
    if optimize_extrinsics:
        parameter_names.extend(["x0", "y0", "z0", "omega", "phi", "kappa"])
    parameter_names.extend(optional_names)
    if include_interface:
        parameter_names.extend(["glass_e1", "glass_e2"])

    initial_blocks = []
    for cam in optimized_cam_indices:
        initial_block = _camera_parameter_block(
            base_cals[cam], optional_names, include_interface
        )
        if not optimize_extrinsics:
            initial_block = initial_block[6:]
        initial_blocks.append(initial_block)
        base_camera_blocks.append(initial_block.copy())
        if include_interface:
            base_glass_vectors[cam] = base_cals[cam].glass_par.copy()

    camera_block_size = len(parameter_names)
    payloads = []
    if initial_blocks:
        payloads.extend(initial_blocks)
    if optimize_points:
        payloads.append(points0.ravel())
    if payloads:
        x0 = np.concatenate(payloads)
    else:
        x0 = np.empty(0, dtype=np.float64)

    point_offset = camera_block_size * len(optimized_cam_indices)
    obs_mask = np.all(np.isfinite(observed_pixels), axis=2)
    observation_point_indices = [
        np.flatnonzero(obs_mask[:, cam]) for cam in range(num_cams)
    ]

    active_priors = []
    for cam_index, _cam in enumerate(optimized_cam_indices):
        for param_index, (name, base_value) in enumerate(
            zip(parameter_names, base_camera_blocks[cam_index])
        ):
            sigma = prior_sigmas.get(name)
            if sigma is None or sigma <= 0:
                continue
            active_priors.append((cam_index, param_index, sigma, base_value))

    active_known_points = _normalize_known_point_constraints(
        num_points,
        known_points,
        known_point_sigmas,
    )

    camera_lower, camera_upper = _expand_parameter_limits(
        parameter_names,
        parameter_bounds,
        base_camera_blocks,
        len(optimized_cam_indices),
    )
    if optimize_points:
        point_lower = np.full(points0.size, -np.inf, dtype=np.float64)
        point_upper = np.full(points0.size, np.inf, dtype=np.float64)
        bounds = (
            np.concatenate([camera_lower, point_lower]),
            np.concatenate([camera_upper, point_upper]),
        )
    else:
        bounds = (camera_lower, camera_upper)

    normalized_x_scale = _bundle_adjustment_x_scale(
        x_scale,
        parameter_names,
        optimized_cam_indices,
        optimize_points,
        num_points,
        x0.size,
    )

    def unpack_parameters(params: np.ndarray) -> Tuple[List[Calibration], np.ndarray]:
        trial_cals = [_clone_calibration(cal) for cal in base_cals]
        offset = 0
        for cam in optimized_cam_indices:
            block = params[offset : offset + camera_block_size]
            if not optimize_extrinsics:
                base_pose = _camera_parameter_block(base_cals[cam], [], False)[:6]
                block = np.concatenate([base_pose, block])
            _apply_camera_parameter_block(
                trial_cals[cam],
                block,
                optional_names,
                include_interface,
                base_glass_vectors.get(cam),
            )
            offset += camera_block_size

        if optimize_points:
            points = params[point_offset:].reshape(num_points, 3)
        else:
            points = points0.copy()
        return trial_cals, points

    def residual_vector(params: np.ndarray) -> np.ndarray:
        trial_cals, points = unpack_parameters(params)
        residuals = np.empty(
            int(np.count_nonzero(obs_mask) * 2)
            + len(active_priors)
            + 3 * len(active_known_points)
        )
        row = 0

        for cam, point_indices in enumerate(observation_point_indices):
            if point_indices.size == 0:
                continue

            projected_pixels = arr_metric_to_pixel(
                image_coordinates(points[point_indices], trial_cals[cam], _get_multimedia_par(cpar)),
                cpar,
            )
            observed = observed_pixels[point_indices, cam, :]
            diffs = projected_pixels - observed
            residual_count = point_indices.size * 2
            residuals[row : row + residual_count : 2] = diffs[:, 0] / cpar.pix_x
            residuals[row + 1 : row + residual_count : 2] = diffs[:, 1] / cpar.pix_y
            row += residual_count

        for cam_index, param_index, sigma, base_value in active_priors:
            value = params[cam_index * camera_block_size + param_index]
            residuals[row] = (value - base_value) / sigma
            row += 1

        for point_index, target, point_sigma in active_known_points:
            residuals[row : row + 3] = (points[point_index] - target) / point_sigma
            row += 3

        return residuals

    initial_cals, initial_points = unpack_parameters(x0)
    initial_rms = reprojection_rms(observed_pixels, initial_points, initial_cals, cpar)
    initial_per_camera = reprojection_rms_per_camera(
        observed_pixels, initial_points, initial_cals, cpar
    )

    least_squares_kwargs = {
        "method": method,
        "loss": loss,
        "f_scale": f_scale,
        "max_nfev": max_nfev,
        "bounds": bounds,
    }
    if normalized_x_scale is not None:
        least_squares_kwargs["x_scale"] = normalized_x_scale
    if method != "lm" and x0.size > 0:
        least_squares_kwargs["jac_sparsity"] = _bundle_adjustment_jacobian_sparsity(
            obs_mask,
            num_cams,
            optimized_cam_indices,
            camera_block_size,
            point_offset,
            optimize_points,
            active_priors,
            active_known_points,
        )
    if ftol is not None:
        least_squares_kwargs["ftol"] = ftol
    if xtol is not None:
        least_squares_kwargs["xtol"] = xtol
    if gtol is not None:
        least_squares_kwargs["gtol"] = gtol

    result = scipy.optimize.least_squares(
        residual_vector,
        x0,
        **least_squares_kwargs,
    )

    refined_cals, refined_points = unpack_parameters(result.x)
    result["initial_reprojection_rms"] = initial_rms
    result["final_reprojection_rms"] = reprojection_rms(
        observed_pixels, refined_points, refined_cals, cpar
    )
    result["initial_reprojection_rms_per_camera"] = initial_per_camera
    result["final_reprojection_rms_per_camera"] = reprojection_rms_per_camera(
        observed_pixels, refined_points, refined_cals, cpar
    )
    result["optimized_camera_indices"] = optimized_cam_indices
    result["known_point_indices"] = [
        point_index for point_index, _, _ in active_known_points
    ]

    return refined_cals, refined_points, result


def guarded_two_step_bundle_adjustment(
    observed_pixels: np.ndarray,
    cals: List[Calibration],
    cpar: ControlPar,
    pose_orient_par: OrientPar,
    intrinsic_orient_par: OrientPar,
    *,
    point_init: Optional[np.ndarray] = None,
    fixed_camera_indices: Optional[List[int]] = None,
    pose_release_camera_order: Optional[List[int]] = None,
    pose_stage_ray_slack: float = 0.0,
    pose_prior_sigmas: Optional[Dict[str, float]] = None,
    pose_parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    pose_loss: str = "linear",
    pose_method: str = "trf",
    pose_max_nfev: Optional[int] = None,
    pose_x_scale: Optional[float | Sequence[float] | Dict[str, float]] = None,
    pose_stage_configs: Optional[Sequence[Dict[str, object]]] = None,
    intrinsic_prior_sigmas: Optional[Dict[str, float]] = None,
    intrinsic_parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    intrinsic_loss: str = "linear",
    intrinsic_method: str = "trf",
    intrinsic_max_nfev: Optional[int] = None,
    intrinsic_x_scale: Optional[float | Sequence[float] | Dict[str, float]] = None,
    intrinsic_ftol: Optional[float] = 1e-12,
    intrinsic_xtol: Optional[float] = 1e-12,
    intrinsic_gtol: Optional[float] = 1e-12,
    pose_optimize_points: bool = True,
    intrinsic_optimize_points: bool = True,
    known_points: Optional[Dict[int, np.ndarray]] = None,
    known_point_sigmas: Optional[float | np.ndarray] = None,
    geometry_reference_points: Optional[np.ndarray] = None,
    geometry_reference_cals: Optional[List[Calibration]] = None,
    geometry_guard_mode: str = "off",
    geometry_guard_threshold: Optional[float] = None,
    correspondence_original_ids: Optional[np.ndarray] = None,
    correspondence_point_frame_indices: Optional[np.ndarray] = None,
    correspondence_frame_target_pixels: Optional[Sequence[Sequence[np.ndarray]]] = None,
    correspondence_guard_mode: str = "off",
    correspondence_guard_threshold: Optional[float] = None,
    correspondence_guard_reference_rate: Optional[float] = None,
    reject_worse_solution: bool = True,
    reject_on_ray_convergence: bool = True,
) -> Tuple[List[Calibration], np.ndarray, Dict[str, object]]:
    """Run pose-only BA then tightly constrained intrinsics BA with acceptance checks."""
    if pose_stage_ray_slack < 0:
        raise ValueError("pose_stage_ray_slack must be non-negative")

    normalized_pose_stage_configs = list(pose_stage_configs or [])

    def pose_stage_variants() -> List[Dict[str, object]]:
        if not normalized_pose_stage_configs:
            return [
                {
                    "prior_sigmas": pose_prior_sigmas,
                    "parameter_bounds": pose_parameter_bounds,
                    "max_nfev": pose_max_nfev,
                    "optimize_points": pose_optimize_points,
                    "x_scale": pose_x_scale,
                    "loss": pose_loss,
                    "method": pose_method,
                }
            ]

        variants: List[Dict[str, object]] = []
        for variant in normalized_pose_stage_configs:
            variant_dict = dict(variant)
            variants.append(
                {
                    "prior_sigmas": cast(
                        Optional[Dict[str, float]],
                        variant_dict.get("prior_sigmas", pose_prior_sigmas),
                    ),
                    "parameter_bounds": cast(
                        Optional[Dict[str, Tuple[float, float]]],
                        variant_dict.get("parameter_bounds", pose_parameter_bounds),
                    ),
                    "max_nfev": cast(
                        Optional[int],
                        variant_dict.get("max_nfev", pose_max_nfev),
                    ),
                    "optimize_points": cast(
                        bool,
                        variant_dict.get("optimize_points", pose_optimize_points),
                    ),
                    "x_scale": cast(
                        Optional[float | Sequence[float] | Dict[str, float]],
                        variant_dict.get("x_scale", pose_x_scale),
                    ),
                    "loss": cast(str, variant_dict.get("loss", pose_loss)),
                    "method": cast(str, variant_dict.get("method", pose_method)),
                }
            )
        return variants

    def projection_drift_summaries(
        reference_cals: List[Calibration],
        candidate_cals: List[Calibration],
        reference_points: Optional[np.ndarray],
    ) -> Optional[List[Dict[str, float]]]:
        if reference_points is None:
            return None

        summaries: List[Dict[str, float]] = []
        for camera_index, (reference_cal, candidate_cal) in enumerate(
            zip(reference_cals, candidate_cals),
            start=1,
        ):
            reference_pixels = []
            candidate_pixels = []
            for point in reference_points:
                ref_x, ref_y = img_coord(point, reference_cal, _get_multimedia_par(cpar))
                cand_x, cand_y = img_coord(point, candidate_cal, _get_multimedia_par(cpar))
                reference_pixels.append(metric_to_pixel(ref_x, ref_y, cpar))
                candidate_pixels.append(metric_to_pixel(cand_x, cand_y, cpar))

            displacement = np.linalg.norm(
                np.asarray(candidate_pixels) - np.asarray(reference_pixels),
                axis=1,
            )
            summaries.append(
                {
                    "camera_index": float(camera_index),
                    "mean_distance": float(displacement.mean()),
                    "p95_distance": float(np.percentile(displacement, 95.0)),
                    "max_distance": float(displacement.max()),
                }
            )

        return summaries

    def max_projection_drift(
        summaries: Optional[List[Dict[str, float]]],
    ) -> Optional[float]:
        if not summaries:
            return None
        return max(item["max_distance"] for item in summaries)

    def geometry_stage_ok(
        candidate_metric: Optional[float],
        baseline_metric: Optional[float],
    ) -> bool:
        if geometry_guard_mode == "off" or candidate_metric is None:
            return True
        if geometry_guard_mode == "soft":
            if baseline_metric is None:
                return True
            return candidate_metric <= baseline_metric + 1e-12
        if geometry_guard_mode == "hard":
            if geometry_guard_threshold is None or geometry_guard_threshold <= 0:
                raise ValueError(
                    "geometry_guard_threshold must be positive when geometry_guard_mode='hard'"
                )
            return candidate_metric <= geometry_guard_threshold
        raise ValueError("geometry_guard_mode must be one of 'off', 'soft', or 'hard'")

    def correspondence_replacement_summary(
        candidate_points: np.ndarray,
        candidate_cals: List[Calibration],
    ) -> Optional[Dict[str, object]]:
        if (
            correspondence_original_ids is None
            or correspondence_point_frame_indices is None
            or correspondence_frame_target_pixels is None
        ):
            return None

        projected_pixels = np.empty(
            (candidate_points.shape[0], len(candidate_cals), 2),
            dtype=np.float64,
        )
        for camera_index, cal in enumerate(candidate_cals):
            projected_pixels[:, camera_index, :] = arr_metric_to_pixel(
                image_coordinates(candidate_points, cal, _get_multimedia_par(cpar)),
                cpar,
            )

        replacement_ids = np.empty_like(correspondence_original_ids)
        nearest_distances = np.empty_like(
            correspondence_original_ids,
            dtype=np.float64,
        )
        for point_index in range(candidate_points.shape[0]):
            frame_targets = correspondence_frame_target_pixels[
                int(correspondence_point_frame_indices[point_index])
            ]
            for camera_index in range(len(candidate_cals)):
                deltas = (
                    frame_targets[camera_index]
                    - projected_pixels[point_index, camera_index]
                )
                squared_distances = np.sum(deltas * deltas, axis=1)
                nearest_index = int(np.argmin(squared_distances))
                replacement_ids[point_index, camera_index] = nearest_index
                nearest_distances[point_index, camera_index] = float(
                    np.sqrt(squared_distances[nearest_index])
                )

        changed_mask = np.any(
            replacement_ids != correspondence_original_ids,
            axis=1,
        )
        camera_change_rates = [
            float(
                np.mean(
                    replacement_ids[:, camera_index]
                    != correspondence_original_ids[:, camera_index]
                )
            )
            for camera_index in range(len(candidate_cals))
        ]
        return {
            "replacement_rate": float(np.mean(changed_mask)),
            "camera_change_rates": camera_change_rates,
            "mean_nearest_distance": float(np.mean(nearest_distances)),
            "p95_nearest_distance": float(np.percentile(nearest_distances, 95.0)),
            "max_nearest_distance": float(np.max(nearest_distances)),
        }

    def correspondence_stage_ok(
        candidate_rate: Optional[float],
        prior_rate: Optional[float],
    ) -> bool:
        if correspondence_guard_mode == "off" or candidate_rate is None:
            return True
        if correspondence_guard_mode == "soft":
            if correspondence_guard_reference_rate is not None:
                return candidate_rate <= correspondence_guard_reference_rate + 1e-12
            if prior_rate is None:
                return True
            return candidate_rate <= prior_rate + 1e-12
        if correspondence_guard_mode == "hard":
            if (
                correspondence_guard_threshold is None
                or correspondence_guard_threshold <= 0
            ):
                raise ValueError(
                    "correspondence_guard_threshold must be positive when correspondence_guard_mode='hard'"
                )
            return candidate_rate <= correspondence_guard_threshold
        raise ValueError(
            "correspondence_guard_mode must be one of 'off', 'soft', or 'hard'"
        )

    base_cals = [_clone_calibration(cal) for cal in cals]
    if point_init is None:
        base_points, _ = initialize_bundle_adjustment_points(
            observed_pixels, base_cals, cpar
        )
    else:
        base_points = np.asarray(point_init, dtype=np.float64)
        if base_points.shape != (observed_pixels.shape[0], 3):
            raise ValueError("point_init must have shape (num_points, 3)")

    baseline_rms = reprojection_rms(observed_pixels, base_points, base_cals, cpar)
    baseline_ray_convergence = mean_ray_convergence(observed_pixels, base_cals, cpar)
    if geometry_reference_cals is None:
        geometry_reference_cals = [_clone_calibration(cal) for cal in base_cals]

    baseline_geometry = projection_drift_summaries(
        geometry_reference_cals,
        base_cals,
        geometry_reference_points,
    )
    baseline_geometry_max = max_projection_drift(baseline_geometry)
    baseline_correspondence = correspondence_replacement_summary(
        base_points,
        base_cals,
    )
    baseline_correspondence_rate = (
        None
        if baseline_correspondence is None
        else cast(float, baseline_correspondence["replacement_rate"])
    )

    pose_stage_summaries: List[Dict[str, object]] = []
    pose_stage_variants_list = pose_stage_variants()
    pose_result: object
    pose_ok: bool
    if pose_release_camera_order is None:
        stage_cals = [_clone_calibration(cal) for cal in base_cals]
        stage_points = np.asarray(base_points, dtype=np.float64).copy()
        stage_rms = baseline_rms
        stage_ray_convergence = baseline_ray_convergence
        stage_geometry = baseline_geometry
        stage_geometry_max = baseline_geometry_max
        stage_correspondence = baseline_correspondence
        stage_correspondence_rate = baseline_correspondence_rate
        pose_result = {"staged": bool(normalized_pose_stage_configs), "stages": []}
        pose_ok = False
        pose_geometry_ok = True
        pose_correspondence_ok = True

        for micro_stage_index, variant in enumerate(pose_stage_variants_list, start=1):
            candidate_cals, candidate_points, candidate_result = (
                multi_camera_bundle_adjustment(
                    observed_pixels,
                    stage_cals,
                    cpar,
                    pose_orient_par,
                    point_init=stage_points,
                    fixed_camera_indices=fixed_camera_indices,
                    loss=cast(str, variant["loss"]),
                    method=cast(str, variant["method"]),
                    prior_sigmas=cast(
                        Optional[Dict[str, float]], variant["prior_sigmas"]
                    ),
                    parameter_bounds=cast(
                        Optional[Dict[str, Tuple[float, float]]],
                        variant["parameter_bounds"],
                    ),
                    max_nfev=cast(Optional[int], variant["max_nfev"]),
                    optimize_extrinsics=True,
                    optimize_points=cast(bool, variant["optimize_points"]),
                    known_points=known_points,
                    known_point_sigmas=known_point_sigmas,
                    x_scale=cast(
                        Optional[float | Sequence[float] | Dict[str, float]],
                        variant["x_scale"],
                    ),
                )
            )
            candidate_rms = reprojection_rms(
                observed_pixels, candidate_points, candidate_cals, cpar
            )
            candidate_ray_convergence = mean_ray_convergence(
                observed_pixels, candidate_cals, cpar
            )
            candidate_geometry = projection_drift_summaries(
                geometry_reference_cals,
                candidate_cals,
                geometry_reference_points,
            )
            candidate_geometry_max = max_projection_drift(candidate_geometry)
            candidate_correspondence = correspondence_replacement_summary(
                candidate_points,
                candidate_cals,
            )
            candidate_correspondence_rate = (
                None
                if candidate_correspondence is None
                else cast(float, candidate_correspondence["replacement_rate"])
            )
            pose_geometry_ok = geometry_stage_ok(
                candidate_geometry_max,
                stage_geometry_max,
            )
            pose_correspondence_ok = correspondence_stage_ok(
                candidate_correspondence_rate,
                stage_correspondence_rate,
            )
            pose_ok = candidate_rms <= stage_rms and (
                not reject_on_ray_convergence
                or candidate_ray_convergence
                <= stage_ray_convergence + pose_stage_ray_slack
            )
            pose_ok = pose_ok and pose_geometry_ok and pose_correspondence_ok
            pose_stage_summaries.append(
                {
                    "stage_index": 1,
                    "micro_stage_index": micro_stage_index,
                    "released_camera_index": None,
                    "free_camera_indices": [
                        camera_index
                        for camera_index in range(len(cals))
                        if camera_index not in (fixed_camera_indices or [])
                    ],
                    "fixed_camera_indices": fixed_camera_indices,
                    "reprojection_rms": candidate_rms,
                    "mean_ray_convergence": candidate_ray_convergence,
                    "geometry": candidate_geometry,
                    "geometry_max": candidate_geometry_max,
                    "geometry_ok": pose_geometry_ok,
                    "correspondence": candidate_correspondence,
                    "correspondence_rate": candidate_correspondence_rate,
                    "correspondence_ok": pose_correspondence_ok,
                    "accepted": pose_ok or not reject_worse_solution,
                    "optimize_points": cast(bool, variant["optimize_points"]),
                    "x_scale": variant["x_scale"],
                    "result": candidate_result,
                }
            )
            cast(List[object], pose_result["stages"]).append(candidate_result)

            if reject_worse_solution and not pose_ok:
                break

            stage_cals = candidate_cals
            stage_points = candidate_points
            stage_rms = candidate_rms
            stage_ray_convergence = candidate_ray_convergence
            stage_geometry = candidate_geometry
            stage_geometry_max = candidate_geometry_max
            stage_correspondence = candidate_correspondence
            stage_correspondence_rate = candidate_correspondence_rate

        pose_cals = stage_cals
        pose_points = stage_points
        pose_rms = stage_rms
        pose_ray_convergence = stage_ray_convergence
        pose_geometry = stage_geometry
        pose_geometry_max = stage_geometry_max
        pose_correspondence = stage_correspondence
        pose_correspondence_rate = stage_correspondence_rate
    else:
        release_order = [
            int(camera_index) for camera_index in pose_release_camera_order
        ]
        if not release_order:
            raise ValueError("pose_release_camera_order must not be empty")
        if len(set(release_order)) != len(release_order):
            raise ValueError("pose_release_camera_order must not contain duplicates")
        if any(
            camera_index < 0 or camera_index >= len(cals)
            for camera_index in release_order
        ):
            raise ValueError(
                "pose_release_camera_order contains an out-of-range camera index"
            )

        current_cals = [_clone_calibration(cal) for cal in base_cals]
        current_points = np.asarray(base_points, dtype=np.float64).copy()
        current_rms = baseline_rms
        current_ray_convergence = baseline_ray_convergence
        current_geometry = baseline_geometry
        current_geometry_max = baseline_geometry_max
        current_correspondence = baseline_correspondence
        current_correspondence_rate = baseline_correspondence_rate
        pose_result = {"staged": True, "stages": []}
        pose_ok = False
        pose_geometry_ok = True
        pose_correspondence_ok = True
        released_cameras: List[int] = []

        for stage_index, released_camera in enumerate(release_order, start=1):
            released_cameras.append(released_camera)
            stage_fixed = [
                camera_index
                for camera_index in range(len(cals))
                if camera_index not in released_cameras
            ]
            stage_ok = False
            for micro_stage_index, variant in enumerate(
                pose_stage_variants_list,
                start=1,
            ):
                stage_cals, stage_points, stage_result = multi_camera_bundle_adjustment(
                    observed_pixels,
                    current_cals,
                    cpar,
                    pose_orient_par,
                    point_init=current_points,
                    fixed_camera_indices=stage_fixed,
                    loss=cast(str, variant["loss"]),
                    method=cast(str, variant["method"]),
                    prior_sigmas=cast(
                        Optional[Dict[str, float]],
                        variant["prior_sigmas"],
                    ),
                    parameter_bounds=cast(
                        Optional[Dict[str, Tuple[float, float]]],
                        variant["parameter_bounds"],
                    ),
                    max_nfev=cast(Optional[int], variant["max_nfev"]),
                    optimize_extrinsics=True,
                    optimize_points=cast(bool, variant["optimize_points"]),
                    known_points=known_points,
                    known_point_sigmas=known_point_sigmas,
                    x_scale=cast(
                        Optional[float | Sequence[float] | Dict[str, float]],
                        variant["x_scale"],
                    ),
                )
                stage_rms = reprojection_rms(
                    observed_pixels, stage_points, stage_cals, cpar
                )
                stage_ray_convergence = mean_ray_convergence(
                    observed_pixels, stage_cals, cpar
                )
                stage_geometry = projection_drift_summaries(
                    geometry_reference_cals,
                    stage_cals,
                    geometry_reference_points,
                )
                stage_geometry_max = max_projection_drift(stage_geometry)
                stage_correspondence = correspondence_replacement_summary(
                    stage_points,
                    stage_cals,
                )
                stage_correspondence_rate = (
                    None
                    if stage_correspondence is None
                    else cast(float, stage_correspondence["replacement_rate"])
                )
                stage_geometry_ok = geometry_stage_ok(
                    stage_geometry_max,
                    current_geometry_max,
                )
                stage_correspondence_ok = correspondence_stage_ok(
                    stage_correspondence_rate,
                    current_correspondence_rate,
                )
                stage_ok = stage_rms <= current_rms and (
                    not reject_on_ray_convergence
                    or stage_ray_convergence
                    <= current_ray_convergence + pose_stage_ray_slack
                )
                stage_ok = stage_ok and stage_geometry_ok and stage_correspondence_ok
                pose_stage_summaries.append(
                    {
                        "stage_index": stage_index,
                        "micro_stage_index": micro_stage_index,
                        "released_camera_index": released_camera,
                        "free_camera_indices": released_cameras.copy(),
                        "fixed_camera_indices": stage_fixed,
                        "reprojection_rms": stage_rms,
                        "mean_ray_convergence": stage_ray_convergence,
                        "geometry": stage_geometry,
                        "geometry_max": stage_geometry_max,
                        "geometry_ok": stage_geometry_ok,
                        "correspondence": stage_correspondence,
                        "correspondence_rate": stage_correspondence_rate,
                        "correspondence_ok": stage_correspondence_ok,
                        "accepted": stage_ok or not reject_worse_solution,
                        "optimize_points": cast(bool, variant["optimize_points"]),
                        "x_scale": variant["x_scale"],
                        "result": stage_result,
                    }
                )
                cast(List[object], pose_result["stages"]).append(stage_result)

                if reject_worse_solution and not stage_ok:
                    break

                current_cals = stage_cals
                current_points = stage_points
                current_rms = stage_rms
                current_ray_convergence = stage_ray_convergence
                current_geometry = stage_geometry
                current_geometry_max = stage_geometry_max
                current_correspondence = stage_correspondence
                current_correspondence_rate = stage_correspondence_rate
                pose_ok = True
                pose_geometry_ok = stage_geometry_ok
                pose_correspondence_ok = stage_correspondence_ok

            if reject_worse_solution and not stage_ok:
                break

        pose_cals = current_cals
        pose_points = current_points
        pose_rms = current_rms
        pose_ray_convergence = current_ray_convergence
        pose_geometry = current_geometry
        pose_geometry_max = current_geometry_max
        pose_correspondence = current_correspondence
        pose_correspondence_rate = current_correspondence_rate

    intrinsic_fixed = list(range(len(cals)))
    intrinsic_cals, intrinsic_points, intrinsic_result = multi_camera_bundle_adjustment(
        observed_pixels,
        pose_cals,
        cpar,
        intrinsic_orient_par,
        point_init=pose_points,
        fixed_camera_indices=intrinsic_fixed,
        loss=intrinsic_loss,
        method=intrinsic_method,
        prior_sigmas=intrinsic_prior_sigmas,
        parameter_bounds=intrinsic_parameter_bounds,
        max_nfev=intrinsic_max_nfev,
        optimize_extrinsics=False,
        optimize_points=intrinsic_optimize_points,
        known_points=known_points,
        known_point_sigmas=known_point_sigmas,
        x_scale=intrinsic_x_scale,
        ftol=intrinsic_ftol,
        xtol=intrinsic_xtol,
        gtol=intrinsic_gtol,
    )
    intrinsic_rms = reprojection_rms(
        observed_pixels, intrinsic_points, intrinsic_cals, cpar
    )
    intrinsic_ray_convergence = mean_ray_convergence(
        observed_pixels, intrinsic_cals, cpar
    )
    intrinsic_geometry = projection_drift_summaries(
        geometry_reference_cals,
        intrinsic_cals,
        geometry_reference_points,
    )
    intrinsic_geometry_max = max_projection_drift(intrinsic_geometry)
    intrinsic_correspondence = correspondence_replacement_summary(
        intrinsic_points,
        intrinsic_cals,
    )
    intrinsic_correspondence_rate = (
        None
        if intrinsic_correspondence is None
        else cast(float, intrinsic_correspondence["replacement_rate"])
    )

    accepted_stage = "intrinsics"
    final_cals = intrinsic_cals
    final_points = intrinsic_points
    final_rms = intrinsic_rms
    final_ray_convergence = intrinsic_ray_convergence
    intrinsic_ok = intrinsic_rms <= pose_rms and (
        not reject_on_ray_convergence
        or intrinsic_ray_convergence <= pose_ray_convergence
    )
    intrinsic_geometry_ok = geometry_stage_ok(
        intrinsic_geometry_max,
        pose_geometry_max,
    )
    intrinsic_ok = intrinsic_ok and intrinsic_geometry_ok
    intrinsic_correspondence_ok = correspondence_stage_ok(
        intrinsic_correspondence_rate,
        pose_correspondence_rate,
    )
    intrinsic_ok = intrinsic_ok and intrinsic_correspondence_ok

    if reject_worse_solution:
        if not pose_ok:
            accepted_stage = "baseline"
            final_cals = base_cals
            final_points = base_points
            final_rms = baseline_rms
            final_ray_convergence = baseline_ray_convergence
        elif not intrinsic_ok:
            accepted_stage = "pose"
            final_cals = pose_cals
            final_points = pose_points
            final_rms = pose_rms
            final_ray_convergence = pose_ray_convergence

    summary = {
        "baseline_reprojection_rms": baseline_rms,
        "baseline_mean_ray_convergence": baseline_ray_convergence,
        "baseline_cals": base_cals,
        "baseline_points": base_points,
        "baseline_geometry": baseline_geometry,
        "baseline_geometry_max": baseline_geometry_max,
        "baseline_correspondence": baseline_correspondence,
        "baseline_correspondence_rate": baseline_correspondence_rate,
        "pose_reprojection_rms": pose_rms,
        "pose_mean_ray_convergence": pose_ray_convergence,
        "pose_cals": pose_cals,
        "pose_points": pose_points,
        "pose_geometry": pose_geometry,
        "pose_geometry_max": pose_geometry_max,
        "pose_geometry_ok": pose_geometry_ok,
        "pose_release_camera_order": pose_release_camera_order,
        "pose_stage_ray_slack": pose_stage_ray_slack,
        "pose_stage_configs": normalized_pose_stage_configs,
        "pose_stage_summaries": pose_stage_summaries,
        "accepted_pose_stage_count": len(
            [summary for summary in pose_stage_summaries if summary["accepted"]]
        ),
        "pose_correspondence": pose_correspondence,
        "pose_correspondence_rate": pose_correspondence_rate,
        "pose_correspondence_ok": pose_correspondence_ok,
        "intrinsic_reprojection_rms": intrinsic_rms,
        "intrinsic_mean_ray_convergence": intrinsic_ray_convergence,
        "intrinsic_cals": intrinsic_cals,
        "intrinsic_points": intrinsic_points,
        "intrinsic_geometry": intrinsic_geometry,
        "intrinsic_geometry_max": intrinsic_geometry_max,
        "intrinsic_geometry_ok": intrinsic_geometry_ok,
        "intrinsic_correspondence": intrinsic_correspondence,
        "intrinsic_correspondence_rate": intrinsic_correspondence_rate,
        "intrinsic_correspondence_ok": intrinsic_correspondence_ok,
        "geometry_guard_mode": geometry_guard_mode,
        "geometry_guard_threshold": geometry_guard_threshold,
        "correspondence_guard_mode": correspondence_guard_mode,
        "correspondence_guard_threshold": correspondence_guard_threshold,
        "correspondence_guard_reference_rate": correspondence_guard_reference_rate,
        "accepted_stage": accepted_stage,
        "final_reprojection_rms": final_rms,
        "final_mean_ray_convergence": final_ray_convergence,
        "pose_result": pose_result,
        "intrinsic_result": intrinsic_result,
    }

    return final_cals, final_points, summary


def alternating_bundle_adjustment(
    observed_pixels: np.ndarray,
    cals: List[Calibration],
    cpar: ControlPar,
    pose_orient_par: OrientPar,
    intrinsic_orient_par: OrientPar,
    *,
    point_init: Optional[np.ndarray] = None,
    fixed_camera_indices: Optional[List[int]] = None,
    pose_release_camera_order: Optional[List[int]] = None,
    pose_stage_ray_slack: float = 0.0,
    pose_prior_sigmas: Optional[Dict[str, float]] = None,
    pose_parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    pose_loss: str = "linear",
    pose_method: str = "trf",
    pose_max_nfev: Optional[int] = None,
    pose_x_scale: Optional[float | Sequence[float] | Dict[str, float]] = None,
    pose_block_configs: Optional[Sequence[Dict[str, object]]] = None,
    intrinsic_prior_sigmas: Optional[Dict[str, float]] = None,
    intrinsic_parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    intrinsic_loss: str = "linear",
    intrinsic_method: str = "trf",
    intrinsic_max_nfev: Optional[int] = None,
    intrinsic_x_scale: Optional[float | Sequence[float] | Dict[str, float]] = None,
    intrinsic_ftol: Optional[float] = 1e-12,
    intrinsic_xtol: Optional[float] = 1e-12,
    intrinsic_gtol: Optional[float] = 1e-12,
    known_points: Optional[Dict[int, np.ndarray]] = None,
    known_point_sigmas: Optional[float | np.ndarray] = None,
    geometry_reference_points: Optional[np.ndarray] = None,
    geometry_reference_cals: Optional[List[Calibration]] = None,
    geometry_guard_mode: str = "off",
    geometry_guard_threshold: Optional[float] = None,
    first_release_geometry_slack: float = 0.0,
    correspondence_original_ids: Optional[np.ndarray] = None,
    correspondence_point_frame_indices: Optional[np.ndarray] = None,
    correspondence_frame_target_pixels: Optional[Sequence[Sequence[np.ndarray]]] = None,
    correspondence_guard_mode: str = "off",
    correspondence_guard_threshold: Optional[float] = None,
    correspondence_guard_reference_rate: Optional[float] = None,
    first_release_correspondence_slack: float = 0.0,
    reject_worse_solution: bool = True,
    reject_on_ray_convergence: bool = True,
) -> Tuple[List[Calibration], np.ndarray, Dict[str, object]]:
    """Run intrinsics-first alternating BA with point and pose sub-block updates."""
    if pose_stage_ray_slack < 0:
        raise ValueError("pose_stage_ray_slack must be non-negative")
    if first_release_geometry_slack < 0:
        raise ValueError("first_release_geometry_slack must be non-negative")
    if first_release_correspondence_slack < 0:
        raise ValueError("first_release_correspondence_slack must be non-negative")

    def merge_bounds(
        base: Optional[Dict[str, Tuple[float, float]]],
        updates: Optional[Dict[str, Tuple[float, float]]],
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        merged = dict(base or {})
        if updates is not None:
            merged.update(updates)
        return merged or None

    def default_pose_blocks() -> List[Dict[str, object]]:
        return [
            {
                "name": "points_only",
                "optimize_extrinsics": False,
                "optimize_points": True,
                "loss": pose_loss,
                "method": pose_method,
                "max_nfev": 4 if pose_max_nfev is None else min(4, pose_max_nfev),
                "x_scale": {"points": 0.1},
            },
            {
                "name": "omega_only",
                "optimize_extrinsics": True,
                "optimize_points": False,
                "freeze_translation": True,
                "loss": pose_loss,
                "method": pose_method,
                "max_nfev": 3 if pose_max_nfev is None else min(3, pose_max_nfev),
                "x_scale": {
                    "omega": 2e-4,
                },
                "frozen_parameters": ["phi", "kappa"],
            },
            {
                "name": "phi_only",
                "optimize_extrinsics": True,
                "optimize_points": False,
                "freeze_translation": True,
                "loss": pose_loss,
                "method": pose_method,
                "max_nfev": 3 if pose_max_nfev is None else min(3, pose_max_nfev),
                "x_scale": {
                    "phi": 2e-4,
                },
                "frozen_parameters": ["omega", "kappa"],
            },
            {
                "name": "kappa_only",
                "optimize_extrinsics": True,
                "optimize_points": False,
                "freeze_translation": True,
                "loss": pose_loss,
                "method": pose_method,
                "max_nfev": 3 if pose_max_nfev is None else min(3, pose_max_nfev),
                "x_scale": {
                    "kappa": 2e-4,
                },
                "frozen_parameters": ["omega", "phi"],
            },
            {
                "name": "translation_only",
                "optimize_extrinsics": True,
                "optimize_points": False,
                "freeze_rotation": True,
                "loss": pose_loss,
                "method": pose_method,
                "max_nfev": 4 if pose_max_nfev is None else min(4, pose_max_nfev),
                "x_scale": {
                    "x0": 0.02,
                    "y0": 0.02,
                    "z0": 0.02,
                },
            },
            {
                "name": "joint_pose_points",
                "optimize_extrinsics": True,
                "optimize_points": True,
                "loss": pose_loss,
                "method": pose_method,
                "max_nfev": pose_max_nfev,
                "x_scale": pose_x_scale,
            },
        ]

    normalized_pose_block_configs = list(
        default_pose_blocks() if pose_block_configs is None else pose_block_configs
    )

    def projection_drift_summaries(
        reference_cals: List[Calibration],
        candidate_cals: List[Calibration],
        reference_points: Optional[np.ndarray],
    ) -> Optional[List[Dict[str, float]]]:
        if reference_points is None:
            return None

        summaries: List[Dict[str, float]] = []
        for camera_index, (reference_cal, candidate_cal) in enumerate(
            zip(reference_cals, candidate_cals),
            start=1,
        ):
            reference_pixels = []
            candidate_pixels = []
            for point in reference_points:
                ref_x, ref_y = img_coord(point, reference_cal, _get_multimedia_par(cpar))
                cand_x, cand_y = img_coord(point, candidate_cal, _get_multimedia_par(cpar))
                reference_pixels.append(metric_to_pixel(ref_x, ref_y, cpar))
                candidate_pixels.append(metric_to_pixel(cand_x, cand_y, cpar))

            displacement = np.linalg.norm(
                np.asarray(candidate_pixels) - np.asarray(reference_pixels),
                axis=1,
            )
            summaries.append(
                {
                    "camera_index": float(camera_index),
                    "mean_distance": float(displacement.mean()),
                    "p95_distance": float(np.percentile(displacement, 95.0)),
                    "max_distance": float(displacement.max()),
                }
            )

        return summaries

    def max_projection_drift(
        summaries: Optional[List[Dict[str, float]]],
    ) -> Optional[float]:
        if not summaries:
            return None
        return max(item["max_distance"] for item in summaries)

    def geometry_stage_ok(
        candidate_metric: Optional[float],
        baseline_metric: Optional[float],
        *,
        threshold_override: Optional[float] = None,
    ) -> bool:
        if geometry_guard_mode == "off" or candidate_metric is None:
            return True
        if geometry_guard_mode == "soft":
            if baseline_metric is None:
                return True
            return candidate_metric <= baseline_metric + 1e-12
        if geometry_guard_mode == "hard":
            threshold = (
                geometry_guard_threshold
                if threshold_override is None
                else threshold_override
            )
            if threshold is None or threshold <= 0:
                raise ValueError(
                    "geometry_guard_threshold must be positive when geometry_guard_mode='hard'"
                )
            return candidate_metric <= threshold
        raise ValueError("geometry_guard_mode must be one of 'off', 'soft', or 'hard'")

    def correspondence_replacement_summary(
        candidate_points: np.ndarray,
        candidate_cals: List[Calibration],
    ) -> Optional[Dict[str, object]]:
        if (
            correspondence_original_ids is None
            or correspondence_point_frame_indices is None
            or correspondence_frame_target_pixels is None
        ):
            return None

        projected_pixels = np.empty(
            (candidate_points.shape[0], len(candidate_cals), 2),
            dtype=np.float64,
        )
        for camera_index, cal in enumerate(candidate_cals):
            projected_pixels[:, camera_index, :] = arr_metric_to_pixel(
                image_coordinates(candidate_points, cal, _get_multimedia_par(cpar)),
                cpar,
            )

        replacement_ids = np.empty_like(correspondence_original_ids)
        nearest_distances = np.empty_like(
            correspondence_original_ids,
            dtype=np.float64,
        )
        for point_index in range(candidate_points.shape[0]):
            frame_targets = correspondence_frame_target_pixels[
                int(correspondence_point_frame_indices[point_index])
            ]
            for camera_index in range(len(candidate_cals)):
                deltas = (
                    frame_targets[camera_index]
                    - projected_pixels[point_index, camera_index]
                )
                squared_distances = np.sum(deltas * deltas, axis=1)
                nearest_index = int(np.argmin(squared_distances))
                replacement_ids[point_index, camera_index] = nearest_index
                nearest_distances[point_index, camera_index] = float(
                    np.sqrt(squared_distances[nearest_index])
                )

        changed_mask = np.any(
            replacement_ids != correspondence_original_ids,
            axis=1,
        )
        camera_change_rates = [
            float(
                np.mean(
                    replacement_ids[:, camera_index]
                    != correspondence_original_ids[:, camera_index]
                )
            )
            for camera_index in range(len(candidate_cals))
        ]
        return {
            "replacement_rate": float(np.mean(changed_mask)),
            "camera_change_rates": camera_change_rates,
            "mean_nearest_distance": float(np.mean(nearest_distances)),
            "p95_nearest_distance": float(np.percentile(nearest_distances, 95.0)),
            "max_nearest_distance": float(np.max(nearest_distances)),
        }

    def correspondence_stage_ok(
        candidate_rate: Optional[float],
        prior_rate: Optional[float],
        *,
        threshold_override: Optional[float] = None,
    ) -> bool:
        if correspondence_guard_mode == "off" or candidate_rate is None:
            return True
        if correspondence_guard_mode == "soft":
            if correspondence_guard_reference_rate is not None:
                return candidate_rate <= correspondence_guard_reference_rate + 1e-12
            if prior_rate is None:
                return True
            return candidate_rate <= prior_rate + 1e-12
        if correspondence_guard_mode == "hard":
            if threshold_override is None and correspondence_guard_threshold is None:
                threshold = None
            else:
                threshold = (
                    correspondence_guard_threshold
                    if threshold_override is None
                    else threshold_override
                )
            if threshold is None or threshold <= 0:
                raise ValueError(
                    "correspondence_guard_threshold must be positive when correspondence_guard_mode='hard'"
                )
            return candidate_rate <= threshold
        raise ValueError(
            "correspondence_guard_mode must be one of 'off', 'soft', or 'hard'"
        )

    base_cals = [_clone_calibration(cal) for cal in cals]
    if point_init is None:
        base_points, _ = initialize_bundle_adjustment_points(
            observed_pixels, base_cals, cpar
        )
    else:
        base_points = np.asarray(point_init, dtype=np.float64)
        if base_points.shape != (observed_pixels.shape[0], 3):
            raise ValueError("point_init must have shape (num_points, 3)")

    baseline_rms = reprojection_rms(observed_pixels, base_points, base_cals, cpar)
    baseline_ray_convergence = mean_ray_convergence(observed_pixels, base_cals, cpar)
    if geometry_reference_cals is None:
        geometry_reference_cals = [_clone_calibration(cal) for cal in base_cals]

    baseline_geometry = projection_drift_summaries(
        geometry_reference_cals,
        base_cals,
        geometry_reference_points,
    )
    baseline_geometry_max = max_projection_drift(baseline_geometry)
    baseline_correspondence = correspondence_replacement_summary(
        base_points,
        base_cals,
    )
    baseline_correspondence_rate = (
        None
        if baseline_correspondence is None
        else cast(float, baseline_correspondence["replacement_rate"])
    )

    warmstart_fixed = list(range(len(cals)))
    warmstart_cals, warmstart_points, warmstart_result = multi_camera_bundle_adjustment(
        observed_pixels,
        base_cals,
        cpar,
        intrinsic_orient_par,
        point_init=base_points,
        fixed_camera_indices=warmstart_fixed,
        loss=intrinsic_loss,
        method=intrinsic_method,
        prior_sigmas=intrinsic_prior_sigmas,
        parameter_bounds=intrinsic_parameter_bounds,
        max_nfev=intrinsic_max_nfev,
        optimize_extrinsics=False,
        optimize_points=False,
        x_scale=intrinsic_x_scale,
        ftol=intrinsic_ftol,
        xtol=intrinsic_xtol,
        gtol=intrinsic_gtol,
    )
    warmstart_rms = reprojection_rms(
        observed_pixels, warmstart_points, warmstart_cals, cpar
    )
    warmstart_ray_convergence = mean_ray_convergence(
        observed_pixels, warmstart_cals, cpar
    )
    warmstart_geometry = projection_drift_summaries(
        geometry_reference_cals,
        warmstart_cals,
        geometry_reference_points,
    )
    warmstart_geometry_max = max_projection_drift(warmstart_geometry)
    warmstart_correspondence = correspondence_replacement_summary(
        warmstart_points,
        warmstart_cals,
    )
    warmstart_correspondence_rate = (
        None
        if warmstart_correspondence is None
        else cast(float, warmstart_correspondence["replacement_rate"])
    )
    warmstart_ok = warmstart_rms <= baseline_rms and (
        not reject_on_ray_convergence
        or warmstart_ray_convergence <= baseline_ray_convergence
    )
    warmstart_ok = warmstart_ok and geometry_stage_ok(
        warmstart_geometry_max,
        baseline_geometry_max,
    )
    warmstart_ok = warmstart_ok and correspondence_stage_ok(
        warmstart_correspondence_rate,
        baseline_correspondence_rate,
    )

    if reject_worse_solution and not warmstart_ok:
        current_cals = [_clone_calibration(cal) for cal in base_cals]
        current_points = np.asarray(base_points, dtype=np.float64).copy()
        current_rms = baseline_rms
        current_ray_convergence = baseline_ray_convergence
        current_geometry = baseline_geometry
        current_geometry_max = baseline_geometry_max
        current_correspondence = baseline_correspondence
        current_correspondence_rate = baseline_correspondence_rate
    else:
        current_cals = warmstart_cals
        current_points = warmstart_points
        current_rms = warmstart_rms
        current_ray_convergence = warmstart_ray_convergence
        current_geometry = warmstart_geometry
        current_geometry_max = warmstart_geometry_max
        current_correspondence = warmstart_correspondence
        current_correspondence_rate = warmstart_correspondence_rate

    if pose_release_camera_order is None:
        release_order = [
            camera_index
            for camera_index in range(len(cals))
            if camera_index not in (fixed_camera_indices or [])
        ]
    else:
        release_order = [
            int(camera_index) for camera_index in pose_release_camera_order
        ]
    if not release_order:
        raise ValueError("pose_release_camera_order must not be empty")

    alternating_result: Dict[str, object] = {
        "warmstart_result": warmstart_result,
        "stages": [],
    }
    block_summaries: List[Dict[str, object]] = []
    pose_ok = False
    pose_geometry_ok = True
    pose_correspondence_ok = True
    released_cameras: List[int] = []

    for stage_index, released_camera in enumerate(release_order, start=1):
        released_cameras.append(released_camera)
        stage_fixed = [
            camera_index
            for camera_index in range(len(cals))
            if camera_index not in released_cameras
        ]
        stage_completed = True
        for block_index, block in enumerate(normalized_pose_block_configs, start=1):
            block_config = dict(block)
            optimize_extrinsics = cast(
                bool,
                block_config.get("optimize_extrinsics", True),
            )
            optimize_points = cast(bool, block_config.get("optimize_points", True))
            block_bounds = merge_bounds(
                pose_parameter_bounds,
                cast(
                    Optional[Dict[str, Tuple[float, float]]],
                    block_config.get("parameter_bounds"),
                ),
            )
            frozen_parameters = set(
                cast(Sequence[str], block_config.get("frozen_parameters", []))
            )
            if cast(bool, block_config.get("freeze_translation", False)):
                frozen_parameters.update({"x0", "y0", "z0"})
            if cast(bool, block_config.get("freeze_rotation", False)):
                frozen_parameters.update({"omega", "phi", "kappa"})
            unknown_frozen = frozen_parameters.difference(
                {"x0", "y0", "z0", "omega", "phi", "kappa"}
            )
            if unknown_frozen:
                raise ValueError(
                    "Unsupported frozen_parameters entries: "
                    + ", ".join(sorted(unknown_frozen))
                )
            if frozen_parameters:
                block_bounds = merge_bounds(
                    block_bounds,
                    {
                        parameter_name: (0.0, 0.0)
                        for parameter_name in sorted(frozen_parameters)
                    },
                )

            block_known_points = known_points if optimize_points else None
            block_known_point_sigmas = known_point_sigmas if optimize_points else None
            block_cals, block_points, block_result = multi_camera_bundle_adjustment(
                observed_pixels,
                current_cals,
                cpar,
                pose_orient_par,
                point_init=current_points,
                fixed_camera_indices=stage_fixed,
                loss=cast(str, block_config.get("loss", pose_loss)),
                method=cast(str, block_config.get("method", pose_method)),
                prior_sigmas=cast(
                    Optional[Dict[str, float]],
                    block_config.get("prior_sigmas", pose_prior_sigmas),
                ),
                parameter_bounds=block_bounds,
                max_nfev=cast(
                    Optional[int], block_config.get("max_nfev", pose_max_nfev)
                ),
                optimize_extrinsics=optimize_extrinsics,
                optimize_points=optimize_points,
                known_points=block_known_points,
                known_point_sigmas=block_known_point_sigmas,
                x_scale=cast(
                    Optional[float | Sequence[float] | Dict[str, float]],
                    block_config.get("x_scale", pose_x_scale),
                ),
            )
            block_rms = reprojection_rms(
                observed_pixels, block_points, block_cals, cpar
            )
            block_ray_convergence = mean_ray_convergence(
                observed_pixels,
                block_cals,
                cpar,
            )
            block_geometry = projection_drift_summaries(
                geometry_reference_cals,
                block_cals,
                geometry_reference_points,
            )
            block_geometry_max = max_projection_drift(block_geometry)
            block_correspondence = correspondence_replacement_summary(
                block_points,
                block_cals,
            )
            block_correspondence_rate = (
                None
                if block_correspondence is None
                else cast(float, block_correspondence["replacement_rate"])
            )
            first_release_soft_geometry = (
                stage_index == 1
                and optimize_extrinsics
                and geometry_guard_mode == "hard"
                and first_release_geometry_slack > 0
                and geometry_guard_threshold is not None
            )
            first_release_soft_correspondence = (
                stage_index == 1
                and optimize_extrinsics
                and correspondence_guard_mode == "hard"
                and first_release_correspondence_slack > 0
                and correspondence_guard_threshold is not None
            )
            geometry_threshold_override = None
            if first_release_soft_geometry:
                geometry_guard_limit = cast(float, geometry_guard_threshold)
                geometry_threshold_override = (
                    geometry_guard_limit + first_release_geometry_slack
                )
            correspondence_threshold_override = None
            if first_release_soft_correspondence:
                correspondence_guard_limit = cast(float, correspondence_guard_threshold)
                correspondence_threshold_override = (
                    correspondence_guard_limit + first_release_correspondence_slack
                )
            block_geometry_ok = geometry_stage_ok(
                block_geometry_max,
                current_geometry_max,
                threshold_override=geometry_threshold_override,
            )
            block_correspondence_ok = correspondence_stage_ok(
                block_correspondence_rate,
                current_correspondence_rate,
                threshold_override=correspondence_threshold_override,
            )
            block_ok = block_rms <= current_rms and (
                not reject_on_ray_convergence
                or block_ray_convergence
                <= current_ray_convergence + pose_stage_ray_slack
            )
            block_ok = block_ok and block_geometry_ok and block_correspondence_ok
            block_summaries.append(
                {
                    "stage_index": stage_index,
                    "block_index": block_index,
                    "block_name": block_config.get("name", f"block_{block_index}"),
                    "released_camera_index": released_camera,
                    "free_camera_indices": released_cameras.copy(),
                    "fixed_camera_indices": stage_fixed,
                    "reprojection_rms": block_rms,
                    "mean_ray_convergence": block_ray_convergence,
                    "geometry": block_geometry,
                    "geometry_max": block_geometry_max,
                    "geometry_ok": block_geometry_ok,
                    "first_release_soft_geometry": first_release_soft_geometry,
                    "correspondence": block_correspondence,
                    "correspondence_rate": block_correspondence_rate,
                    "correspondence_ok": block_correspondence_ok,
                    "first_release_soft_correspondence": first_release_soft_correspondence,
                    "accepted": block_ok or not reject_worse_solution,
                    "optimize_extrinsics": optimize_extrinsics,
                    "optimize_points": optimize_points,
                    "result": block_result,
                }
            )
            cast(List[object], alternating_result["stages"]).append(block_result)

            if reject_worse_solution and not block_ok:
                stage_completed = False
                break

            current_cals = block_cals
            current_points = block_points
            current_rms = block_rms
            current_ray_convergence = block_ray_convergence
            current_geometry = block_geometry
            current_geometry_max = block_geometry_max
            current_correspondence = block_correspondence
            current_correspondence_rate = block_correspondence_rate
            pose_ok = True
            pose_geometry_ok = block_geometry_ok
            pose_correspondence_ok = block_correspondence_ok

        if reject_worse_solution and not stage_completed:
            break

    pose_cals = current_cals
    pose_points = current_points
    pose_rms = current_rms
    pose_ray_convergence = current_ray_convergence
    pose_geometry = current_geometry
    pose_geometry_max = current_geometry_max
    pose_correspondence = current_correspondence
    pose_correspondence_rate = current_correspondence_rate

    intrinsic_fixed = list(range(len(cals)))
    intrinsic_cals, intrinsic_points, intrinsic_result = multi_camera_bundle_adjustment(
        observed_pixels,
        pose_cals,
        cpar,
        intrinsic_orient_par,
        point_init=pose_points,
        fixed_camera_indices=intrinsic_fixed,
        loss=intrinsic_loss,
        method=intrinsic_method,
        prior_sigmas=intrinsic_prior_sigmas,
        parameter_bounds=intrinsic_parameter_bounds,
        max_nfev=intrinsic_max_nfev,
        optimize_extrinsics=False,
        optimize_points=False,
        x_scale=intrinsic_x_scale,
        ftol=intrinsic_ftol,
        xtol=intrinsic_xtol,
        gtol=intrinsic_gtol,
    )
    intrinsic_rms = reprojection_rms(
        observed_pixels, intrinsic_points, intrinsic_cals, cpar
    )
    intrinsic_ray_convergence = mean_ray_convergence(
        observed_pixels, intrinsic_cals, cpar
    )
    intrinsic_geometry = projection_drift_summaries(
        geometry_reference_cals,
        intrinsic_cals,
        geometry_reference_points,
    )
    intrinsic_geometry_max = max_projection_drift(intrinsic_geometry)
    intrinsic_correspondence = correspondence_replacement_summary(
        intrinsic_points,
        intrinsic_cals,
    )
    intrinsic_correspondence_rate = (
        None
        if intrinsic_correspondence is None
        else cast(float, intrinsic_correspondence["replacement_rate"])
    )

    accepted_stage = "intrinsics"
    final_cals = intrinsic_cals
    final_points = intrinsic_points
    final_rms = intrinsic_rms
    final_ray_convergence = intrinsic_ray_convergence
    intrinsic_ok = intrinsic_rms <= pose_rms and (
        not reject_on_ray_convergence
        or intrinsic_ray_convergence <= pose_ray_convergence
    )
    intrinsic_ok = intrinsic_ok and geometry_stage_ok(
        intrinsic_geometry_max,
        pose_geometry_max,
    )
    intrinsic_ok = intrinsic_ok and correspondence_stage_ok(
        intrinsic_correspondence_rate,
        pose_correspondence_rate,
    )

    if reject_worse_solution:
        if not pose_ok:
            if warmstart_ok:
                accepted_stage = "warmstart"
                final_cals = warmstart_cals
                final_points = warmstart_points
                final_rms = warmstart_rms
                final_ray_convergence = warmstart_ray_convergence
            else:
                accepted_stage = "baseline"
                final_cals = base_cals
                final_points = base_points
                final_rms = baseline_rms
                final_ray_convergence = baseline_ray_convergence
        elif not intrinsic_ok:
            accepted_stage = "pose_blocks"
            final_cals = pose_cals
            final_points = pose_points
            final_rms = pose_rms
            final_ray_convergence = pose_ray_convergence

    summary = {
        "baseline_reprojection_rms": baseline_rms,
        "baseline_mean_ray_convergence": baseline_ray_convergence,
        "baseline_cals": base_cals,
        "baseline_points": base_points,
        "baseline_geometry": baseline_geometry,
        "baseline_geometry_max": baseline_geometry_max,
        "baseline_correspondence": baseline_correspondence,
        "baseline_correspondence_rate": baseline_correspondence_rate,
        "warmstart_reprojection_rms": warmstart_rms,
        "warmstart_mean_ray_convergence": warmstart_ray_convergence,
        "warmstart_cals": warmstart_cals,
        "warmstart_points": warmstart_points,
        "warmstart_geometry": warmstart_geometry,
        "warmstart_geometry_max": warmstart_geometry_max,
        "warmstart_correspondence": warmstart_correspondence,
        "warmstart_correspondence_rate": warmstart_correspondence_rate,
        "warmstart_ok": warmstart_ok,
        "pose_reprojection_rms": pose_rms,
        "pose_mean_ray_convergence": pose_ray_convergence,
        "pose_cals": pose_cals,
        "pose_points": pose_points,
        "pose_geometry": pose_geometry,
        "pose_geometry_max": pose_geometry_max,
        "pose_geometry_ok": pose_geometry_ok,
        "pose_release_camera_order": release_order,
        "pose_stage_ray_slack": pose_stage_ray_slack,
        "pose_block_configs": normalized_pose_block_configs,
        "pose_block_summaries": block_summaries,
        "accepted_pose_block_count": len(
            [block for block in block_summaries if block["accepted"]]
        ),
        "pose_correspondence": pose_correspondence,
        "pose_correspondence_rate": pose_correspondence_rate,
        "pose_correspondence_ok": pose_correspondence_ok,
        "intrinsic_reprojection_rms": intrinsic_rms,
        "intrinsic_mean_ray_convergence": intrinsic_ray_convergence,
        "intrinsic_cals": intrinsic_cals,
        "intrinsic_points": intrinsic_points,
        "intrinsic_geometry": intrinsic_geometry,
        "intrinsic_geometry_max": intrinsic_geometry_max,
        "intrinsic_correspondence": intrinsic_correspondence,
        "intrinsic_correspondence_rate": intrinsic_correspondence_rate,
        "geometry_guard_mode": geometry_guard_mode,
        "geometry_guard_threshold": geometry_guard_threshold,
        "first_release_geometry_slack": first_release_geometry_slack,
        "correspondence_guard_mode": correspondence_guard_mode,
        "correspondence_guard_threshold": correspondence_guard_threshold,
        "correspondence_guard_reference_rate": correspondence_guard_reference_rate,
        "first_release_correspondence_slack": first_release_correspondence_slack,
        "accepted_stage": accepted_stage,
        "final_reprojection_rms": final_rms,
        "final_mean_ray_convergence": final_ray_convergence,
        "warmstart_result": warmstart_result,
        "pose_result": alternating_result,
        "intrinsic_result": intrinsic_result,
    }

    return final_cals, final_points, summary
