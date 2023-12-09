"""Functions for the orientation of the camera."""
from typing import List, Optional, Tuple

import numpy as np
import scipy

from openptv_python.constants import COORD_UNUSED

from .calibration import Calibration
from .constants import CONVERGENCE, IDT, NPAR, NUM_ITER, POS_INF
from .epi import epi_mm_2D
from .imgcoord import img_coord

# from .lsqadj import ata, atl, matinv, matmul
from .parameters import ControlPar, MultimediaPar, OrientPar, VolumePar
from .ray_tracing import ray_tracing
from .sortgrid import sortgrid
from .tracking_frame_buf import Target, TargetArray
from .trafo import correct_brown_affine, pixel_to_metric
from .vec_utils import unit_vector, vec_norm, vec_set


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
            dist = np.linalg.norm(res[0] - res[1])
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
    cal.ext_par.update_rotation_matrix()
    xs, ys = img_coord(pos, cal, cpar.mm)
    # print(f"  xs = {xs}, ys = {ys}")

    for pd in range(6):
        cal.ext_par.increment_attribute(var[pd], steps[pd])
        # print(f"exterior = {cal.ext_par}")
        if pd > 2:
            cal.ext_par.update_rotation_matrix()

        xpd, ypd = img_coord(pos, cal, cpar.mm)
        # print(f" xpd = {xpd}, ypd = {ypd}")
        x_ders[pd] = (xpd - xs) / steps[pd]
        y_ders[pd] = (ypd - ys) / steps[pd]

        # print(f"   x_ders[{pd}] = {x_ders[pd]}, y_ders[{pd}] = {y_ders[pd]}")

        cal.ext_par.increment_attribute(var[pd], -steps[pd])
        # print(f"exterior = {cal.ext_par}")

    cal.ext_par.update_rotation_matrix()

    return (x_ders, y_ders)


def orient(
    cal: Calibration,
    cpar: ControlPar,
    nfix: int,
    fix: np.ndarray,
    pix: List[Target],
    flags: OrientPar,
    sigmabeta: np.ndarray,
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
    fix : List[vec3d]
        each of nfix items is one 3D position of known point on
        the calibration object.
    pix : List[target]
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

    dm: float = 0.000001
    drad: float = 0.0000001

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

    glass_dir = vec_set(cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z)
    nGl = vec_norm(glass_dir)

    e1_x = 2 * cal.glass_par.vec_z - 3 * cal.glass_par.vec_x
    e1_y = 3 * cal.glass_par.vec_x - 1 * cal.glass_par.vec_z
    e1_z = 1 * cal.glass_par.vec_y - 2 * cal.glass_par.vec_y
    tmp_vec = vec_set(e1_x, e1_y, e1_z)
    e1 = unit_vector(tmp_vec)

    e2_x = e1_y * cal.glass_par.vec_z - e1_z * cal.glass_par.vec_x
    e2_y = e1_z * cal.glass_par.vec_x - e1_x * cal.glass_par.vec_z
    e2_z = e1_x * cal.glass_par.vec_y - e1_y * cal.glass_par.vec_y
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
        cal.added_par.k1,
        cal.added_par.k2,
        cal.added_par.k3,
        cal.added_par.p1,
        cal.added_par.p2,
        cal.added_par.scx,
        cal.added_par.she,
    ]

    # backup for changing back and forth
    safety_x = cal.glass_par.vec_x
    safety_y = cal.glass_par.vec_y
    safety_z = cal.glass_par.vec_z

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
            cal.ext_par.update_rotation_matrix()
            xp, yp = img_coord(fix[i], cal, cpar.mm)

            # derivatives of distortion parameters
            r = np.sqrt(xp * xp + yp * yp)

            X[n][7] = cal.added_par.scx
            X[n + 1][7] = np.sin(cal.added_par.she)

            X[n][8] = 0
            X[n + 1][8] = 1

            X[n][9] = cal.added_par.scx * xp * r * r
            X[n + 1][9] = yp * r * r

            X[n][10] = cal.added_par.scx * xp * pow(r, 4)
            X[n + 1][10] = yp * pow(r, 4)

            X[n][11] = cal.added_par.scx * xp * pow(r, 6)
            X[n + 1][11] = yp * pow(r, 6)

            X[n][12] = cal.added_par.scx * (2 * xp * xp + r * r)
            X[n + 1][12] = 2 * xp * yp

            X[n][13] = 2 * cal.added_par.scx * xp * yp
            X[n + 1][13] = 2 * yp * yp + r * r

            qq = cal.added_par.k1 * r * r
            qq += cal.added_par.k2 * pow(r, 4)
            qq += cal.added_par.k3 * pow(r, 6)
            qq += 1
            X[n][14] = (
                xp * qq
                + cal.added_par.p1 * (r * r + 2 * xp * xp)
                + 2 * cal.added_par.p2 * xp * yp
            )
            X[n + 1][14] = 0

            X[n][15] = -np.cos(cal.added_par.she) * yp
            X[n + 1][15] = -np.sin(cal.added_par.she) * yp

            # numeric derivatives of projection coordinates over external parameters,
            # 3D position and the angles
            X[n][:6], X[n + 1][:6] = num_deriv_exterior(cal, cpar, dm, drad, fix[i])

            # Num. deriv. of projection coords over sensor distance from PP
            cal.int_par.cc += dm
            cal.ext_par.update_rotation_matrix()
            xpd, ypd = img_coord(fix[i], cal, cpar.mm)
            X[n][6] = (xpd - xp) / dm
            X[n + 1][6] = ypd - yp
            # for i in range(len(fix)):
            #     dm = 0.0001
            #     xp, yp = 0.0, 0.0
            #     xc, yc = fix[i][0], fix[i][1]
            #     al, be, ga = cal.alpha, cal.beta, cal.gamma
            #     safety_x, safety_y, safety_z = cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z
            #     nGl = cal.glass_par.n / cal.air_par.n

            cal.int_par.cc -= dm

            # al += dm
            cal.glass_par.vec_x += e1[0] * nGl * dm
            cal.glass_par.vec_y += e1[1] * nGl * dm
            cal.glass_par.vec_z += e1[2] * nGl * dm
            xpd, ypd = img_coord(fix[i], cal, cpar.mm)
            X[n][16] = (xpd - xp) / dm
            X[n + 1][16] = (ypd - yp) / dm
            # al -= dm
            cal.glass_par.vec_x = safety_x
            cal.glass_par.vec_y = safety_y
            cal.glass_par.vec_z = safety_z

            # be += dm
            cal.glass_par.vec_x += e2[0] * nGl * dm
            cal.glass_par.vec_y += e2[1] * nGl * dm
            cal.glass_par.vec_z += e2[2] * nGl * dm
            xpd, ypd = img_coord(fix[i], cal, cpar.mm)
            X[n][17] = (xpd - xp) / dm
            X[n + 1][17] = (ypd - yp) / dm
            # be -= dm
            cal.glass_par.vec_x = safety_x
            cal.glass_par.vec_y = safety_y
            cal.glass_par.vec_z = safety_z

            # ga += dm
            cal.glass_par.vec_x += cal.glass_par.vec_x * nGl * dm
            cal.glass_par.vec_y += cal.glass_par.vec_y * nGl * dm
            cal.glass_par.vec_z += cal.glass_par.vec_z * nGl * dm
            xpd, ypd = img_coord(fix[i], cal, cpar.mm)
            X[n][18] = (xpd - xp) / dm
            X[n + 1][18] = (ypd - yp) / dm
            # ga -= dm
            cal.glass_par.vec_x = safety_x
            cal.glass_par.vec_y = safety_y
            cal.glass_par.vec_z = safety_z

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
        y[n_obs + 3] = ident[3] - cal.added_par.k1
        y[n_obs + 4] = ident[4] - cal.added_par.k2
        y[n_obs + 5] = ident[5] - cal.added_par.k3
        y[n_obs + 6] = ident[6] - cal.added_par.p1
        y[n_obs + 7] = ident[7] - cal.added_par.p2
        y[n_obs + 8] = ident[8] - cal.added_par.scx
        y[n_obs + 9] = ident[9] - cal.added_par.she

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

        beta, residuals, rank, singular_values = np.linalg.lstsq(
            Xh[:, :numbers], yh, rcond=None
        )

        # Interpret the results
        print(
            f"Coefficients (beta): {beta} \n \
                Residuals: {residuals} \n \
                singular_values: {singular_values} \n \
                rank: {rank} \n \
            "
        )

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
        cal.added_par.k1 += beta[9]
        cal.added_par.k2 += beta[10]
        cal.added_par.k3 += beta[11]
        cal.added_par.p1 += beta[12]
        cal.added_par.p2 += beta[13]
        cal.added_par.scx += beta[14]
        cal.added_par.she += beta[15]

        if flags.interfflag:
            cal.glass_par.vec_x += e1[0] * nGl * beta[16]
            cal.glass_par.vec_y += e1[1] * nGl * beta[16]
            cal.glass_par.vec_z += e1[2] * nGl * beta[16]
            cal.glass_par.vec_x += e2[0] * nGl * beta[17]
            cal.glass_par.vec_y += e2[1] * nGl * beta[17]
            cal.glass_par.vec_z += e2[2] * nGl * beta[17]

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

    XPX = np.linalg.inv(np.dot(X[:, :numbers].T, X[:, :numbers]))

    for i in range(numbers):
        sigmabeta[i] = sigmabeta[NPAR] * np.sqrt(XPX[i][i])

    if stopflag:
        cal.ext_par.update_rotation_matrix()
        return resi
    else:
        return None


def raw_orient(
    cal: Calibration,
    cpar: ControlPar,
    nfix: int,
    fix: np.ndarray,
    pix: List[Target],
) -> bool:
    """Calculate orientation of the camera, updating its calibration."""
    X = np.zeros((10, 6))
    y = np.zeros(10)
    # XPX = np.zeros((6, 6))
    # XPy = np.zeros(6)
    beta = np.zeros(6)
    dm = 0.0001
    drad = 0.0001
    pos = np.zeros(3)

    cal.added_par.k1 = 0
    cal.added_par.k2 = 0
    cal.added_par.k3 = 0
    cal.added_par.p1 = 0
    cal.added_par.p2 = 0
    cal.added_par.scx = 1
    cal.added_par.she = 0

    itnum = 0
    stopflag = False

    while stopflag == 0 and itnum < 20:
        itnum += 1
        n = 0
        for i in range(nfix):
            xc, yc = pixel_to_metric(pix[i].x, pix[i].y, cpar)
            pos = fix[i]
            cal.ext_par.update_rotation_matrix()
            xp, yp = img_coord(pos, cal, cpar.mm)
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
            X, y
        )  # , rcond=None)

        # Interpret the results
        print("Coefficients (beta):", beta)
        print("Residuals:", residuals)
        print("rank:", rank)
        print("singular_values:", singular_values)

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
        cal.ext_par.update_rotation_matrix()

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
    img_pts: TargetArray,
    cparam: ControlPar,
    orient_par: OrientPar,
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
    residuals = orient(cal, cparam, len(ref_pts), ref_pts, img_pts, orient_par, err_est)

    # free(orip)

    if residuals is None:
        # free(residuals)
        raise ValueError("Orientation iteration failed, need better setup.")

    ret = np.empty((len(img_pts), 2))
    used = np.empty(len(img_pts), dtype=np.int_)

    for ix, img_pt in enumerate(img_pts):
        ret[ix] = (residuals[2 * ix], residuals[2 * ix + 1])
        used[ix] = img_pt.pnr

    # free(residuals)
    return ret, used, err_est


def match_detection_to_ref(
    cal: Calibration,
    ref_pts: np.ndarray,
    img_pts: TargetArray,
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
    targets: np.ndarray, cparam: ControlPar, cals: List[Calibration], vparam: VolumePar
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
    ControlParams cparam - needed for the parameters of the tank through which
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
        res, rcm = single_cam_point_positions(targets, cparam, cals, vparam)
    elif len(cals) > 1:
        res, rcm = multi_cam_point_positions(targets, cparam, cals)
    else:
        raise ValueError("wrong number of cameras in point_positions")

    return res, rcm


def single_cam_point_positions(
    targets: np.ndarray, cparam: ControlPar, cals: List[Calibration], vparam: VolumePar
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
        res[pt, :] = epi_mm_2D(targ[0][0], targ[0][1], cals[0], cparam.mm, vparam)

    return res, rcm


def multi_cam_point_positions(
    targets: np.ndarray, cparam: ControlPar, cals: List[Calibration]
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

    num_targets = targets.shape[0]
    num_cams = targets.shape[1]
    res = np.empty((num_targets, 3))
    rcm = np.empty(num_targets)

    for pt in range(num_targets):
        rcm[pt], res = point_position(targets[pt], num_cams, cparam.mm, cals)

    return res, rcm
