import math
from typing import List, Tuple

import numpy as np
from numba import njit

from .calibration import Calibration, Exterior, Glass
from .parameters import (
    ControlPar,
    MultimediaPar,
    VolumePar,
)
from .ray_tracing import ray_tracing
from .trafo import correct_brown_affine, pixel_to_metric
from .vec_utils import norm, vec_dot, vec_norm, vec_set


def multimed_nlay(
    cal: Calibration, mm: MultimediaPar, pos: np.ndarray
) -> Tuple[float, float]:
    """Create the Xq,Yq points for each X,Y point in the image space.

    using radial shift from the multimedia model
    """
    radial_shift = multimed_r_nlay(cal, mm, pos)
    Xq =  cal.ext_par.x0 + (pos[0] -  cal.ext_par.x0) * radial_shift
    Yq =  cal.ext_par.y0 + (pos[1] -  cal.ext_par.y0) * radial_shift
    return Xq, Yq


def multimed_r_nlay(cal: Calibration, mm: MultimediaPar, pos: np.ndarray) -> float:
    """Calculate the radial shift for the multimedia model."""
    # 1-medium case
    if mm.n1 == 1 and mm.nlay == 1 and mm.n2[0] == 1 and mm.n3 == 1:
        return 1.0

    #  interpolation using the existing mmlut
    if cal.mmlut.data is not None:
        print("going into get_mmf_from_mmlut\n")
        mmf = get_mmf_from_mmlut(cal, pos)
        if mmf > 0:
            print(f"mmf from data = {mmf}")
            return mmf

    mmf = fast_multimed_r_nlay(
        mm.nlay,
        mm.n1,
        np.array(mm.n2),
        mm.n3,
        np.array(mm.d),
        cal.ext_par.x0,
        cal.ext_par.y0,
        cal.ext_par.z0,
        pos)
    print(f"mmf from a loop = {mmf}")

    return mmf


@njit
def fast_multimed_r_nlay(
    nlay:int,
    n1: float,
    n2: np.ndarray,
    n3: float,
    d: np.ndarray,
    x0,
    y0,
    z0,
    pos: np.ndarray
    ) -> float:
    """Calculate the radial shift for the multimedia model.

    mm = MultimediaPar.to_dict()
    mm = {'nlay': 2, 'n1': 1, 'n2': [1.49, 0.0, 0.0], 'd': [5.0, 0.0, 0.0], 'n3': 1.33}
    data = cal.mmlut.data (np.ndarray)

    x0,y0,z0 =  x0, y0, z0

    """
    n_iter = 40
    rdiff = 0.1
    beta2 = np.zeros(nlay, dtype=np.float64)

    X, Y, Z = pos
    zout = Z
    for i in range(1, nlay):
        zout += d[i]

    r = np.linalg.norm(np.array([X-x0, Y-y0, 0], dtype=np.float64))
    rq = r

    it = 0
    while (rdiff > 0.001 or rdiff < -0.001) and it < n_iter:
        zdiff =  z0 - Z
        if zdiff == 0:
            zdiff = 1.0
        beta1 = math.atan(rq / zdiff)
        for i in range(nlay):
            beta2[i] = math.asin(math.sin(beta1) * n1 / n2[i])
        beta3 = math.asin(math.sin(beta1) * n1 / n3)

        rbeta = ( z0 - d[0]) * math.tan(beta1) - zout * math.tan(beta3)
        for i in range(nlay):
            rbeta += d[i] * math.tan(beta2[i])

        rdiff = r - rbeta
        rq += rdiff
        it += 1

    if it >= n_iter:
        print("multimed_r_nlay stopped after", n_iter, "iterations")
        return 1.0

    return 1.0 if r == 0 else float(rq / r)


def trans_cam_point(
    ex: Exterior, mm: MultimediaPar, glass: Glass, pos: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Transform the camera and point coordinates to the glass coordinates.

    ex = Exterior(x0=ex_x, y0=ex_y, z0=ex_z)
    mm = MultimediaPar(d=mm_d)
    glass = Glass(vec_x=gl_vec_x, vec_y=gl_vec_y, vec_z=gl_vec_z)
    pos = np.array([pos_x, pos_y, pos_z])

    pos_t, cross_p, cross_c = trans_cam_point(ex, mm, glass, pos, ex_t)
    """
    glass_dir = np.array([glass.vec_x, glass.vec_y, glass.vec_z])
    primary_point = np.array([ex.x0, ex.y0, ex.z0])

    dist_o_glass = vec_norm(glass_dir)  # vector length
    dist_cam_glas = (
        np.dot(primary_point, glass_dir) / dist_o_glass - dist_o_glass - mm.d[0]
    )

    dist_point_glass = vec_dot(pos, glass_dir) / dist_o_glass - dist_o_glass

    renorm_glass = glass_dir * (dist_cam_glas / dist_o_glass)
    cross_c = primary_point - renorm_glass

    renorm_glass = glass_dir * (dist_point_glass / dist_o_glass)
    cross_p = pos - renorm_glass

    z0 = dist_cam_glas + mm.d[0]

    renorm_glass = glass_dir * (mm.d[0] / dist_o_glass)
    temp = cross_c - renorm_glass
    temp = cross_p - temp
    pos_t = np.r_[np.linalg.norm(temp), 0, dist_point_glass]

    return pos_t, cross_p, cross_c, z0


def back_trans_point(
    pos_t: np.ndarray,
    mm: MultimediaPar,
    G: Glass,
    cross_p: np.ndarray,
    cross_c: np.ndarray,
) -> np.ndarray:
    """
    Transform the point coordinates from the glass to the camera coordinates.

    Args:
    ----
        pos_t: A numpy array representing the position of the point in the glass coordinate system.
        mm: SomeType (TODO: specify type). A parameter used to scale the glass direction vector.
        G: A Glass object representing the glass coordinate system.
        cross_p: A numpy array representing the position of the point in the pixel coordinate system.
        cross_c: A numpy array representing the position of the point in the camera coordinate system.

    Returns
    -------
        A numpy array representing the position of the point in the camera coordinate system.
    """
    # Calculate the glass direction vector
    glass_direction = np.array([G.vec_x, G.vec_y, G.vec_z])
    norm_glass_direction = np.linalg.norm(glass_direction)

    # Normalize the glass direction vector
    renorm_glass = glass_direction * (mm.d[0] / norm_glass_direction)

    # Calculate the position of the point after passing through the glass
    after_glass = cross_c - renorm_glass

    # Calculate the vector between the point in the glass and the point after passing through the glass
    temp = cross_p - after_glass

    # Calculate the norm of the vector temp
    norm_temp = np.linalg.norm(temp)

    # Calculate the position of the point in the camera coordinate system
    renorm_glass = glass_direction * (-pos_t[2] / norm_glass_direction)
    pos = after_glass - renorm_glass

    # If the norm of the vector temp is greater than zero, adjust the position
    # of the point in the camera coordinate system
    if norm_temp > 0:
        renorm_temp = temp * (-pos_t[0] / norm_temp)
        pos = pos - renorm_temp

    return pos


def move_along_ray(glob_z: float, vertex: np.ndarray, direct: np.ndarray) -> np.ndarray:
    """Move along the ray to the global z plane.

    move_along_ray() calculates the position of a point in a global Z value
    along a ray whose vertex and direction are given.

    Arguments:
    ---------
    double glob_z - the Z value of the result point in the global
        coordinate system.
    vec3d vertex - the ray vertex.
    vec3d direct - the ray direction, a unit vector.
    vec3d out - result buffer.

    """
    out = np.zeros(3, dtype=np.float64)
    if direct[2] == 0:
        direct[2] = 1  # avoid division by zero

    out[0] = vertex[0] + (glob_z - vertex[2]) * direct[0] / direct[2]
    out[1] = vertex[1] + (glob_z - vertex[2]) * direct[1] / direct[2]
    out[2] = glob_z

    return out


def init_mmlut(vpar: VolumePar, cpar: ControlPar, cal: Calibration) -> Calibration:
    """Initialize the multilayer lookup table."""
    rw = 2
    Rmax = 0.0

    # image corners
    xc = [0.0, float(cpar.imx)]
    yc = [0.0, float(cpar.imy)]

    # find extrema of imaged object volume
    z_min = min(vpar.z_min_lay)
    z_max = max(vpar.z_max_lay)

    z_min -= math.fmod(z_min, rw)
    z_max += rw - math.fmod(z_max, rw)

    z_min_t = z_min
    z_max_t = z_max

    # intersect with image vertices rays
    cal_t = Calibration(mmlut  = cal.mmlut)

    for i in range(2):
        for j in range(2):
            x, y = pixel_to_metric(xc[i], yc[j], cpar)
            x -= cal.int_par.xh
            y -= cal.int_par.yh
            x, y = correct_brown_affine(x, y, cal.added_par)
            pos, a = ray_tracing(x, y, cal, cpar.mm)
            xyz = move_along_ray(z_min, pos, a)
            xyz_t, _, _, cal_t.ext_par.z0 = trans_cam_point(
                cal.ext_par, cpar.mm, cal.glass_par, xyz
            )

            if xyz_t[2] < z_min_t:
                z_min_t = xyz_t[2]
            if xyz_t[2] > z_max_t:
                z_max_t = xyz_t[2]

            R = norm(xyz_t[0] - cal_t.ext_par.x0, xyz_t[1] - cal_t.ext_par.y0, 0)

            if R > Rmax:
                Rmax = R

            xyz = move_along_ray(z_max, pos, a)
            xyz_t, _, _, cal_t.ext_par.z0 = trans_cam_point(
                cal.ext_par, cpar.mm, cal.glass_par, xyz
            )

            if xyz_t[2] < z_min_t:
                z_min_t = xyz_t[2]
            if xyz_t[2] > z_max_t:
                z_max_t = xyz_t[2]

            R = norm(xyz_t[0] - cal_t.ext_par.x0, xyz_t[1] - cal_t.ext_par.y0, 0)

            if R > Rmax:
                Rmax = R

    # round values (-> enlarge)
    Rmax += rw - math.fmod(Rmax, rw)

    # get # of rasterlines in r, z
    nr = int(Rmax / rw + 1)
    nz = int((z_max_t - z_min_t) / rw + 1)

    # create two dimensional mmlut structure
    cal.mmlut.origin = np.r_[cal_t.ext_par.x0, cal_t.ext_par.y0, z_min_t]
    cal.mmlut.nr = nr
    cal.mmlut.nz = nz
    cal.mmlut.rw = rw

    if cal.mmlut.data is None:
        data = np.empty((nr, nz), dtype=np.float64)
        Ri = np.arange(nr) * rw
        Zi = np.arange(nz) * rw + z_min_t

        for i in range(nr):
            for j in range(nz):
                xyz = vec_set(Ri[i] + cal_t.ext_par.x0, cal_t.ext_par.y0, Zi[j])
                data.flat[i * nz + j] = multimed_r_nlay(cal_t, cpar.mm, xyz)

        cal.mmlut.data = data

    return cal


def get_mmf_from_mmlut(cal: Calibration, pos: np.ndarray) -> float:
    """Get the refractive index of the medium at a given position."""
    rw = cal.mmlut.rw
    origin = cal.mmlut.origin
    data = cal.mmlut.data.flat  # type: ignore
    nz = cal.mmlut.nz
    nr = cal.mmlut.nr

    temp = pos - origin
    sz = temp[2] / rw
    iz = int(sz)
    sz -= iz

    R = norm(temp[0], temp[1], 0)
    sr = R / rw
    ir = int(sr)
    sr -= ir

    if ir > nr:
        return 0.0
    if iz < 0 or iz > nz:
        return 0.0

    # bilinear interpolation in r/z box
    # get vertices of box
    v4 = [
        ir * nz + iz,
        ir * nz + (iz + 1),
        (ir + 1) * nz + iz,
        (ir + 1) * nz + (iz + 1),
    ]

    # 2. check wther point is inside camera's object volume
    # important for epipolar line computation
    for i in range(4):
        if v4[i] < 0 or v4[i] > nr * nz:
            return 0.0

    # interpolate
    mmf = (
        data[v4[0]] * (1 - sr) * (1 - sz)
        + data[v4[1]] * (1 - sr) * sz
        + data[v4[2]] * sr * (1 - sz)
        + data[v4[3]] * sr * sz
    )

    return mmf


def volumedimension(
    xmax: float,
    xmin: float,
    ymax: float,
    ymin: float,
    z_max: float,
    z_min: float,
    vpar: VolumePar,
    cpar: ControlPar,
    cal: List[Calibration],
) -> Tuple[float, float, float, float, float, float]:
    """Calculate the volume dimensions."""
    xc = [0.0, cpar.imx]
    yc = [0.0, cpar.imy]

    z_min = vpar.z_min_lay[0]
    z_max = vpar.z_max_lay[0]

    if vpar.z_min_lay[1] < z_min:
        z_min = vpar.z_min_lay[1]
    if vpar.z_max_lay[1] > z_max:
        z_max = vpar.z_max_lay[1]

    for i_cam in range(cpar.num_cams):
        for i in range(2):
            for j in range(2):
                x, y = pixel_to_metric(xc[i], yc[j], cpar)

                x -= cal[i_cam].int_par.xh
                y -= cal[i_cam].int_par.yh

                x, y = correct_brown_affine(x, y, cal[i_cam].added_par)

                pos, a = ray_tracing(x, y, cal[i_cam], cpar.mm)

                # TODO: seems that it should be + pos[2] instead of - pos[2]
                X = pos[0] + (z_min + pos[2]) * a[0] / a[2]
                Y = pos[1] + (z_min + pos[2]) * a[1] / a[2]

                if X > xmax:
                    xmax = X
                if X < xmin:
                    xmin = X
                if Y > ymax:
                    ymax = Y
                if Y < ymin:
                    ymin = Y

                X = pos[0] + (z_max - pos[2]) * a[0] / a[2]
                Y = pos[1] + (z_max - pos[2]) * a[1] / a[2]

                if X > xmax:
                    xmax = X
                if X < xmin:
                    xmin = X
                if Y > ymax:
                    ymax = Y
                if Y < ymin:
                    ymin = Y

    return (xmax, xmin, ymax, ymin, z_max, z_min)
