import math
from typing import List, Tuple

import numpy as np

from .calibration import Calibration, Exterior, Glass
from .parameters import (
    ControlPar,
    MultimediaPar,
    VolumePar,
)
from .ray_tracing import ray_tracing
from .trafo import correct_brown_affine, pixel_to_metric
from .vec_utils import norm, vec3d, vec_set, vec_subt


def multimed_nlay(
    cal: Calibration, mm: MultimediaPar, pos: np.ndarray
) -> Tuple[float, float]:
    """Create the Xq,Yq points for each X,Y point in the image space.

    using radial shift from the multimedia model
    """
    radial_shift = multimed_r_nlay(cal, mm, pos)
    Xq = cal.ext_par.x0 + (pos[0] - cal.ext_par.x0) * radial_shift
    Yq = cal.ext_par.y0 + (pos[1] - cal.ext_par.y0) * radial_shift
    return Xq, Yq


def multimed_r_nlay(cal: Calibration, mm: MultimediaPar, pos: np.ndarray) -> float:
    """Calculate the radial shift for the multimedia model."""
    i, it = 0, 0
    n_iter = 40
    beta1, beta2, beta3, r, rbeta, rdiff, rq, mmf = (
        0.0,
        [0.0] * 32,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
    )
    X, Y, Z = 0.0, 0.0, 0.0
    zout = 0.0

    # 1-medium case
    if mm.n1 == 1 and mm.nlay == 1 and mm.n2[0] == 1 and mm.n3 == 1:
        return 1.0

    # interpolation using the existing mmlut
    if cal.mmlut.data is not None:
        mmf = get_mmf_from_mmlut(cal, pos)
        if mmf > 0:
            return mmf

    # iterative procedure
    X = pos[0]
    Y = pos[1]
    Z = pos[2]

    # Extra layers protrude into water side:
    zout = Z + np.sum(mm.d[: mm.nlay])
    # for i in range(1, mm.nlay):
    #     zout += mm.d[i]

    r = norm(X - cal.ext_par.x0, Y - cal.ext_par.y0, 0)
    rq = r.copy()

    while np.abs(rdiff) > 0.001 and it < n_iter:
        beta1 = math.atan(rq / (cal.ext_par.z0 - Z))
        for i in range(0, mm.nlay):
            beta2[i] = math.asin(math.sin(beta1) * mm.n1 / mm.n2[i])
        beta3 = math.asin(math.sin(beta1) * mm.n1 / mm.n3)
        rbeta = (cal.ext_par.z0 - mm.d[0]) * math.tan(beta1) - zout * math.tan(beta3)
        for i in range(0, mm.nlay):
            rbeta += mm.d[i] * math.tan(beta2[i])
        rdiff = r - rbeta
        rq += rdiff
        it += 1

    if it >= n_iter:
        print(f"multimed_r_nlay stopped after {n_iter} iterations\n")
        return 1.0

    if r != 0:
        return rq / r
    else:
        return 1.0


def trans_Cam_Point(
    ex: Exterior, mm: MultimediaPar, glass: Glass, pos: np.ndarray
) -> Tuple[Exterior, np.ndarray, np.ndarray, np.ndarray]:
    """Transform the camera and point coordinates to the glass coordinates.

    ex = np.array([ex_x, ex_y, ex_z])
    mm = np.array([mm_d])
    glass_dir = np.array([gl_vec_x, gl_vec_y, gl_vec_z])
    pos = np.array([pos_x, pos_y, pos_z])

    ex_t, pos_t, cross_p, cross_c = trans_Cam_Point(ex, mm, glass_dir, pos)
    """
    glass_dir = np.array([glass.vec_x, glass.vec_y, glass.vec_z])
    primary_point = np.array([ex.x0, ex.y0, ex.z0])

    dist_o_glass = np.linalg.norm(glass_dir)  # vector length
    dist_cam_glas = (
        np.dot(primary_point, glass_dir) / dist_o_glass - dist_o_glass - mm.d[0]
    )
    dist_point_glass = np.dot(pos, glass_dir) / dist_o_glass - dist_o_glass

    renorm_glass = glass_dir * (dist_cam_glas / dist_o_glass)
    cross_c = primary_point - renorm_glass

    renorm_glass = glass_dir * (dist_point_glass / dist_o_glass)
    cross_p = pos - renorm_glass

    ex_t = Exterior()
    ex_t.z0 = dist_cam_glas + mm.d[0]

    renorm_glass = glass_dir * (mm.d[0] / dist_o_glass)
    temp = cross_c - renorm_glass
    temp = cross_p - temp
    pos_t = np.array([np.linalg.norm(temp), 0, dist_point_glass])

    return ex_t, pos_t, cross_p, cross_c


def back_trans_Point(
    pos_t: np.ndarray, mm, G: Glass, cross_p: np.ndarray, cross_c: np.ndarray
) -> np.ndarray:
    """Transform the point coordinates from the glass to the camera coordinates."""
    glass_dir = np.array([G.vec_x, G.vec_y, G.vec_z])
    nGl = np.linalg.norm(glass_dir)

    renorm_glass = glass_dir * (mm.d[0] / nGl)
    after_glass = cross_c - renorm_glass
    temp = cross_p - after_glass
    nVe = np.linalg.norm(temp)

    renorm_glass = glass_dir * (-pos_t[2] / nGl)
    pos = after_glass - renorm_glass

    if nVe > 0:
        renorm_glass = temp * (-pos_t[0] / nVe)
        pos = pos - renorm_glass

    return pos


def move_along_ray(glob_Z: float, vertex: np.ndarray, direct: np.ndarray) -> vec3d:
    """Move along the ray to the global Z plane.

    move_along_ray() calculates the position of a point in a global Z value
    along a ray whose vertex and direction are given.

    Arguments:
    ---------
    double glob_Z - the Z value of the result point in the global
        coordinate system.
    vec3d vertex - the ray vertex.
    vec3d direct - the ray direction, a unit vector.
    vec3d out - result buffer.

    """
    out = np.zeros(3, dtype=np.float64)
    out[0] = vertex[0] + (glob_Z - vertex[2]) * direct[0] / direct[2]
    out[1] = vertex[1] + (glob_Z - vertex[2]) * direct[1] / direct[2]
    out[2] = glob_Z
    return out


def init_mmlut(vpar: VolumePar, cpar: ControlPar, cal: Calibration) -> np.ndarray:
    """Initialize the multilayer lookup table."""
    nr, nz = 0, 0
    Rmax, Zmax = 0, 0
    rw = 2.0

    cal_t = Calibration()

    # image corners
    xc = [0.0, cpar["imx"]]
    yc = [0.0, cpar["imy"]]

    # find extrema of imaged object volume
    Zmin = vpar["Zmin_lay"][0]
    Zmax = vpar["Zmax_lay"][0]
    Zmin_t = Zmin
    Zmax_t = Zmax

    for i in range(2):
        for j in range(2):
            x, y = pixel_to_metric(xc[i], yc[j], cpar)
            x -= cal["int_par"]["xh"]
            y -= cal["int_par"]["yh"]
            x, y = correct_brown_affine(x, y, cal.added_par)
            pos, a = ray_tracing(x, y, cal, cpar.mm)
            xyz = move_along_ray(Zmin, pos, a)
            _, xyz_t, _, _ = trans_Cam_Point(
                cal["ext_par"], cpar["mm"], cal["glass_par"], xyz
            )

            if xyz_t[2] < Zmin_t:
                Zmin_t = xyz_t[2]
            if xyz_t[2] > Zmax_t:
                Zmax_t = xyz_t[2]

            R = norm(xyz_t[0] - cal.ext_par.x0, xyz_t[1] - cal.ext_par.y0, 0)

            if R > Rmax:
                Rmax = R

            xyz = move_along_ray(Zmax, pos, a)
            _, xyz_t, _, _ = trans_Cam_Point(
                cal["ext_par"], cpar["mm"], cal["glass_par"], xyz
            )

            if xyz_t[2] < Zmin_t:
                Zmin_t = xyz_t[2]
            if xyz_t[2] > Zmax_t:
                Zmax_t = xyz_t[2]

            R = norm(
                xyz_t[0] - cal["ext_par"]["x0"], xyz_t[1] - cal["ext_par"]["y0"], 0
            )

            if R > Rmax:
                Rmax = R

    # round values (-> enlarge)
    Rmax += rw - (Rmax % rw)

    # get # of rasterlines in r, z
    nr = int(Rmax / rw + 1)
    nz = int((Zmax_t - Zmin_t) / rw + 1)

    # create two dimensional mmlut structure
    cal.mmlut.origin = [cal.ext_par.x0, cal.ext_par.y0, Zmin_t]
    cal.mmlut.nr = nr
    cal.mmlut.nz = nz
    cal.mmlut.rw = rw

    if cal.mmlut.data is None:
        data = np.empty((nr * nz,), dtype=np.float64)
        Ri = np.arange(nr) * rw
        Zi = np.arange(nz) * rw + Zmin_t

        for i in range(nr):
            for j in range(nz):
                xyz = vec_set(Ri[i] + cal_t.ext_par.x0, cal_t.ext_par.y0, Zi[j])
                data[i * nz + j] = multimed_r_nlay(cal_t, cpar.mm, xyz)

        cal.mmlut.data = data

    return cal


def get_mmf_from_mmlut(cal: Calibration, pos: np.ndarray) -> float:
    i, ir, iz, nr, nz, rw, v4 = 0, 0, 0, 0, 0, 0, [0, 0, 0, 0]
    mmf = 1.0
    temp = [0.0, 0.0, 0.0]

    rw = cal.mmlut.rw

    temp = vec_subt(pos, cal.mmlut.origin)
    rw = cal.mmlut.rw
    origin = cal.mmlut.origin
    data = cal.mmlut.data
    nz = cal.mmlut.nz
    nr = cal.mmlut.nr

    temp = pos - origin
    sz = temp[2] / rw
    iz = int(sz)
    sz -= iz

    R = np.sqrt(temp[0] ** 2 + temp[1] ** 2)
    sr = R / rw
    ir = int(sr)
    sr -= ir

    if ir > nr or iz < 0 or iz > nz:
        return 0.0

    v4 = [
        ir * nz + iz,
        ir * nz + (iz + 1),
        (ir + 1) * nz + iz,
        (ir + 1) * nz + (iz + 1),
    ]
    for i in range(4):
        if v4[i] < 0 or v4[i] > nr * nz:
            return 0.0

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
    zmax: float,
    zmin: float,
    vpar: VolumePar,
    cpar: ControlPar,
    cal: List[Calibration],
) -> Tuple[float, float, float, float, float, float]:
    xc = [0.0, cpar["imx"]]
    yc = [0.0, cpar["imy"]]

    Zmin = vpar["Zmin_lay"][0]
    Zmax = vpar["Zmax_lay"][0]

    if vpar["Zmin_lay"][1] < Zmin:
        Zmin = vpar["Zmin_lay"][1]
    if vpar["Zmax_lay"][1] > Zmax:
        Zmax = vpar["Zmax_lay"][1]

    zmin = Zmin
    zmax = Zmax

    for i_cam in range(cpar["num_cams"]):
        for i in range(2):
            for j in range(2):
                x, y = pixel_to_metric(xc[i], yc[j], cpar)

                x -= cal[i_cam]["int_par"]["xh"]
                y -= cal[i_cam]["int_par"]["yh"]

                x, y = correct_brown_affine(x, y, cal[i_cam]["added_par"])

                pos, a = ray_tracing(x, y, cal[i_cam], cpar["mm"])

                X = pos[0] + (Zmin - pos[2]) * a[0] / a[2]
                Y = pos[1] + (Zmin - pos[2]) * a[1] / a[2]

                if X > xmax:
                    xmax = X
                if X < xmin:
                    xmin = X
                if Y > ymax:
                    ymax = Y
                if Y < ymin:
                    ymin = Y

                X = pos[0] + (Zmax - pos[2]) * a[0] / a[2]
                Y = pos[1] + (Zmax - pos[2]) * a[1] / a[2]

                if X > xmax:
                    xmax = X
                if X < xmin:
                    xmin = X
                if Y > ymax:
                    ymax = Y
                if Y < ymin:
                    ymin = Y

    return (xmax, xmin, ymax, ymin, zmax, zmin)
