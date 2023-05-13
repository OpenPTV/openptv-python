import copy
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
from .vec_utils import norm, vec3d, vec_set


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
    n_iter = 40

    rdiff = 0.1
    it = 0

    # 1-medium case
    if mm.n1 == 1 and mm.nlay == 1 and mm.n2[0] == 1 and mm.n3 == 1:
        return 1.0

    # interpolation using the existing mmlut
    if cal.mmlut.data is not None:
        mmf = get_mmf_from_mmlut(cal, pos)
        if mmf > 0:
            return mmf

    r = norm(pos[0] - cal.ext_par.x0, pos[1] - cal.ext_par.y0, 0)
    if r == 0:
        return 1.0

    # Extra layers protrude into water side:
    zout = pos[2] + np.sum(mm.d)

    rq = r.copy()

    beta2 = [1.0] * mm.nlay

    while np.abs(rdiff) > 0.001 and it < n_iter:
        beta1 = (cal.ext_par.z0 - pos[2]) and math.atan(rq / (cal.ext_par.z0 - pos[2]))
        for i in range(mm.nlay):
            beta2[i] = math.asin(math.sin(beta1) * mm.n1 / mm.n2[i])
        beta3 = math.asin(math.sin(beta1) * mm.n1 / mm.n3)
        rbeta = (cal.ext_par.z0 - mm.d[0]) * math.tan(beta1) - zout * math.tan(beta3)
        for i in range(1, mm.nlay):
            rbeta += mm.d[i] * math.tan(beta2[i])

        rdiff = r - rbeta
        # print(beta1, beta2,beta3, r, rbeta, rdiff)
        rq += rdiff
        it += 1

    if it >= n_iter:
        print(f"multimed_r_nlay stopped after {n_iter} iterations\n")
        return 1.0

    return rq / r


def trans_cam_point(
    ex: Exterior, mm: MultimediaPar, glass: Glass, pos: np.ndarray, ex_t: Exterior
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform the camera and point coordinates to the glass coordinates.

    ex = Exterior(x0=ex_x, y0=ex_y, z0=ex_z)
    mm = MultimediaPar(d=mm_d)
    glass = Glass(vec_x=gl_vec_x, vec_y=gl_vec_y, vec_z=gl_vec_z)
    pos = np.array([pos_x, pos_y, pos_z])

    pos_t, cross_p, cross_c = trans_cam_point(ex, mm, glass, pos, ex_t)
    """
    glass_dir = np.array([glass.vec_x, glass.vec_y, glass.vec_z])
    primary_point = np.array([ex.x0, ex.y0, ex.z0])

    dist_o_glass = np.linalg.norm(glass_dir)  # vector length
    dist_cam_glas = (
        np.dot(primary_point, glass_dir) / dist_o_glass - dist_o_glass - mm.d
    )

    dist_point_glass = np.dot(pos, glass_dir) / dist_o_glass - dist_o_glass

    renorm_glass = glass_dir * (dist_cam_glas / dist_o_glass)
    cross_c = primary_point - renorm_glass

    renorm_glass = glass_dir * (dist_point_glass / dist_o_glass)
    cross_p = pos - renorm_glass

    ex_t.x0 = 0.0
    ex_t.y0 = 0.0
    ex_t.z0 = dist_cam_glas + mm.d[0]

    renorm_glass = glass_dir * (mm.d / dist_o_glass)
    temp = cross_c - renorm_glass
    temp = cross_p - temp
    pos_t = np.r_[np.linalg.norm(temp), 0, dist_point_glass]

    return pos_t, cross_p, cross_c


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

    Returns:
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
    if direct[2] == 0:
        direct[2] = 1  # avoid division by zero

    out[0] = vertex[0] + (glob_Z - vertex[2]) * direct[0] / direct[2]
    out[1] = vertex[1] + (glob_Z - vertex[2]) * direct[1] / direct[2]
    out[2] = glob_Z
    return out


def init_mmlut(vpar: VolumePar, cpar: ControlPar, cal: Calibration) -> Calibration:
    """Initialize the multilayer lookup table."""
    rw = 2.0
    Rmax = 0.0

    # image corners
    xc = [0.0, float(cpar.imx)]
    yc = [0.0, float(cpar.imy)]

    # find extrema of imaged object volume
    Zmin = min(vpar.Zmin_lay)
    Zmax = max(vpar.Zmax_lay)

    Zmin -= math.fmod(Zmin, rw)
    Zmax += rw - math.fmod(Zmax, rw)

    Zmin_t = Zmin
    Zmax_t = Zmax

    # intersect with image vertices rays
    cal_t = copy.deepcopy(cal)

    for i in range(2):
        for j in range(2):
            print(i, j)
            x, y = pixel_to_metric(xc[i], yc[j], cpar)
            print(x, y)
            x -= cal.int_par.xh
            y -= cal.int_par.yh
            x, y = correct_brown_affine(x, y, cal.added_par)
            print(f"corrected {x},{y}")
            print(f"cal = {cal}")
            print(f"cpar.mm = {cpar.mm}")
            pos, a = ray_tracing(x, y, cal, cpar.mm)
            print(f"pos = {pos}, a = {a}")
            xyz = move_along_ray(Zmin, pos, a)
            print(xyz)
            xyz_t, _, _ = trans_cam_point(
                cal.ext_par, cpar.mm, cal.glass_par, xyz, cal_t.ext_par
            )
            print(xyz_t)

            if xyz_t[2] < Zmin_t:
                Zmin_t = xyz_t[2]
            if xyz_t[2] > Zmax_t:
                Zmax_t = xyz_t[2]

            R = norm(xyz_t[0] - cal.ext_par.x0, xyz_t[1] - cal.ext_par.y0, 0)

            if R > Rmax:
                Rmax = R

            xyz = move_along_ray(Zmax, pos, a)
            xyz_t, _, _ = trans_cam_point(
                cal.ext_par, cpar.mm, cal.glass_par, xyz, cal_t.ext_par
            )

            if xyz_t[2] < Zmin_t:
                Zmin_t = xyz_t[2]
            if xyz_t[2] > Zmax_t:
                Zmax_t = xyz_t[2]

            R = norm(xyz_t[0] - cal.ext_par.x0, xyz_t[1] - cal.ext_par.y0, 0)

            if R > Rmax:
                Rmax = R

    # round values (-> enlarge)
    Rmax += rw - math.fmod(Rmax, rw)

    print(f"inside init_mmlut: {Rmax}, {Zmin_t}, {Zmax_t}")

    # get # of rasterlines in r, z
    nr = int(Rmax / rw + 1)
    nz = int((Zmax_t - Zmin_t) / rw + 1)

    # create two dimensional mmlut structure
    cal.mmlut.origin = np.r_[cal_t.ext_par.x0, cal_t.ext_par.y0, Zmin_t]
    cal.mmlut.nr = nr
    cal.mmlut.nz = nz
    cal.mmlut.rw = rw

    if cal.mmlut.data is None:
        data = np.empty(nr * nz, dtype=np.float64)
        Ri = np.arange(nr) * rw
        Zi = np.arange(nz) * rw + Zmin_t

        for i in range(nr):
            for j in range(nz):
                xyz = vec_set(Ri[i] + cal_t.ext_par.x0, cal_t.ext_par.y0, Zi[j])
                data[i * nz + j] = multimed_r_nlay(cal_t, cpar.mm, xyz)

        cal.mmlut.data = data

    return cal


def get_mmf_from_mmlut(cal: Calibration, pos: np.ndarray) -> float:
    """Get the refractive index of the medium at a given position."""
    rw = cal.mmlut.rw
    origin = cal.mmlut.origin
    data = cal.mmlut.data
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

                pos, a = ray_tracing(x, y, cal[i_cam], cpar.mm)

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
