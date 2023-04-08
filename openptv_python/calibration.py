"""Calibration data structures and functions."""
import json
import math
import pathlib
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Exterior:
    """Exterior parameters."""

    dm: np.array = np.zeros((3, 3), dtype=float)
    omega: float = 0.0
    phi: float = 0.0
    kappa: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0


@dataclass
class Interior:
    """Interior parameters."""

    xh: float = 0.0
    yh: float = 0.0
    cc: float = 0.0


@dataclass
class Glass:
    """Glass vector."""

    vec_x: float = 0.0
    vec_y: float = 0.0
    vec_z: float = 0.0


@dataclass
class ap_52:
    """5+2 parameters for distortion correction."""

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    scx: float = 0.0
    she: float = 0.0


@dataclass
class mmlut:
    """3D lookup table for the mapping between the image plane and the object."""

    origin: np.array = np.zeros(3, dtype=float)
    nr: int = 0
    nz: int = 0
    rw: int = 0
    data: list[float] = None


@dataclass
class Calibration:
    """Calibration data structure."""

    ext_par: Exterior = Exterior()
    int_par: Interior = Interior()
    glass_par: Glass = Glass()
    added_par: ap_52 = ap_52()
    mmlut: mmlut = mmlut()


def write_ori(
    Ex: Exterior, In: Interior, G: Glass, ap: ap_52, filename: str, add_file: str = None
):
    """Write exterior and interior orientation.

    if available, also parameters for
    distortion corrections.

    Arguments:
    ---------
    Exterior Ex - exterior orientation.
    Interior I - interior orientation.
    Glass G - glass parameters.
    ap_52 addp - optional additional (distortion) parameters. NULL is fine if
       add_file is NULL.
    char *filename - path of file to contain interior, exterior and glass
       orientation data.
    char *add_file - path of file to contain added (distortions) parameters.
    """
    try:
        with open(filename, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(asdict(Ex), indent=4) + "\n")
            fp.write(json.dumps(asdict(In), indent=4) + "\n")
            fp.write(json.dumps(asdict(G), indent=4) + "\n")
            if ap:
                fp.write(json.dumps(asdict(ap), indent=4) + "\n")
    except IOError:
        print("Can't open file: {}".format(filename))
        return False

    if add_file is None:
        return True

    try:
        with open(add_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(asdict(ap), indent=4) + "\n")
    except IOError:
        print("Can't open file: {}".format(add_file))
        return False

    return True


def read_ori(filename: str) -> Calibration:
    """Read the orientation file and the additional parameters file."""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    ext_par = Exterior(**data["ext_par"])
    int_par = Interior(**data["int_par"])
    glass_par = Glass(**data["glass_par"])
    added_par = ap_52(**data["added_par"])
    mm_lut = mmlut(**data["mmlut"])

    return Calibration(
        ext_par=ext_par,
        int_par=int_par,
        glass_par=glass_par,
        added_par=added_par,
        mmlut=mm_lut,
    )


# def read_ori(Ex, In, G, ori_file, addp, add_file, add_fallback):
#     """Read the orientation file and the additional parameters file."""
#     try:
#         fp = open(ori_file, "r", encoding="utf-8")
#     except IOError:
#         print("Can't open ORI file: %s\n", ori_file)
#         return 0

#     # Exterior
#     scan_res = fp.readline().split()
#     Ex.x0, Ex.y0, Ex.z0, Ex.omega, Ex.phi, Ex.kappa = map(float, scan_res)
#     if len(scan_res) != 6:
#         return 0

#     # Exterior rotation matrix
#     for i in range(3):
#         scan_res = fp.readline().split()
#         Ex.dm[i] = list(map(float, scan_res))
#         if len(scan_res) != 3:
#             return 0

#     # Interior
#     scan_res = fp.readline().split()
#     In.xh, In.yh, In.cc = map(float, scan_res)
#     if len(scan_res) != 3:
#         return 0

#     # Glass
#     scan_res = fp.readline().split()
#     G.vec_x, G.vec_y, G.vec_z = map(float, scan_res)
#     if len(scan_res) != 3:
#         return 0
#     fp.close()

#     # Additional:
#     try:
#         fp = open(add_file, "r", encoding="utf-8")
#     except IOError:
#         if add_fallback:
#             try:
#                 fp = open(add_fallback, "r", encoding="utf-8")
#             except IOError:
#                 pass

#     if fp:
#         scan_res = fp.readline().split()
#         addp.k1, addp.k2, addp.k3, addp.p1, addp.p2, addp.scx, addp.she = map(
#             float, scan_res
#         )
#         fp.close()
#     else:
#         print("no addpar fallback used\n")  # Waits for proper logging.
#         addp.k1 = addp.k2 = addp.k3 = addp.p1 = addp.p2 = addp.she = 0.0
#         addp.scx = 1.0

#     return 1


def compare_exterior(e1: Exterior, e2: Exterior) -> bool:
    """Compare exterior orientation parameters."""
    return (
        np.allclose(e1.dm, e2.dm, atol=1e-6)
        and (e1.x0 == e2.x0)
        and (e1.y0 == e2.y0)
        and (e1.z0 == e2.z0)
        and (e1.omega == e2.omega)
        and (e1.phi == e2.phi)
        and (e1.kappa == e2.kappa)
    )


def compare_interior(i1: Interior, i2: Interior) -> bool:
    return i1.xh == i2.xh and i1.yh == i2.yh and i1.cc == i2.cc


def compare_glass(g1: Glass, g2: Glass):
    """Compare `Glass` parameters.

    objects that need to be compared. The function then returns `1` if all
    `vec_x`, `vec_y` and `vec_z` values of `g1` are equal to the corresponding
    values in `g2`. Else, the function returns `0`.

    Args:
    ----
        g1 (_type_): _description_
        g2 (_type_): _description_

    Returns:
    -------
        _type_: _description_
    """
    return g1.vec_x == g2.vec_x and g1.vec_y == g2.vec_y and g1.vec_z == g2.vec_z


def compare_addpar(a1, a2):
    return (
        (a1.k1 == a2.k1)
        and (a1.k2 == a2.k2)
        and (a1.k3 == a2.k3)
        and (a1.p1 == a2.p1)
        and (a1.p2 == a2.p2)
        and (a1.scx == a2.scx)
        and (a1.she == a2.she)
    )


def read_calibration(ori_file: pathlib.Path):
    """Read the orientation file including the added parameters."""
    ret = read_ori(ori_file)
    ret.ext_par = rotation_matrix(ret.ext_par)
    ret.mmlut.data = None  # no multimedia data yet
    return ret


def write_calibration(cal, ori_file, add_file):
    return write_ori(
        cal.ext_par, cal.int_par, cal.glass_par, cal.added_par, ori_file, add_file
    )


def rotation_matrix(Ex: Exterior) -> Exterior:
    """Calculate the necessary trigonometric functions to rotate the Dmatrix of Exterior Ex."""
    cp = math.cos(Ex.phi)
    sp = math.sin(Ex.phi)
    co = math.cos(Ex.omega)
    so = math.sin(Ex.omega)
    ck = math.cos(Ex.kappa)
    sk = math.sin(Ex.kappa)

    # Modify the Exterior Ex with the new Dmatrix
    Ex.dm[0][0] = cp * ck
    Ex.dm[0][1] = -cp * sk
    Ex.dm[0][2] = sp
    Ex.dm[1][0] = co * sk + so * sp * ck
    Ex.dm[1][1] = co * ck - so * sp * sk
    Ex.dm[1][2] = -so * cp
    Ex.dm[2][0] = so * sk - co * sp * ck
    Ex.dm[2][1] = so * ck + co * sp * sk
    Ex.dm[2][2] = co * cp

    return Ex
