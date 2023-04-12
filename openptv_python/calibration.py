"""Calibration data structures and functions."""
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .vec_utils import vec3d


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
    scx: float = 1.0  # scale is 1.0 by default
    she: float = 0.0


@dataclass
class mmlut:
    """3D lookup table for the mapping between the image plane and the object."""

    origin: vec3d = np.zeros(3, dtype=float)
    nr: int = 0
    nz: int = 0
    rw: int = 0
    data: np.ndarray | None = None


@dataclass
class Calibration:
    """Calibration data structure."""

    ext_par: Exterior = Exterior()
    int_par: Interior = Interior()
    glass_par: Glass = Glass()
    added_par: ap_52 = ap_52()
    mmlut: mmlut = mmlut()

    def from_file(self, ori_file: str, addpar_file: str):
        """
        Populate calibration fields from .ori and .addpar files.

        Arguments:
        ---------
        ori_file - path to file containing exterior, interior and glass
            parameters.
        add_file - optional path to file containing distortion parameters.
        fallback_file - optional path to file used in case ``add_file`` fails
            to open.
        """
        read_ori(ori_file, addpar_file)

    def set_pos(self, x_y_z_np):
        """
        Set exterior position.

        Parameter: x_y_z_np - numpy array of 3 elements for x, y, z.
        """
        if len(x_y_z_np) != 3:
            raise ValueError(
                "Illegal array argument "
                + x_y_z_np.__str__()
                + " for x, y, z. Expected array/list of 3 numbers"
            )
        self.ext_par.x0 = x_y_z_np[0]
        self.ext_par.y0 = x_y_z_np[1]
        self.ext_par.z0 = x_y_z_np[2]

    def get_pos(self):
        """Return array of 3 elements representing exterior's x, y, z."""
        ret_x_y_z_np = np.empty(3)
        ret_x_y_z_np[0] = self.ext_par.x0
        ret_x_y_z_np[1] = self.ext_par.y0
        ret_x_y_z_np[2] = self.ext_par.z0

        return ret_x_y_z_np

    def set_angles(self, o_p_k_np):
        """
        Set angles (omega, phi, kappa) and recalculates Dmatrix accordingly.

        Parameter: o_p_k_np - array of 3 elements.
        """
        if len(o_p_k_np) != 3:
            raise ValueError(
                f"Illegal array argument {o_p_k_np} for "
                "omega, phi, kappa. Expected array/list of 3 numbers"
            )
        self.ext_par.omega = o_p_k_np[0]
        self.ext_par.phi = o_p_k_np[1]
        self.ext_par.kappa = o_p_k_np[2]

        # recalculate the Dmatrix dm according to new angles
        self.ext_par = rotation_matrix(self.ext_par)

    def get_angles(self):
        """Return an array of 3 elements representing omega, phi, kappa."""
        ret_o_p_k_np = np.empty(3)
        ret_o_p_k_np[0] = self.ext_par.omega
        ret_o_p_k_np[1] = self.ext_par.phi
        ret_o_p_k_np[2] = self.ext_par.kappa

        return ret_o_p_k_np

    def get_rotation_matrix(self):
        """Return a 3x3 numpy array that represents Exterior's rotation matrix."""
        ret_dmatrix_np = np.empty(shape=(3, 3))
        for i in range(3):
            for j in range(3):
                ret_dmatrix_np[i][j] = self.ext_par.dm[i][j]

        return ret_dmatrix_np

    def set_primary_point(self, prim_point_pos: np.ndarray):
        """
        Set the camera's primary point position (a.k.a. interior orientation).

        Arguments:
        ---------
        prim_point_pos - a 3 element array holding the values of x and y shift
            of point from sensor middle and sensor-point distance, in this
            order.
        """
        if prim_point_pos.shape != (3,):
            raise ValueError("Expected a 3-element array")

        self.int_par.xh = prim_point_pos[0]
        self.int_par.yh = prim_point_pos[1]
        self.int_par.cc = prim_point_pos[2]

    def get_primary_point(self):
        """
        Return the primary point position (a.k.a. interior orientation) as a 3.

        element array holding the values of x and y shift of point from sensor
        middle and sensor-point distance, in this order.
        """
        ret = np.empty(3)
        ret[0] = self.int_par.xh
        ret[1] = self.int_par.yh
        ret[2] = self.int_par.cc
        return ret

    def set_radial_distortion(self, dist_coeffs: np.ndarray):
        """
        Set the parameters for the image radial distortion, where the x/y.

        coordinates are corrected by a polynomial in r = sqrt(x**2 + y**2):
        p = k1*r**2 + k2*r**4 + k3*r**6.

        Arguments:
        ---------
        dist_coeffs - length-3 array, holding k_i.
        """
        if dist_coeffs.shape != (3,):
            raise ValueError("Expected a 3-element array")

        self.added_par.k1 = dist_coeffs[0]
        self.added_par.k2 = dist_coeffs[1]
        self.added_par.k3 = dist_coeffs[2]

    def get_radial_distortion(self):
        """
        Return the radial distortion polynomial coefficients as a 3 element.

        array, from lowest power to highest.
        """
        ret = np.empty(3)
        ret[0] = self.added_par.k1
        ret[1] = self.added_par.k2
        ret[2] = self.added_par.k3
        return ret

    def set_decentering(self, decent: np.ndarray):
        """
        Set the parameters of decentering distortion (a.k.a. p1, p2, see [1]).

        Arguments:
        ---------
        decent - array, holding p_i
        """
        if decent.shape != (2,):
            raise ValueError("Expected a 2-element array")

        self.added_par.p1 = decent[0]
        self.added_par.p2 = decent[1]

    def get_decentering(self):
        """Return the decentering parameters [1] as a 2 element array, (p_1, p_2)."""
        ret = np.empty(2)
        ret[0] = self.added_par.p1
        ret[1] = self.added_par.p2
        return ret

    def set_affine_trans(self, affine):
        """
        Set the affine transform parameters (x-scale, shear) of the image.

        Arguments:
        ---------
        affine - array, holding (x-scale, shear) in order.
        """
        if affine.shape != (2,):
            raise ValueError("Expected a 2-element array")

        self.added_par.scx = affine[0]
        self.added_par.she = affine[1]

    def get_affine(self):
        """Return the affine transform parameters [1] as a 2 element array, (scx, she)."""
        ret = np.empty(2)
        ret[0] = self.added_par.scx
        ret[1] = self.added_par.she
        return ret

    def set_glass_vec(self, gvec: np.ndarray):
        """
        Set the glass vector: a vector from the origin to the glass, directed.

        normal to the glass.

        Arguments:
        ---------
        gvec - a 3-element array, the glass vector.
        """
        if len(gvec) != 3:
            raise ValueError("Expected a 3-element array")

        self.glass_par.vec_x = gvec[0]
        self.glass_par.vec_y = gvec[1]
        self.glass_par.vec_z = gvec[2]

    def get_glass_vec(self):
        """Return the glass vector, a 3-element array."""
        ret = np.empty(3)
        ret[0] = self.glass_par.vec_x
        ret[1] = self.glass_par.vec_y
        ret[2] = self.glass_par.vec_z
        return ret


def write_ori(
    Ex: Exterior,
    In: Interior,
    G: Glass,
    ap: Optional[ap_52],
    filename: str,
    add_file: Optional[str],
) -> bool:
    """Write an orientation file."""
    success = False

    with open(filename, "w", encoding="utf-8") as fp:
        fp.write(f"{Ex.x0:.8f} {Ex.y0:.8f} {Ex.z0:.8f}\n")
        fp.write(f"    {Ex.omega:.8f} {Ex.phi:.8f} {Ex.kappa:.8f}\n\n")
        for row in Ex.dm:
            fp.write(f"    {row[0]:.7f} {row[1]:.7f} {row[2]:.7f}\n")
        fp.write(f"\n    {In.xh:.4f} {In.yh:.4f}\n    {In.cc:.4f}\n")
        fp.write(f"\n    {G.vec_x:.15f} {G.vec_y:.15f} {G.vec_z:.15f}\n")

    if add_file is None:
        return success

    with open(add_file, "w", encoding="utf-8") as fp:
        fp.write(
            f"{ap.k1:.8f} {ap.k2:.8f} {ap.k3:.8f} {ap.p1:.8f} {ap.p2:.8f} {ap.scx:.8f} {ap.she:.8f}"
        )
        success = True

    return success


def read_ori(
    ori_file: str, add_file: str = None, add_fallback: str = None
) -> Calibration:
    """
    Read exterior and interior orientation.

    and - if available, parameters for distortion corrections.

    Arguments:
    ---------
    - Ex: output buffer for exterior orientation.
    - I: output buffer for interior orientation.
    - G: output buffer for glass parameters.
    - ori_file: path of file containing interior and exterior orientation data.
    - addp: output buffer for additional (distortion) parameters.
    - add_file: path of file containing added (distortions) parameters.
    - add_fallback: path to file for use if add_file can't be opened.

    Returns:
    -------
    - cal: Calibration object without multimedia lookup table.
    """
    Ex = Exterior()
    In = Interior()
    G = Glass()
    addp = ap_52()

    with open(ori_file, "r", encoding="utf-8") as fp:
        # Exterior
        scan_res = fp.read().split()
        Ex.x0, Ex.y0, Ex.z0 = map(float, scan_res[:3])
        Ex.omega, Ex.phi, Ex.kappa = map(float, scan_res[3:6])
        # Exterior rotation matrix
        for i in range(3):
            scan_res = fp.read().split()
            Ex.dm[i] = list(map(float, scan_res))
        # Interior
        scan_res = fp.read().split()
        In.xh, In.yh, In.cc = map(float, scan_res)
        # Glass
        scan_res = fp.read().split()
        G.vec_x, G.vec_y, G.vec_z = map(float, scan_res)

    # Additional parameters

    try:
        with open(add_file, "r", encoding="utf-8") as fp:
            scan_res = fp.read().split()
            addp.k1, addp.k2, addp.k3 = map(float, scan_res[:3])
            addp.p1, addp.p2 = map(float, scan_res[3:5])
            addp.scx, addp.she = map(float, scan_res[5:])
    except FileNotFoundError:
        if add_fallback:
            with open(add_fallback, "r", encoding="utf-8") as fp:
                scan_res = fp.read().split()
                addp.k1, addp.k2, addp.k3 = map(float, scan_res[:3])
                addp.p1, addp.p2 = map(float, scan_res[3:5])
                addp.scx, addp.she = map(float, scan_res[5:])
        else:
            print("no addpar fallback used")  # Waits for proper logging.
            addp.k1 = addp.k2 = addp.k3 = addp.p1 = addp.p2 = addp.she = 0.0
            addp.scx = 1.0

    cal = Calibration(Ex, In, G, addp)

    return cal


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


def read_calibration(
    ori_file: pathlib.Path, addpar_file: str = None, fallback_file: str = None
) -> Calibration:
    """Read the orientation file including the added parameters."""
    ret = read_ori(ori_file, addpar_file, fallback_file)
    ret.ext_par = rotation_matrix(ret.ext_par)
    ret.mmlut.data = None  # no multimedia data yet
    return ret


def write_calibration(cal, ori_file, add_file):
    return write_ori(
        cal.ext_par, cal.int_par, cal.glass_par, cal.added_par, ori_file, add_file
    )


def rotation_matrix(Ex: Exterior) -> Exterior:
    """Calculate the necessary trigonometric functions to rotate the Dmatrix of Exterior Ex."""
    cp = np.cos(Ex.phi)
    sp = np.sin(Ex.phi)
    co = np.cos(Ex.omega)
    so = np.sin(Ex.omega)
    ck = np.cos(Ex.kappa)
    sk = np.sin(Ex.kappa)

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

    Ex.dm = np.round(Ex.dm, 6)
    return Ex
