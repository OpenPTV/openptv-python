"""Calibration data structures and functions."""

import pathlib
from typing import List, Optional

import numpy as np
from numba import njit


@njit
def rotation_matrix(phi: float, omega: float, kappa: float) -> np.ndarray:
    """Calculate the necessary trigonometric functions to rotate the Dmatrix of Exterior Ex."""
    cp = np.cos(phi)
    sp = np.sin(phi)
    co = np.cos(omega)
    so = np.sin(omega)
    ck = np.cos(kappa)
    sk = np.sin(kappa)

    dm = np.zeros((3, 3), dtype=np.float64)
    dm[0, 0] = cp * ck
    dm[0, 1] = -cp * sk
    dm[0, 2] = sp
    dm[1, 0] = co * sk + so * sp * ck
    dm[1, 1] = co * ck - so * sp * sk
    dm[1, 2] = -so * cp
    dm[2, 0] = so * sk - co * sp * ck
    dm[2, 1] = so * ck + co * sp * sk
    dm[2, 2] = co * cp
    return dm

class Exterior:
    """Exterior orientation data structure."""

    def __init__(self, x0=0.0, y0=0.0, z0=0.0, omega=0.0, phi=0.0, kappa=0.0, dm=None):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.omega = omega
        self.phi = phi
        self.kappa = kappa
        self.dm = dm if dm is not None else np.identity(3, dtype=np.float64)


    def update_rotation_matrix(self) -> None:
        """Create rotation matrix using three angles of the camera."""
        self.dm = rotation_matrix(self.phi, self.omega, self.kappa)

    def set_rotation_matrix(self, dm: np.ndarray) -> None:
        """Set the rotation matrix of the camera."""
        self.dm = dm

    def set_pos(self, pos: List[float]) -> None:
        """Set the position of the camera."""
        self.x0, self.y0, self.z0 = pos

    def set_angles(self, angles: List[float]) -> None:
        """Set the angles of the camera."""
        self.omega, self.phi, self.kappa = angles

        # adjust rotation matrix
        self.update_rotation_matrix()

    def increment_attribute(self, attr_name, increment_value):
        """Update the value of an attribute by increment_value."""
        if hasattr(self, attr_name):
            setattr(self, attr_name, getattr(self, attr_name) + increment_value)

    def __repr__(self) -> str:
        """Return a string representation of the Exterior object."""
        output = f"Exterior: x0={self.x0}, y0={self.y0}, z0={self.z0}\n"
        output += f"omega={self.omega}, phi={self.phi}, kappa={self.kappa}\n"
        return output


class Interior:
    """Interior orientation data structure."""

    def __init__(self, xh=0.0, yh=0.0, cc=0.0):
        self.xh = xh
        self.yh = yh
        self.cc = cc

    def set_primary_point(self, point: np.ndarray) -> None:
        """Set the primary point of the camera."""
        self.xh, self.yh, self.cc = point

    def set_back_focal_distance(self, cc: float) -> None:
        """Set the back focal distance of the camera."""
        self.cc = cc


class Glass:
    """Glass data structure."""

    def __init__(self, vec_x=0.0, vec_y=0.0, vec_z=1.0):
        self.vec_x = vec_x
        self.vec_y = vec_y
        self.vec_z = vec_z

    def set_glass_vec(self, vec: np.ndarray) -> None:
        """Set the glass vector."""
        self.vec_x, self.vec_y, self.vec_z = vec


class ap_52:
    """Additional parameters for distortion correction."""

    def __init__(self, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0, scx=1.0, she=0.0):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2
        self.scx = scx
        self.she = she

    def set_radial_distortion(self, dist_array: np.ndarray) -> None:
        """Set the radial distortion parameters k1, k2, k3."""
        self.k1, self.k2, self.k3 = dist_array

    def set_decentering(self, decent: np.ndarray) -> None:
        """Set the decentring parameters p1 and p2."""
        self.p1, self.p2 = decent

    def set_affine_distortion(self, affine: np.ndarray) -> None:
        """Set the affine distortion parameters scx and she."""
        self.scx, self.she = affine


class mm_lut:
    """Multimedia lookup table data structure."""

    def __init__(self, origin=None, nr=3, nz=3, rw=0, data=None):
        if origin is None:
            origin = np.zeros(3, dtype=np.float32)
        # if data is None:
        #     data = np.zeros((nr, nz), dtype=np.float32)  # Assuming data is a 2D array, adjust as needed
        self.origin = origin
        self.nr = nr
        self.nz = nz
        self.rw = rw
        self.data = data


class Calibration:
    """Calibration data structure."""

    def __init__(self, ext_par=None, int_par=None, glass_par=None, added_par=None, mmlut=None):
        if ext_par is None:
            ext_par = Exterior()
        if int_par is None:
            int_par = Interior()
        if glass_par is None:
            glass_par = Glass()
        if added_par is None:
            added_par = ap_52()
        if mmlut is None:
            mmlut = mm_lut(np.zeros(3), 0, 0, 0, None)

        self.ext_par = ext_par
        self.int_par = int_par
        self.glass_par = glass_par
        self.added_par = added_par
        self.mmlut = mmlut

    @classmethod
    def from_file(cls, ori_file: str, add_file: str):
        """
        Read exterior and interior orientation, and if available, parameters for distortion corrections.

        Arguments:
        ---------
        - ori_file: path of file containing interior and exterior orientation data.
        - add_file: path of file containing added (distortions) parameters.
        - add_fallback: path to file for use if add_file can't be opened.

        Returns
        -------
        - Ex, In, G, addp: Calibration object parts without multimedia lookup table.
        """
        if not pathlib.Path(ori_file).exists():
            raise IOError(f"File {ori_file} does not exist")

        ret = cls()

        with open(ori_file, "r", encoding="utf-8") as fp:
            # Exterior
            ret.set_pos([float(x) for x in fp.readline().split()])
            ret.set_angles([float(x) for x in fp.readline().split()])

            # ret.ext_par.set_pos(np.fromstring(fp.readline(), dtype=float, sep="\t"))
            # ret.ext_par.set_angles(np.fromstring(fp.readline(), dtype=float, sep="\t"))

            # Exterior rotation matrix
            # skip line
            fp.readline()

            # read 3 lines and set a rotation matrix
            _ = [[float(x) for x in fp.readline().split()] for _ in range(3)]

            # I skip reading rotation matrix as it's set up by set_angles.
            # self.set_rotation_matrix(np.array(float_list).reshape(3, 3))

            # Interior
            # skip
            fp.readline()

            tmp = [float(x) for x in fp.readline().split()]  # xh,yh
            tmp += [float(x) for x in fp.readline().split()]  # cc
            ret.int_par.set_primary_point(np.array(tmp))
            # self.int_par.set_back_focal_distance(float(fp.readline()))

            # Glass
            # skip
            fp.readline()
            ret.glass_par.set_glass_vec(np.array([float(x) for x in fp.readline().split()]))

        # double-check that we have the correct rotation matrix
        # self.ext_par.rotation_matrix()

        # this is anyhow default
        # self.mmlut.data = None  # no multimedia data yet

        # Additional parameters
        try:
            with open(add_file, "r", encoding="utf-8") as fp:
                tmp = list(map(float, fp.readline().split()))

                ret.added_par.set_radial_distortion(np.array(tmp[:3]))
                ret.added_par.set_decentering(np.array(tmp[3:5]))
                ret.added_par.set_affine_distortion(np.array(tmp[5:]))

        except FileNotFoundError:
            print("no addpar fallback used")  # Waits for proper logging.
            ret.added_par.k1 = ret.added_par.k2 = ret.added_par.k3 \
            = ret.added_par.p1 = ret.added_par.p2 = ret.added_par.she = 0.0
            ret.added_par.scx = 1.0


        return ret
        # print(f"Calibration data read from files {ori_file} and {add_file}")

    def write(self, ori_file: str, addpar_file: str):
        """Write calibration to file."""
        success = write_ori(
            self.ext_par,
            self.int_par,
            self.glass_par,
            self.added_par,
            ori_file,
            addpar_file,
        )
        if not success:
            raise IOError("Failed to write ori file")

    def set_rotation_matrix(self, dm: np.ndarray) -> None:
        """Set exterior rotation matrix."""
        if dm.shape != (3, 3):
            raise ValueError("Illegal argument for exterior rotation matrix")
        self.ext_par.set_rotation_matrix(dm)

    def set_pos(self, x_y_z_np: List[float]) -> None:
        """
        Set exterior position.

        Parameter: x_y_z_np - numpy array of 3 elements for x, y, z.
        """
        if len(x_y_z_np) != 3:
            raise ValueError(
                "Illegal array argument "
                + str(x_y_z_np)
                + " for x, y, z. Expected array/list of 3 numbers"
            )
        self.ext_par.set_pos(x_y_z_np)

    def get_pos(self):
        """Return array of 3 elements representing exterior's x, y, z."""
        return np.r_[self.ext_par.x0, self.ext_par.y0, self.ext_par.z0]

    def set_angles(self, o_p_k_np: List[float]) -> None:
        """
        Set angles (omega, phi, kappa) and recalculates Dmatrix accordingly.

        Parameter: o_p_k_np - array of 3 elements.
        """
        if len(o_p_k_np) != 3:
            raise ValueError(
                f"Illegal array argument {o_p_k_np} for "
                "omega, phi, kappa. Expected array/list of 3 numbers"
            )
        self.ext_par.set_angles(o_p_k_np)

    def get_angles(self):
        """Return an array of 3 elements representing omega, phi, kappa."""
        return np.r_[self.ext_par.omega, self.ext_par.phi, self.ext_par.kappa]

    def get_rotation_matrix(self):
        """Return a 3x3 numpy array that represents Exterior's rotation matrix."""
        return self.ext_par.dm

    def set_primary_point(self, prim_point_pos: List[float]) -> None:
        """
        Set the camera's primary point position (a.k.a. interior orientation).

        Arguments:
        ---------
        prim_point_pos - a 3 element array holding the values of x and y shift
            of point from sensor middle and sensor-point distance, in this
            order.
        """
        if len(prim_point_pos) != 3:
            raise ValueError("Expected a 3-element list")

        self.int_par.set_primary_point(prim_point_pos)

    def get_primary_point(self):
        """
        Return the primary point position (a.k.a. interior orientation) as a 3.

        element array holding the values of x and y shift of point from sensor
        middle and sensor-point distance, in this order.
        """
        return np.r_[self.int_par.xh, self.int_par.yh, self.int_par.cc]

    def set_radial_distortion(self, dist_coeffs: np.ndarray) -> None:
        """
        Set the parameters for the image radial distortion, where the x/y.

        coordinates are corrected by a polynomial in r = sqrt(x**2 + y**2):
        p = k1*r**2 + k2*r**4 + k3*r**6.

        Arguments:
        ---------
        dist_coeffs - length-3 array, holding k_i.
        """
        if len(dist_coeffs) != 3:
            raise ValueError("Expected a 3-element array")

        self.added_par.set_radial_distortion(dist_coeffs)

    def get_radial_distortion(self):
        """
        Return the radial distortion polynomial coefficients as a 3 element.

        array, from lowest power to highest.
        """
        return np.r_[self.added_par.k1, self.added_par.k2, self.added_par.k3]

    def set_decentering(self, decent: List[float]) -> None:
        """
        Set the parameters of decentering distortion (a.k.a. p1, p2, see [1]).

        Arguments:
        ---------
        decent - array, holding p_i
        """
        if len(decent) != 2:
            raise ValueError("Expected a 2-element list")

        self.added_par.set_decentering(decent)

    def get_decentering(self):
        """Return the decentering parameters [1] as a 2 element array, (p_1, p_2)."""
        ret = np.empty(2)
        ret[0] = self.added_par.p1
        ret[1] = self.added_par.p2
        return ret

    def set_affine_trans(self, affine: List[float]) -> None:
        """
        Set the affine transform parameters (x-scale, shear) of the image.

        Arguments:
        ---------
        affine - array, holding (x-scale, shear) in order.
        """
        if len(affine) != 2:
            raise ValueError("Expected a 2-element list")
        self.added_par.set_affine_distortion(affine)

    def get_affine(self):
        """Return the affine transform parameters [1] as a 2 element array, (scx, she)."""
        return np.r_[self.added_par.scx, self.added_par.she]

    def set_glass_vec(self, gvec: np.ndarray):
        """
        Set the glass vector: a vector from the origin to the glass, directed.

        normal to the glass.

        Arguments:
        ---------
        gvec - a 3-element array, the glass vector.
        """
        if len(gvec) != 3:
            raise ValueError("Expected a 3-element list")

        self.glass_par.set_glass_vec(gvec)

    def get_glass_vec(self):
        """Return the glass vector, a 3-element array."""
        return [self.glass_par.vec_x, self.glass_par.vec_y, self.glass_par.vec_z]

    def set_added_par(self, listpar: np.ndarray | list):
        """Set added par from an numpy array of parameters."""
        if isinstance(listpar, list):
            listpar = np.array(listpar)

        self.added_par = ap_52(*listpar.tolist())

    def copy(self, new_copy):
        """Copy the calibration data to a new object."""
        new_copy = Calibration()
        new_copy.ext_par = self.ext_par
        new_copy.int_par = self.int_par
        new_copy.glass_par = self.glass_par
        new_copy.added_par = self.added_par
        new_copy.mmlut = self.mmlut


def write_ori(
    Ex: Exterior,
    In: Interior,
    G: Glass,
    added_par: ap_52,
    filename: str,
    add_file: Optional[str],
) -> bool:
    """Write an orientation file."""
    success = False

    with open(filename, "w", encoding="utf-8") as fp:
        fp.write(f"{Ex.x0:.8f} {Ex.y0:.8f} {Ex.z0:.8f}\n")
        fp.write(f"{Ex.omega:.8f} {Ex.phi:.8f} {Ex.kappa:.8f}\n\n")
        for row in Ex.dm:
            fp.write(f"{row[0]:.7f} {row[1]:.7f} {row[2]:.7f}\n")
        fp.write(f"\n{In.xh:.4f} {In.yh:.4f}\n{In.cc:.4f}\n")
        fp.write(f"\n{G.vec_x:.15f} {G.vec_y:.15f} {G.vec_z:.15f}\n")

    if add_file is None:
        return success

    with open(add_file, "w", encoding="utf-8") as fp:
        fp.write(
            f"{added_par.k1:.8f} {added_par.k2:.8f} {added_par.k3:.8f} "
            f"{added_par.p1:.8f} {added_par.p2:.8f} {added_par.scx:.8f} {added_par.she:.8f}\n"
        )
        success = True

    return success


def read_ori(ori_file: str, add_file: str) -> Calibration:
    """
    Read exterior and interior orientation, and if available, parameters for distortion corrections.

    Arguments:
    ---------
    - ori_file: path of file containing interior and exterior orientation data.
    - add_file: path of file containing added (distortions) parameters.
    - add_fallback: path to file for use if add_file can't be opened.

    Returns
    -------
    - Ex, In, G, addp: Calibration object parts without multimedia lookup table.
    """
    ret = Calibration()
    ret.from_file(ori_file, add_file)

    return ret


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
    """Compare interior orientation parameters."""
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

    Returns
    -------
        _type_: _description_
    """
    return g1.vec_x == g2.vec_x and g1.vec_y == g2.vec_y and g1.vec_z == g2.vec_z


def compare_calibration(c1: Calibration, c2: Calibration) -> bool:
    """Compare calibration parameters."""
    return (
        compare_exterior(c1.ext_par, c2.ext_par)
        and compare_interior(c1.int_par, c2.int_par)
        and compare_glass(c1.glass_par, c2.glass_par)
    )


def compare_addpar(a1, a2):
    """Compare added parameters."""
    return (
        (a1.k1 == a2.k1)
        and (a1.k2 == a2.k2)
        and (a1.k3 == a2.k3)
        and (a1.p1 == a2.p1)
        and (a1.p2 == a2.p2)
        and (a1.scx == a2.scx)
        and (a1.she == a2.she)
    )


def read_calibration(ori_file: str, addpar_file: str) -> Calibration:
    """Read the orientation file including the added parameters."""
    return Calibration().from_file(ori_file, addpar_file)


def write_calibration(cal, ori_file, add_file):
    """Write the orientation file including the added parameters."""
    return write_ori(
        cal.ext_par, cal.int_par, cal.glass_par, cal.added_par, ori_file, add_file
    )


# def rotation_matrix(Ex: Exterior) -> None:
#     """Calculate the necessary trigonometric functions to rotate the Dmatrix of Exterior Ex."""
#     cp = np.cos(Ex.phi)
#     sp = np.sin(Ex.phi)
#     co = np.cos(Ex.omega)
#     so = np.sin(Ex.omega)
#     ck = np.cos(Ex.kappa)
#     sk = np.sin(Ex.kappa)

#     # Modify the Exterior Ex with the new Dmatrix
#     Ex.dm[0][0] = cp * ck
#     Ex.dm[0][1] = -cp * sk
#     Ex.dm[0][2] = sp
#     Ex.dm[1][0] = co * sk + so * sp * ck
#     Ex.dm[1][1] = co * ck - so * sp * sk
#     Ex.dm[1][2] = -so * cp
#     Ex.dm[2][0] = so * sk - co * sp * ck
#     Ex.dm[2][1] = so * ck + co * sp * sk
#     Ex.dm[2][2] = co * cp

#     # Ex.dm = np.round(Ex.dm, 6)
