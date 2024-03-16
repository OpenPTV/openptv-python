"""Calibration data structures and functions."""

import copy
from pathlib import Path
from typing import Optional

import numpy as np
from numba import njit


@njit
def rotation_matrix(ext: np.ndarray) -> None:
    """Calculate the necessary trigonometric functions to rotate the Dmatrix of Exterior ext_par.

    Rotation is performed by multiplication of three rotation matrices,
    rotation around X axis is performed first, then around Y axis, and finally around Z axis.
    rotation around X axis does not change values along X axis, i.e. X_omega = X

    Maas, H.G., Gruen, A. & Papantoniou, D. Particle tracking velocimetry in
    three-dimensional flows. Experiments in Fluids 15, 133–146 (1993).
    https://doi.org/10.1007/BF00190953

    """
    omega, phi, kappa = ext['omega'], ext['phi'], ext['kappa']


    co = np.cos(omega)
    so = np.sin(omega)

    cp = np.cos(phi)
    sp = np.sin(phi)

    ck = np.cos(kappa)
    sk = np.sin(kappa)

    # dm = np.zeros((3, 3), dtype=np.float64)
    dm = ext['dm'] # shortcut to the dm field of the first element of the array

    dm[0, 0] = cp * ck
    dm[0, 1] = -cp * sk
    dm[0, 2] = sp
    dm[1, 0] = co * sk + so * sp * ck
    dm[1, 1] = co * ck - so * sp * sk
    dm[1, 2] = -so * cp
    dm[2, 0] = so * sk - co * sp * ck
    dm[2, 1] = so * ck + co * sp * sk
    dm[2, 2] = co * cp

    # print(dm)

    return None

exterior_dtype = np.dtype([
    ('x0', np.float64),
    ('y0', np.float64),
    ('z0', np.float64),
    ('omega', np.float64),
    ('phi', np.float64),
    ('kappa', np.float64),
    ('dm', np.float64, (3, 3))
    ])
Exterior = np.array((0, 0, 0, 0, 0, 0, np.eye(3)), dtype = exterior_dtype)
rotation_matrix(Exterior)             # rotation should be a unit matrix
assert np.allclose(np.eye(3), Exterior['dm'])

interior_dtype = np.dtype([
    ('xh', np.float64),
    ('yh', np.float64),
    ('cc', np.float64)
    ])
# Interior = np.zeros(1, dtype = interior_dtype)
Interior = np.array((0.,0.,0.), dtype = interior_dtype)


# ap52_dtype = np.dtype([
#     ('k1', np.float64),
#     ('k2', np.float64),
#     ('k3', np.float64),
#     ('p1', np.float64),
#     ('p2', np.float64),
#     ('scx', np.float64),
#     ('she', np.float64)
#     ])
ap_52 = np.array((0, 0, 0, 0, 0, 1, 0), dtype = np.float64)
# ap_52 = np.array((0, 0, 0, 0, 0, 1, 0), dtype = ap52_dtype)

mmlut_dtype = np.dtype([
    ('origin', np.float64, 3),
    ('nr', np.int32),
    ('nz', np.int32),
    ('rw', np.int32),
    ])

mm_lut = np.array(([0.,0.,0.], 0, 0, 0), dtype = mmlut_dtype)
mm_lut_data = np.empty((mm_lut['nr'], mm_lut['nz']), dtype=np.float64)


class Calibration:
    """Calibration data structure."""

    def __init__(self,
                 ext_par=None,
                 int_par=None,
                 glass_par=None,
                 added_par=None,
                 mmlut=None,
                 mmlut_data=None):
        if ext_par is None:
            ext_par = Exterior.copy()
        if int_par is None:
            int_par = Interior.copy()
        if glass_par is None:
            glass_par = np.array([0.0, 0.0, 1.0])
        if added_par is None:
            added_par = np.array((0, 0, 0, 0, 0, 1, 0), dtype = np.float64)
        if mmlut is None:
            mmlut = copy.deepcopy(mm_lut) # (np.zeros(3), 0, 0, 0)
        if mmlut_data is None:
            mmlut_data = np.zeros((mmlut['nr'], mmlut['nz']), dtype=np.float64)


        self.ext_par = ext_par
        self.int_par = int_par
        self.glass_par = glass_par
        self.added_par = added_par
        self.mmlut = mmlut
        self.mmlut_data = mmlut_data


    @classmethod
    def from_file(cls, ori_file: Path, add_file: Path | None):
        """
        Read exterior and interior orientation, and if available, parameters for distortion corrections.

        Arguments:
        ---------
        - ori_file: path of file containing interior and exterior orientation data.
        - add_file: path of file containing added (distortions) parameters.
        - add_fallback: path to file for use if add_file can't be opened.

        Returns
        -------
        - ext_par, int_par, glass, addp: Calibration object parts without multimedia lookup table.
        """
        if not ori_file.exists():
            raise IOError(f"File {ori_file} does not exist")

        ret = cls()

        with open(ori_file, "r", encoding="utf-8") as fp:
            tmp  = fp.read()

        data = np.fromstring(tmp, dtype=np.float64, sep=" ")
        # print(data)

        ret.set_pos(data[:3])
        ret.set_angles(data[3:6])
        # no need. set_angles updates rotation matrix automatically
        # ret.set_rotation_matrix(data[6:15].reshape(3, 3))
        ret.set_primary_point(data[15:18]) # xh, yh, cc
        ret.glass_par = data[18:] # glass position vector from the water side


        # double-check that we have the correct rotation matrix
        # self.ext_par.rotation_matrix()

        # this is anyhow default
        # self.mmlut.data = None  # no multimedia data yet

        ret.added_par = np.array([0,0,0,0,0,1,0], dtype=np.float64)

        # Additional parameters
        if add_file is not None:
            with open(add_file, "r", encoding="utf-8") as fp:
                addtmp = list(map(float, fp.readline().split()))

                ret.set_radial_distortion(np.array(addtmp[:3]))
                ret.set_decentering(np.array(addtmp[3:5]))
                ret.set_affine_distortion(np.array(addtmp[5:]))

        # except FileNotFoundError:
        else:
            # print("no addpar fallback used")  # Waits for proper logging.
            print("No addpar file found. Using default values for radial distortion")

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


    def increment_attribute(self, attr_name, increment_value):
        """Update the value of an attribute by increment_value."""
        if hasattr(self.ext_par, attr_name):
            setattr(self.ext_par, attr_name, getattr(
                self.ext_par, attr_name) + increment_value)
        if hasattr(self.int_par, attr_name):
            setattr(self.int_par, attr_name, getattr(
                self.int_par, attr_name) + increment_value)

    def update_rotation_matrix(self) -> None:
        """Update the rotation matrix of the exterior orientation."""
        rotation_matrix(self.ext_par)

    def set_rotation_matrix(self, dm: np.ndarray) -> None:
        """Set exterior rotation matrix."""
        if dm.shape != (3, 3):
            raise ValueError("Illegal argument for exterior rotation matrix")
        self.ext_par[0]['dm'] = dm

    def set_pos(self, pos: np.ndarray) -> None:
        """
        Set exterior position.

        Parameter: x_y_z_np - numpy array of 3 elements for x, y, z.
        """
        pos = np.array(pos, dtype = np.float64)

        if pos.shape != (3,):
            raise ValueError(
                "Illegal array argument "
                + str(pos)
                + " for x, y, z. Expected array/list of 3 numbers"
            )
        self.ext_par['x0'], self.ext_par['y0'], self.ext_par['z0'] = pos

    def get_pos(self) -> np.ndarray:
        """Return array of 3 elements representing exterior's x, y, z."""
        return np.r_[self.ext_par['x0'], self.ext_par['y0'], self.ext_par['z0']]

    def set_angles(self, o_p_k_np: np.ndarray) -> None:
        """
        Set angles (omega, phi, kappa) and recalculates Dmatrix accordingly.

        Parameter: o_p_k_np - array of 3 elements.
        """
        o_p_k_np = np.array(o_p_k_np, dtype=np.float64)

        if o_p_k_np.shape != (3,):
            raise ValueError(
                f"Illegal array argument {o_p_k_np} for "
                "omega, phi, kappa. Expected array or list of 3 float"
            )
        self.ext_par['omega'], self.ext_par['phi'], self.ext_par['kappa'] = o_p_k_np
        self.update_rotation_matrix()

    def get_angles(self) -> np.ndarray:
        """Return an array of 3 elements representing omega, phi, kappa."""
        return np.r_[self.ext_par['omega'], self.ext_par['phi'], self.ext_par['kappa']]

    def get_rotation_matrix(self) -> np.ndarray:
        """Return a 3x3 numpy array that represents Exterior's rotation matrix."""
        return self.ext_par['dm'].copy()

    def set_primary_point(self, prim_point_pos: np.ndarray) -> None:
        """
        Set the camera's primary point position (a.k.a. interior orientation).

        Arguments:
        ---------
        prim_point_pos - a 3 element array holding the values of x and y shift
            of point from sensor middle and sensor-point distance, int_par this
            order.
        """
        if prim_point_pos.shape != (3,):
            raise ValueError("Expected a 3-element array")

        self.int_par['xh'], self.int_par['yh'], self.int_par['cc'] = prim_point_pos
        # self.int_par.set_primary_point(prim_point_pos)

    def get_primary_point(self):
        """
        Return the primary point position (a.k.a. interior orientation) as a 3.

        element array holding the values of x and y shift of point from sensor
        middle and sensor-point distance, int_par this order.
        """
        return np.r_[self.int_par['xh'], self.int_par['yh'], self.int_par['cc']]

    def set_radial_distortion(self, dist_coeffs: np.ndarray) -> None:
        """
        Set the parameters for the image radial distortion, where the x/y.

        coordinates are corrected by a polynomial int_par r = sqrt(x**2 + y**2):
        p = k1*r**2 + k2*r**4 + k3*r**6.

        Arguments:
        ---------
        dist_coeffs - length-3 array, holding k_i.
        """
        if dist_coeffs.shape != (3,):
            raise ValueError("Expected a 3-element array")

        self.added_par[:3] = dist_coeffs

        # self.added_par.k1, self.added_par.k2, self.added_par.k3 = dist_coeffs


    def get_radial_distortion(self):
        """
        Return the radial distortion polynomial coefficients as a 3 element.

        array, from lowest power to highest.
        """
        # return np.r_[self.added_par.k1, self.added_par.k2, self.added_par.k3]
        return self.added_par[:3]

    def set_decentering(self, decent: np.ndarray) -> None:
        """
        Set the parameters of decentering distortion (a.k.a. p1, p2, see [1]).

        Arguments:
        ---------
        decent - array, holding p_i
        """
        if decent.shape != (2,):
            raise ValueError("Expected a 2-element list")

        # self.added_par.p1, self.added_par.p2 = decent
        self.added_par[3:5] = decent

    def get_decentering(self):
        """Return the decentering parameters [1] as a 2 element array, (p_1, p_2)."""
        # return np.r_[self.added_par.p1, self.added_par.p2]
        return self.added_par[3:5]

    def set_affine_distortion(self, affine: np.ndarray) -> None:
        """
        Set the affine transform parameters (x-scale, shear) of the image.

        Arguments:
        ---------
        affine - array, holding (x-scale, shear) int_par order.
        """
        if affine.shape != (2,):
            raise ValueError("Expected a 2-element list")

        # self.added_par.scx, self.added_par.she = affine
        self.added_par[5:] = affine

    def get_affine(self):
        """Return the affine transform parameters [1] as a 2 element array, (scx, she)."""
        # return np.r_[self.added_par.scx, self.added_par.she]
        return self.added_par[5:]

    def set_glass_vec(self, gvec: np.ndarray):
        """
        Set the glass vector: a vector from the origin to the glass, directed.

        normal to the glass.

        Arguments:
        ---------
        gvec - a 3-element array, the glass vector.
        """
        gvec = np.array(gvec, dtype=np.float64)

        if gvec.shape != (3,):
            raise ValueError("Expected a 3-element list or array")

        self.glass_par = gvec

    def get_glass_vec(self) -> np.ndarray:
        """Return the glass vector, a 3-element array of float."""
        return self.glass_par

    def set_added_par(self, ap52_array: np.ndarray):
        """Set added par from an numpy array of parameters."""
        # self.added_par = np.array(tuple(ap52_array.tolist()), dtype = ap52_dtype)
        self.added_par = ap52_array.astype(np.float64)

    def copy(self, new_copy):
        """Copy the calibration data to a new object."""
        new_copy = Calibration()
        new_copy.ext_par = self.ext_par
        new_copy.int_par = self.int_par
        new_copy.glass_par = self.glass_par
        new_copy.added_par = self.added_par
        new_copy.mmlut = self.mmlut


def write_ori(
    ext_par: np.ndarray,
    int_par: np.ndarray,
    glass: np.ndarray,
    added_par: np.ndarray,
    filename: str,
    add_file: Optional[str],
) -> bool:
    """Write an orientation file."""
    success = False

    with open(filename, "w", encoding="utf-8") as fp:
        fp.write(f"{ext_par['x0']:.8f} {ext_par['y0']:.8f} {ext_par['z0']:.8f}\n")
        fp.write(f"{ext_par['omega']:.8f} {ext_par['phi']:.8f} {ext_par['kappa']:.8f}\n\n")
        for row in ext_par['dm']:
            fp.write(f"{row[0]:.7f} {row[1]:.7f} {row[2]:.7f}\n")
        fp.write(f"\n{int_par['xh']:.4f} {int_par['yh']:.4f}\n{int_par['cc']:.4f}\n")
        fp.write(
            f"\n{glass[0]:.15f} {glass[1]:.15f} {glass[2]:.15f}\n")

    if add_file is None:
        return success

    with open(add_file, "w", encoding="utf-8") as fp:
        fp.write(
            f"{added_par[0]:.8f} {added_par[1]:.8f} {added_par[2]:.8f} "
            f"{added_par[3]:.8f} {added_par[4]:.8f} {added_par[5]:.8f} {added_par[6]:.8f}\n"
        )
        success = True

    return success


def read_ori(ori_file: Path, add_file: Path) -> Calibration:
    """
    Read exterior and interior orientation, and if available, parameters for distortion corrections.

    Arguments:
    ---------
    - ori_file: path of file containing interior and exterior orientation data.
    - add_file: path of file containing added (distortions) parameters.
    - add_fallback: path to file for use if add_file can't be opened.

    Returns
    -------
    - ext_par, int_par, glass, addp: Calibration object parts without multimedia lookup table.
    """
    ret = Calibration()
    ret.from_file(ori_file, add_file)

    return ret


def compare_exterior(e1: np.ndarray, e2: np.ndarray) -> bool:
    """Compare exterior orientation parameters."""
    return (
        np.allclose(e1['dm'], e2['dm'], atol=1e-6)
        and (e1['x0'] == e2['x0'])
        and (e1['y0'] == e2['y0'])
        and (e1['z0'] == e2['z0'])
        and (e1['omega'] == e2['omega'])
        and (e1['phi'] == e2['phi'])
        and (e1['kappa'] == e2['kappa'])
    )


def compare_interior(i1: np.ndarray, i2: np.ndarray) -> bool:
    """Compare interior orientation parameters."""
    return i1['xh'] == i2['xh'] and i1['yh'] == i2['yh'] and i1['cc'] == i2['cc']


def compare_glass(g1: np.ndarray, g2: np.ndarray) -> bool:
    """Compare `Glass` parameters.

    objects that need to be compared. The function then returns `1` if all
    `vec_x`, `vec_y` and `vec_z` values of `g1` are equal to the corresponding
    values int_par `g2`. Else, the function returns `0`.

    Args:
    ----
        g1 (float array): vector pointing from the 3D origin to the surface of the glass
        g2 (float array): another vector for comparison

    Returns
    -------
        bool: True if vectors are identical, False otherwise
    """
    return np.array_equal(g1, g2)


def compare_calibration(c1: Calibration, c2: Calibration) -> bool:
    """Compare calibration parameters."""
    return (
        compare_exterior(c1.ext_par, c2.ext_par)
        and compare_interior(c1.int_par, c2.int_par)
        and compare_glass(c1.glass_par, c2.glass_par)
    )


def compare_addpar(a1, a2):
    """Compare added parameters."""
    return np.array_equal(a1, a2)


def read_calibration(ori_file: Path, addpar_file: Path | None) -> Calibration:
    """Read the orientation file including the added parameters."""
    return Calibration().from_file(ori_file, addpar_file)


def write_calibration(cal, ori_file, add_file):
    """Write the orientation file including the added parameters."""
    return write_ori(
        cal.ext_par, cal.int_par, cal.glass_par, cal.added_par, ori_file, add_file
    )
