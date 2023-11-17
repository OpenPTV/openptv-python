"""Parameters for OpenPTV-Python."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from openptv_python.constants import TR_MAX_CAMS


@dataclass
class MultimediaPar:
    """Multimedia parameters."""

    nlay: int = 1
    n1: float = 1.0
    n2: List[float] = field(default_factory=lambda: [1.0])
    d: List[float] = field(default_factory=lambda: [0.0])
    n3: float = 1.0

    def __post_init__(self):
        if len(self.n2) != len(self.d):
            raise ValueError("n2 and d must have the same length")

    def get_nlay(self):
        """Return the number of layers."""
        return self.nlay

    def get_n1(self):
        """Return the refractive index of the first medium."""
        return self.n1

    def set_n1(self, n1: float):
        """Return the refractive index of the first medium."""
        self.n1 = n1

    def get_n3(self):
        """Return the refractive index of the last medium."""
        return self.n3

    def set_n3(self, n3: float):
        """Return the refractive index of the last medium."""
        self.n3 = n3

    def get_n2(self):
        """Return the refractive index of the second medium."""
        return self.n2

    def get_d(self):
        """Return the thickness of the second medium."""
        return self.d

    def set_layers(self, refr_index: list[float], thickness: list[float]):
        """Set the layers of the medium."""
        if len(refr_index) != len(thickness):
            raise ValueError("Lengths of refractive index and thickness must be equal.")
        else:
            self.n2 = refr_index
            self.d = thickness
            # self.nlay = len(refr_index)

    def __str__(self) -> str:
        return f"nlay = {self.nlay}, n1 = {self.n1}, n2 = {self.n2}, d = {self.d}, n3 = {self.n3}"


def compare_mm_np(mm_np1: MultimediaPar, mm_np2: MultimediaPar) -> bool:
    """Compare two MultimediaPar objects."""
    return (mm_np1.nlay, mm_np1.n1, mm_np1.n3, mm_np1.n2[0], mm_np1.d[0]) == (
        mm_np2.nlay,
        mm_np2.n1,
        mm_np2.n3,
        mm_np2.n2[0],
        mm_np2.d[0],
    )


@dataclass
class SequencePar:
    """Sequence parameters."""

    num_cams: int = 0
    img_base_name: List[str] = field(default_factory=list)  # empty list
    first: int = 0
    last: int = 0

    def __post_init__(self):
        if len(self.img_base_name) == 0:
            self.img_base_name = [""] * self.num_cams

    def set_img_base_name(self, icam: int, new_name: str):
        """Set the image base name for each camera."""
        if icam > self.num_cams:
            raise ValueError("Length of names must be equal to num_cams.")
        if self.img_base_name is None:
            self.img_base_name = [""] * self.num_cams

        self.img_base_name[icam] = new_name

    def get_img_base_name(self, icam):
        """Get the image base name for each camera."""
        if icam > self.num_cams:
            raise ValueError("Length of names must be equal to num_cams.")
        return self.img_base_name[icam]

    def set_first(self, newfirst: int):
        """Set the first frame number."""
        self.first = newfirst

    def get_first(self):
        """Get the first frame number."""
        return self.first

    def set_last(self, newlast: int):
        """Set the last frame number."""
        self.last = newlast

    def get_last(self):
        """Get the last frame number."""
        return self.last

    def from_file(self, filename: str):
        """Read sequence parameters from file."""
        if not Path(filename).exists():
            raise IOError("File {filename} does not exist.")

        with open(filename, "r", encoding="utf-8") as par_file:
            self.img_base_name = [
                par_file.readline().strip() for _ in range(self.num_cams)
            ]
            self.first = int(par_file.readline().strip())
            self.last = int(par_file.readline().strip())


def read_sequence_par(filename: str, num_cams: int) -> SequencePar:
    """Read sequence parameters from file and return SequencePar object."""
    ret = SequencePar(num_cams=num_cams)
    ret.from_file(filename)
    return ret


def compare_sequence_par(sp1: SequencePar, sp2: SequencePar) -> bool:
    """Compare two SequencePar objects."""
    return all(
        getattr(sp1, field) == getattr(sp2, field)
        for field in SequencePar.__annotations__
    )


@dataclass
class TrackPar:
    """Tracking parameters."""

    dvxmax: float = 0.0
    dvxmin: float = 0.0
    dvymax: float = 0.0
    dvymin: float = 0.0
    dvzmax: float = 0.0
    dvzmin: float = 0.0
    dangle: float = 0.0
    dacc: float = 0.0
    add: int = 0
    dsumg: float = 0.0
    dn: float = 0.0
    dnx: float = 0.0
    dny: float = 0.0

    def from_file(self, filename: str):
        """Read tracking parameters from file and return TrackPar object.

        Note that the structure has 13 attributes, from which we read only 9
        dsumg, dn, dnx, dny are set to 0 automatically and used later in
        track.py

        """
        try:
            with open(filename, "r", encoding="utf-8") as fpp:
                self.dvxmin = float(fpp.readline().rstrip())
                self.dvxmax = float(fpp.readline().rstrip())
                self.dvymin = float(fpp.readline().rstrip())
                self.dvymax = float(fpp.readline().rstrip())
                self.dvzmin = float(fpp.readline().rstrip())
                self.dvzmax = float(fpp.readline().rstrip())
                self.dangle = float(fpp.readline().rstrip())
                self.dacc = float(fpp.readline().rstrip())
                self.add = int(fpp.readline().rstrip())
        except IOError as exc:
            raise (f"Error reading tracking parameters from {filename}") from exc  # type: ignore

    def get_dvxmin(self):
        """Return the minimum velocity in x direction."""
        return self.dvxmin

    def get_dvxmax(self):
        """Return the maximum velocity in x direction."""
        return self.dvxmax

    def get_dvymin(self):
        """Return the minimum velocity in y direction."""
        return self.dvymin

    def get_dvymax(self):
        """Return the maximum velocity in y direction."""
        return self.dvymax

    def get_dvz_min(self):
        """Return the minimum velocity in z direction."""
        return self.dvzmin

    def get_dvz_max(self):
        """Return the minimum velocity in z direction."""
        return self.dvzmax

    def get_dangle(self):
        """Return the maximum angle."""
        return self.dangle

    def get_dacc(self):
        """Return the maximum acceleration."""
        return self.dacc

    def get_add(self):
        """Return the adding new particles parameter."""
        return self.add

    def get_dsumg(self):
        """Return the maximum sum of the gradient."""
        return self.dsumg

    def set_dsumg(self, dsumg):
        """Set the maximum sum of the gradient."""
        self.dsumg = dsumg

    def get_dn(self):
        """Return the maximum refractive index."""
        return self.dn

    def set_dn(self, dn):
        """Set the maximum refractive index."""
        self.dn = dn

    def get_dnx(self):
        """Return the maximum refractive index in x direction."""
        return self.dnx

    def set_dnx(self, dnx):
        """Set the maximum refractive index in x direction."""
        self.dnx = dnx

    def set_dny(self, dny):
        """Set the maximum refractive index in y direction."""
        self.dny = dny

    def get_dny(self):
        """Return the maximum refractive index in y direction."""
        return self.dny


def read_track_par(filename: str) -> TrackPar:
    """Read tracking parameters from file and return TrackPar object."""
    tpar = TrackPar()
    tpar.from_file(filename)
    return tpar


def compare_track_par(t1: TrackPar, t2: TrackPar) -> bool:
    """Compare two TrackPar objects."""
    return all(getattr(t1, field) == getattr(t2, field) for field in t1.__annotations__)


@dataclass
class VolumePar:
    """Volume parameters.

    /* Volume parameters */
    fpp = fopen(filename, "r");
    if(fscanf(fpp, &(ret->x_lay[0])) == 0) goto handle_error;
    if(fscanf(fpp, &(ret->z_min_lay[0])) == 0) goto handle_error;
    if(fscanf(fpp, &(ret->z_max_lay[0])) == 0) goto handle_error;
    if(fscanf(fpp, &(ret->x_lay[1])) == 0) goto handle_error;
    if(fscanf(fpp, &(ret->z_min_lay[1])) == 0) goto handle_error;
    if(fscanf(fpp, &(ret->z_max_lay[1])) == 0) goto handle_error;

    if(fscanf(fpp, &(ret->cnx)) == 0) goto handle_error;
    if(fscanf(fpp, &(ret->cny)) == 0) goto handle_error;
    if(fscanf(fpp, &(ret->cn)) == 0) goto handle_error;
    if(fscanf(fpp, &(ret->csumg)) == 0) goto handle_error;
    if(fscanf(fpp, &(ret->corrmin)) == 0) goto handle_error;
    if(fscanf(fpp, &(ret->eps0)) == 0) goto handle_error;
    /* End of volume parameters */


    """

    x_lay: List[float] = field(default_factory=list)
    z_min_lay: List[float] = field(default_factory=list)
    z_max_lay: List[float] = field(default_factory=list)
    # minimal criteria for number of pixels
    cn: float = field(default_factory=float)
    cnx: float = field(default_factory=float)  # same in x direction
    cny: float = field(default_factory=float)  # same in y direction
    csumg: float = field(default_factory=float)  # same in sum of grey values
    # minimal criteria for epipolar distance
    eps0: float = field(default_factory=float)
    corrmin: float = field(
        default_factory=float
    )  # minimal correlation value of all criteria

    def set_z_min_lay(self, z_min_lay: list[float]) -> None:
        """Set the minimum z coordinate of the layers."""
        self.z_min_lay = z_min_lay

    def set_z_max_lay(self, z_max_lay: list[float]) -> None:
        """Set the maximum z coordinate of the layers."""
        self.z_max_lay = z_max_lay

    def set_cn(self, cn: float) -> None:
        """Set the refractive index."""
        self.cn = cn

    def set_cnx(self, cnx: float) -> None:
        """Set the refractive index in x direction."""
        self.cnx = cnx

    def set_csumg(self, csumg: float) -> None:
        """Set the maximum sum of the gradient."""
        self.csumg = csumg

    def set_eps0(self, eps0: float) -> None:
        """Set the maximum sum of the gradient."""
        self.eps0 = eps0

    def set_corrmin(self, corrmin: float) -> None:
        self.corrmin = corrmin

    def from_file(self, filename: str):
        """Read volume parameters from file.

        Args:
        ----
            filename (str): filename

        """
        with open(filename, "r", encoding="utf-8") as f:
            self.x_lay.append(float(f.readline()))
            self.z_min_lay.append(float(f.readline()))
            self.z_max_lay.append(float(f.readline()))
            self.x_lay.append(float(f.readline()))
            self.z_min_lay.append(float(f.readline()))
            self.z_max_lay.append(float(f.readline()))
            self.cnx, self.cny, self.cn, self.csumg, self.corrmin, self.eps0 = [
                float(f.readline()) for _ in range(6)
            ]


def read_volume_par(filename: str) -> VolumePar:
    """Read volume parameters from file and returns volume_par object.

    Args:
    ----
        filename (str): filename

    Returns:
    -------
        VolumePar: volume of interest parameters
    """
    vpar = VolumePar()
    vpar.from_file(filename)
    return vpar


def compare_volume_par(v1: VolumePar, v2: VolumePar) -> bool:
    """Compare two VolumePar objects."""
    return all(
        getattr(v1, attr) == getattr(v2, attr) for attr in VolumePar.__annotations__
    )


@dataclass
class ControlPar:
    """Control parameters."""

    num_cams: int = field(default_factory=int)
    img_base_name: List[str] = field(default_factory=list)
    cal_img_base_name: List[str] = field(default_factory=list)
    hp_flag: int = field(default=1)
    allCam_flag: int = field(default=0)
    tiff_flag: int = field(default=1)
    imx: int = field(default_factory=int)
    imy: int = field(default_factory=int)
    pix_x: float = field(default_factory=float)
    pix_y: float = field(default_factory=float)
    chfield: int = field(default_factory=int)
    mm: MultimediaPar = field(default_factory=MultimediaPar)

    def set_image_size(self, imsize: Tuple[int, int]):
        """Set image size in pixels."""
        self.imx = imsize[0]
        self.imy = imsize[1]

    def get_image_size(self) -> Tuple[int, int]:
        """Set image size in pixels."""
        return (self.imx, self.imy)

    def set_pixel_size(self, pixsize: Tuple[float, float]):
        """Set pixel size in mm."""
        self.pix_x = pixsize[0]
        self.pix_y = pixsize[1]

    def get_pixel_size(self) -> Tuple[float, float]:
        """Set pixel size in mm."""
        return (self.pix_x, self.pix_y)

    def set_chfield(self, chfield: int):
        """Set chfield."""
        self.chfield = chfield

    def get_chfield(self):
        """Set chfield."""
        return self.chfield

    def get_multimedia_par(self):
        """Return multimedia parameters."""
        return self.mm

    def get_num_cams(self):
        """Return number of cameras."""
        return self.num_cams

    def set_hp_flag(self, hp_flag):
        """Return high pass flag."""
        self.hp_flag = hp_flag

    def get_hp_flag(self):
        """Return high pass flag."""
        return self.hp_flag

    def get_allCam_flag(self):
        """Return allCam flag."""
        return self.allCam_flag

    def get_tiff_flag(self):
        """Return tiff flag."""
        return self.tiff_flag

    def from_file(self, filename: str):
        """Read control parameters from file and return ControlPar object."""
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Could not open file {filename}")

        with open(filename, "r", encoding="utf-8") as par_file:
            self.num_cams = int(par_file.readline().strip())

            for _ in range(self.num_cams):
                self.img_base_name.append(par_file.readline().strip())
                self.cal_img_base_name.append(par_file.readline().strip())

            self.hp_flag = int(par_file.readline().strip())
            self.allCam_flag = int(par_file.readline().strip())
            self.tiff_flag = int(par_file.readline().strip())
            self.imx = int(par_file.readline().strip())
            self.imy = int(par_file.readline().strip())
            self.pix_x = float(par_file.readline().strip())
            self.pix_y = float(par_file.readline().strip())
            self.chfield = int(par_file.readline().strip())
            self.mm.n1 = float(par_file.readline().strip())
            self.mm.n2[0] = float(par_file.readline().strip())
            self.mm.n3 = float(par_file.readline().strip())
            self.mm.d[0] = float(par_file.readline().strip())

        # the original idea is to have more layers inside water with different
        # refractive indices and different thicknesses.
        # therefore the following lines convert it to numpy arrays
        # we expect the user to provide this parameter file lines with equal
        # number of floats
        # self.mm.d = [self.mm.d]
        # self.mm.n2 = [self.mm.n2]

    def get_multimedia_params(self):
        """Return multimedia parameters."""
        return self.mm


def read_control_par(filename: str) -> ControlPar:
    """Read control parameters from file and return ControlPar object."""
    cpar = ControlPar()
    cpar.from_file(filename)
    return cpar


def compare_control_par(c1: ControlPar, c2: ControlPar) -> bool:
    """Compare two ControlPar objects."""
    return (
        c1.num_cams == c2.num_cams
        and all(
            getattr(c1, attr) == getattr(c2, attr)
            for attr in ["img_base_name", "cal_img_base_name"]
        )
        and c1.hp_flag == c2.hp_flag
        and c1.allCam_flag == c2.allCam_flag
        and c1.tiff_flag == c2.tiff_flag
        and c1.imx == c2.imx
        and c1.imy == c2.imy
        and c1.pix_x == c2.pix_x
        and c1.pix_y == c2.pix_y
        and c1.chfield == c2.chfield
        and c1.mm == c2.mm
    )


@dataclass
class TargetPar:
    """Target parameters."""

    gvthresh: list[int] = field(default_factory=list)
    discont: int = 100  # discontinuity
    nnmin: int = 1
    nnmax: int = 100
    nxmin: int = 1
    nxmax: int = 100
    nymin: int = 1
    nymax: int = 100
    sumg_min: int = 10  # minimum sum of grey values
    cr_sz: int = 1


def read_target_par(filename: str) -> TargetPar | None:
    """Read target parameters from file and returns target_par object.

    Reads target recognition parameters from a legacy .par file, which
    holds one parameter per line. The arguments are read in this order:

    1. gvthres[0]
    2. gvthres[1]
    3. gvthres[2]
    4. gvthres[3]
    5. discont
    6. nnmin
    7. nnmax
    8. nxmin
    9. nxmax
    10. nymin
    11. nymax
    12. sumg_min
    13. cr_sz



    """
    ret = TargetPar()
    try:
        with open(filename, "r", encoding="utf-8") as file:
            for _ in range(TR_MAX_CAMS):  # todo - make it no. cameras
                ret.gvthresh.append(int(file.readline()))

            ret.discont = int(file.readline())
            ret.nnmin = int(file.readline())
            ret.nnmax = int(file.readline())
            ret.nxmin = int(file.readline())
            ret.nxmax = int(file.readline())
            ret.nymin = int(file.readline())
            ret.nymax = int(file.readline())
            ret.sumg_min = int(file.readline())
            ret.cr_sz = int(file.readline())
        return ret
    except IOError:
        print(f"Could not open target recognition parameters file {filename}.")
        return None


def compare_target_par(targ1: TargetPar, targ2: TargetPar) -> bool:
    """Compare two target_par objects."""
    return all(
        getattr(targ1, attr) == getattr(targ2, attr)
        for attr in TargetPar.__annotations__
    )


def write_target_par(targ: TargetPar, filename: str) -> None:
    """Write target_par object to file."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(
            f"{targ.gvthresh[0]}\n{targ.gvthresh[1]}\n{targ.gvthresh[2]}\n{targ.gvthresh[3]}\n{targ.discont}\n{targ.nnmin}\n{targ.nnmax}\n{targ.nxmin}\n{targ.nxmax}\n{targ.nymin}\n{targ.nymax}\n{targ.sumg_min}\n{targ.cr_sz}"
        )


@dataclass
class OrientPar:
    """Orientation parameters."""

    useflag: int = 0
    ccflag: int = 0
    xhflag: int = 0
    yhflag: int = 0
    k1flag: int = 0
    k2flag: int = 0
    k3flag: int = 0
    p1flag: int = 0
    p2flag: int = 0
    scxflag: int = 0
    sheflag: int = 0
    interfflag: int = 0
