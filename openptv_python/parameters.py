"""Parameters for OpenPTV-Python."""
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MultimediaPar:
    """Multimedia parameters."""

    n1: float
    n2: np.ndarray
    d: np.ndarray
    n3: float
    nlay: int

    def __init__(self, n1, n2, n3, d):
        """Initialize MultimediaPar object."""
        self.n1 = n1
        self.n2 = np.array(n2)
        self.d = np.array(d)
        self.n3 = n3
        if self.n2.shape[0] != self.d.shape[0]:
            raise ValueError("n2 and d must have the same length")

        self.nlay = self.d.shape[0]

    def get_nlay(self):
        return self.nlay

    def get_n1(self):
        return self.n1

    def get_n3(self):
        return self.n3

    def get_n2(self):
        return self.n2

    def get_d(self):
        return self.d


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

    num_cams: int = 1
    img_base_name: list[str] = field(default_factory=list)
    first: int = 10000
    last: int = 10004


def read_sequence_par(filename: str, num_cams: int) -> SequencePar:
    """Read sequence parameters from file and return SequencePar object."""
    try:
        with open(filename, "r", encoding="utf-8") as par_file:
            ret = SequencePar(num_cams=num_cams)
            ret.img_base_name = [par_file.readline().strip() for _ in range(num_cams)]
            ret.first = int(par_file.readline().strip())
            ret.last = int(par_file.readline().strip())
            return ret
    except FileNotFoundError:
        return None


def compare_sequence_par(sp1: SequencePar, sp2: SequencePar) -> bool:
    """Compare two SequencePar objects."""
    return all(
        getattr(sp1, field) == getattr(sp2, field)
        for field in SequencePar.__annotations__
    )


@dataclass
class TrackPar:
    """Tracking parameters."""

    dacc: float = 0.0
    dangle: float = 100.0
    dvxmax: float = 1.0
    dvxmin: float = 0.0
    dvymax: float = 1.0
    dvymin: float = 0.0
    dvzmax: float = 1.0
    dvzmin: float = 0.0
    dsumg: float = 1.0
    dn: float = 1.0
    dnx: float = 1.0
    dny: float = 1.0
    add: bool = False


def read_track_par(filename: str) -> TrackPar:
    """Read tracking parameters from file and return TrackPar object."""
    try:
        with open(filename, "r", encoding="utf-8") as fpp:
            return TrackPar(
                dvxmin=float(fpp.readline().rstrip()),
                dvxmax=float(fpp.readline().rstrip()),
                dvymin=float(fpp.readline().rstrip()),
                dvymax=float(fpp.readline().rstrip()),
                dvzmin=float(fpp.readline().rstrip()),
                dvzmax=float(fpp.readline().rstrip()),
                dangle=float(fpp.readline().rstrip()),
                dacc=float(fpp.readline().rstrip()),
                add=bool(int(fpp.readline().rstrip())),
                dsumg=0,
                dn=0,
                dnx=0,
                dny=0,
            )
    except IOError:
        print(f"Error reading tracking parameters from {filename}")
        return None


def compare_track_par(t1: TrackPar, t2: TrackPar) -> bool:
    """Compare two TrackPar objects."""
    return all(getattr(t1, field) == getattr(t2, field) for field in t1.__annotations__)


@dataclass
class VolumePar:
    """Volume parameters."""

    X_lay: list[float] = field(default_factory=list)
    Zmin_lay: float = -10
    Zmax_lay: float = 10
    cn: float = 1
    cnx: float = 1
    cny: float = 1
    csumg: float = 1
    eps0: float = 0.01
    corrmin: float = 0.2


def read_volume_par(filename: str) -> VolumePar:
    """Read volume parameters from file and returns volume_par object.

    Args:
    ----
        filename (str): filename

    Returns:
    -------
        VolumePar: volume of interest parameters
    """
    with open(filename, "r", encoding="utf-8") as f:
        X_lay = [float(f.readline()) for _ in range(2)]
        Zmin_lay = float(f.readline())
        Zmax_lay = float(f.readline())
        cnx, cny, cn, csumg, corrmin, eps0 = [float(f.readline()) for _ in range(6)]
        return VolumePar(
            X_lay=X_lay,
            Zmin_lay=Zmin_lay,
            Zmax_lay=Zmax_lay,
            cn=cn,
            cnx=cnx,
            cny=cny,
            csumg=csumg,
            eps0=eps0,
            corrmin=corrmin,
        )


def compare_volume_par(v1: VolumePar, v2: VolumePar) -> bool:
    """Compare two VolumePar objects."""
    return all(
        getattr(v1, attr) == getattr(v2, attr) for attr in VolumePar.__annotations__
    )


@dataclass
class ControlPar:
    """Control parameters."""

    num_cams: int = 1
    img_base_name: list[str] = field(default_factory=list)
    cal_img_base_name: list[str] = field(default_factory=list)
    hp_flag: bool = True
    allCam_flag: bool = False
    tiff_flag: bool = True
    imx: int = 1024
    imy: int = 1024
    pix_x: float = 0.01
    pix_y: float = 0.01
    chfield: int = 1
    mm: MultimediaPar = MultimediaPar(n1=1, n2=[1], n3=1, d=[1])

    def set_image_size(self, imx, imy):
        """Set image size in pixels."""
        self.imx = imx
        self.imy = imy

    def set_pixel_size(self, pix_x, pix_y):
        """Set pixel size in mm."""
        self.pix_x = pix_x
        self.pix_y = pix_y

    def get_multimedia_par(self, cam):
        """Return multimedia parameters for camera cam."""
        return self.mm[cam]


def read_control_par(filename: str) -> ControlPar:
    """Read control parameters from file and return ControlPar object."""
    if not os.path.isfile(filename):
        print(f"Could not open file {filename}")
        return None

    with open(filename, "r") as par_file:
        num_cams = int(par_file.readline().strip())
        ret = ControlPar(num_cams=num_cams, mm={})

        for cam in range(ret.num_cams):
            ret.img_base_name = par_file.readline().strip()
            ret.cal_img_base_name = par_file.readline().strip()

        ret.hp_flag, ret.allCam_flag, ret.tiff_flag = map(
            bool, map(int, par_file.readline().split())
        )
        ret.imx, ret.imy, ret.pix_x, ret.pix_y, ret.chfield = map(
            int, par_file.readline().split()
        )
        ret.mm["n1"], ret.mm["n2"], ret.mm["n3"], ret.mm["d"] = map(
            float, par_file.readline().split()
        )

    return ret


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

    discont: int = 100  # discontinuity
    gvthresh: list[int] = field(default_factory=list)
    nnmin: int = 1
    nnmax: int = 100
    nxmin: int = 1
    nxmax: int = 100
    nymin: int = 1
    nymax: int = 100
    sumg_min: int = 10  # minimum sum of grey values
    cr_sz: int = 1


def read_target_par(filename: str) -> TargetPar:
    """Read target parameters from file and returns target_par object."""
    ret = TargetPar()
    try:
        with open(filename, "r", encoding="utf-8") as file:
            ret.gvthresh[0] = int(file.readline())
            ret.gvthresh[1] = int(file.readline())
            ret.gvthresh[2] = int(file.readline())
            ret.gvthresh[3] = int(file.readline())
            ret.discont = int(file.readline())
            line = file.readline().split()
            if len(line) == 2:
                ret.nnmin, ret.nnmax = map(int, line)
            line = file.readline().split()
            if len(line) == 2:
                ret.nxmin, ret.nxmax = map(int, line)
            line = file.readline().split()
            if len(line) == 2:
                ret.nymin, ret.nymax = map(int, line)
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
