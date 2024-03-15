"""Parameters for OpenPTV-Python."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from openptv_python.constants import TR_MAX_CAMS


@dataclass
class Parameters:
    """Base class for all parameters with a couple of methods."""

    @classmethod
    def from_dict(cls, data):
        """Read MultimediaPar from a dictionary."""
        return cls(**data)

    @classmethod
    def to_dict(cls, data):
        """Read MultimediaPar from a dictionary."""
        return asdict(data)

    def to_yaml(self, file_path: Path):
        """Write parameters to YAML file."""
        with open(file_path, 'w', encoding='utf-8') as output_file:
            yaml.dump(asdict(self), output_file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, file_path: Path):
        """Read from YAML file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            data_dict = yaml.safe_load(file)
            return cls(**data_dict)



@dataclass
class MultimediaPar(Parameters):
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
class SequencePar(Parameters):
    """Sequence parameters."""

    img_base_name: List[str] = field(default_factory=list)
    first: int = 0
    last: int = 0


    # def to_dict(self):
    #     """Convert SequencePar instance to a dictionary."""
    #     return {
    #         'img_base_name': self.img_base_name,
    #         'first': self.first,
    #         'last': self.last,
    #     }

    def set_img_base_name(self, new_name: List[str]):
        """Set the image base name for each camera."""
        self.img_base_name[:] = new_name

    def get_img_base_name(self, icam: int=0):
        """Get the image base name for each camera."""
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

    @classmethod
    def from_file(cls, filename: Path, num_cams: int):
        """Read sequence parameters from file."""
        if not filename.exists():
            raise IOError("File {filename} does not exist.")

        ret = cls()
        with open(filename, "r", encoding="utf-8") as par_file:
            ret.img_base_name = []
            for _ in range(num_cams):
                ret.img_base_name.append(par_file.readline().strip())
            ret.first = int(par_file.readline().strip())
            ret.last = int(par_file.readline().strip())

        return ret


def read_sequence_par(filename: Path, num_cams: int = TR_MAX_CAMS) -> SequencePar:
    """Read sequence parameters from file and return SequencePar object."""
    return SequencePar().from_file(filename, num_cams)


def compare_sequence_par(sp1: SequencePar, sp2: SequencePar) -> bool:
    """Compare two SequencePar objects."""
    return all(
        getattr(sp1, field) == getattr(sp2, field)
        for field in SequencePar.__annotations__
    )


@dataclass
class TrackPar(Parameters):
    """Tracking parameters."""

    dvxmin: float = 0.0
    dvxmax: float = 0.0
    dvymin: float = 0.0
    dvymax: float = 0.0
    dvzmin: float = 0.0
    dvzmax: float = 0.0
    dangle: float = 0.0
    dacc: float = 0.0
    add: int = 0
    dsumg: float = 0.0
    dn: float = 0.0
    dnx: float = 0.0
    dny: float = 0.0


    # def to_dict(self):
    #     """Convert TrackPar instance to a dictionary."""
    #     return {
    #         'dvxmax': self.dvxmax,
    #         'dvxmin': self.dvxmin,
    #         'dvymax': self.dvymax,
    #         'dvymin': self.dvymin,
    #         'dvzmax': self.dvzmax,
    #         'dvzmin': self.dvzmin,
    #         'dangle': self.dangle,
    #         'dacc': self.dacc,
    #         'add': self.add,
    #         'dsumg': self.dsumg,
    #         'dn': self.dn,
    #         'dnx': self.dnx,
    #         'dny': self.dny,
    #     }

    @classmethod
    def from_file(cls, filename: Path):
        """Read tracking parameters from file and return TrackPar object.

        Note that the structure has 13 attributes, from which we read only 9
        dsumg, dn, dnx, dny are set to 0 automatically and used later in
        track.py

        """
        try:
            with open(filename, "r", encoding="utf-8") as fpp:
                dvxmin = float(fpp.readline().rstrip())
                dvxmax = float(fpp.readline().rstrip())
                dvymin = float(fpp.readline().rstrip())
                dvymax = float(fpp.readline().rstrip())
                dvzmin = float(fpp.readline().rstrip())
                dvzmax = float(fpp.readline().rstrip())
                dangle = float(fpp.readline().rstrip())
                dacc = float(fpp.readline().rstrip())
                add = int(fpp.readline().rstrip())
        except IOError as exc:
            raise (f"Error reading tracking parameters from {filename}") from exc  # type: ignore

        return cls(dvxmin, dvxmax, dvymin, dvymax, dvzmin, dvzmax, dangle, dacc, add)

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


def read_track_par(filename: Path) -> TrackPar:
    """Read tracking parameters from file and return TrackPar object."""
    return TrackPar().from_file(filename)


def compare_track_par(t1: TrackPar, t2: TrackPar) -> bool:
    """Compare two TrackPar objects."""
    return all(getattr(t1, field) == getattr(t2, field) for field in t1.__annotations__)


@dataclass
class VolumePar(Parameters):
    """Volume parameters."""

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

    # def to_dict(self):
    #     """Convert VolumePar instance to a dictionary."""
    #     return {
    #         'x_lay': self.x_lay,
    #         'z_min_lay': self.z_min_lay,
    #         'z_max_lay': self.z_max_lay,
    #         'cn': self.cn,
    #         'cnx': self.cnx,
    #         'cny': self.cny,
    #         'csumg': self.csumg,
    #         'eps0': self.eps0,
    #         'corrmin': self.corrmin,
    #     }

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

    def set_corrmin(self, corrmin: float):
        """Set the minimum correlation value of all criteria."""
        self.corrmin = corrmin

    @classmethod
    def from_file(cls, filename: Path):
        """Read volume parameters from file.

        Args:
        ----
            filename (str): filename

        """
        x_lay, z_min_lay, z_max_lay = [], [], []

        with open(filename, "r", encoding="utf-8") as f:
            x_lay.append(float(f.readline()))
            z_min_lay.append(float(f.readline()))
            z_max_lay.append(float(f.readline()))
            x_lay.append(float(f.readline()))
            z_min_lay.append(float(f.readline()))
            z_max_lay.append(float(f.readline()))
            cnx, cny, cn, csumg, corrmin, eps0 = [
                float(f.readline()) for _ in range(6)
            ]

        return cls(x_lay, z_min_lay, z_max_lay, cn, cnx, cny, csumg, eps0, corrmin)


def read_volume_par(filename: Path) -> VolumePar:
    """Read volume parameters from file and returns volume_par object.

    Args:
    ----
        filename (str): filename

    Returns
    -------
        VolumePar: volume of interest parameters
    """
    return VolumePar().from_file(filename)


def compare_volume_par(v1: VolumePar, v2: VolumePar) -> bool:
    """Compare two VolumePar objects."""
    return all(
        getattr(v1, attr) == getattr(v2, attr) for attr in VolumePar.__annotations__
    )


@dataclass
class ControlPar(Parameters):
    """Control parameters."""

    num_cams: int = field(default_factory=int)
    img_base_name: List[str] = field(default_factory=list)
    cal_img_base_name: List[str] = field(default_factory=list)
    hp_flag: int = field(default=1)
    all_cam_flag: int = field(default=0)
    tiff_flag: int = field(default=1)
    imx: np.int32 = field(default_factory=np.int32)
    imy: np.int32 = field(default_factory=np.int32)
    pix_x: np.float64 = field(default_factory=np.float64)
    pix_y: np.float64 = field(default_factory=np.float64)
    chfield: int = field(default_factory=int)
    mm: MultimediaPar = field(default_factory=MultimediaPar)

    @classmethod
    def from_dict(cls, data):
        """Read ControlPar from a dictionary."""
        mm_data = data.get('mm', {})
        data['mm'] = MultimediaPar.from_dict(mm_data)
        return cls(**data)

    def set_image_size(self, imsize: Tuple[np.int32, np.int32]):
        """Set image size in pixels."""
        self.imx = imsize[0]
        self.imy = imsize[1]

    def get_image_size(self) -> Tuple[np.int32, np.int32]:
        """Set image size in pixels."""
        return (self.imx, self.imy)

    def set_pixel_size(self, pixsize: Tuple[np.float64, np.float64]):
        """Set pixel size in mm."""
        self.pix_x = pixsize[0]
        self.pix_y = pixsize[1]

    def get_pixel_size(self) -> Tuple[np.float64, np.float64]:
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
        return self.all_cam_flag

    def get_tiff_flag(self):
        """Return tiff flag."""
        return self.tiff_flag

    @classmethod
    def from_file(cls, filename: Path):
        """Read control parameters from file and return ControlPar object."""
        ret = cls()
        if not filename.exists():
            raise FileNotFoundError(f"Could not open file {filename}")

        with open(filename, "r", encoding="utf-8") as par_file:
            ret.num_cams = int(par_file.readline().strip())

            for _ in range(ret.num_cams):
                ret.img_base_name.append(par_file.readline().strip())
                ret.cal_img_base_name.append(par_file.readline().strip())

            ret.hp_flag = int(par_file.readline().strip())
            ret.all_cam_flag = int(par_file.readline().strip())
            ret.tiff_flag = int(par_file.readline().strip())
            ret.imx = np.int32(par_file.readline().strip())
            ret.imy = np.int32(par_file.readline().strip())
            ret.pix_x = np.float64(par_file.readline().strip())
            ret.pix_y = np.float64(par_file.readline().strip())
            ret.chfield = int(par_file.readline().strip())
            ret.mm.n1 = float(par_file.readline().strip())
            ret.mm.n2[0] = float(par_file.readline().strip())
            ret.mm.n3 = float(par_file.readline().strip())
            ret.mm.d[0] = float(par_file.readline().strip())

        # the original idea is to have more layers inside water with different
        # refractive indices and different thicknesses.
        # therefore the following lines convert it to numpy arrays
        # we expect the user to provide this parameter file lines with equal
        # number of floats
        # self.mm.d = [self.mm.d]
        # self.mm.n2 = [self.mm.n2]

        return ret

    def get_multimedia_params(self):
        """Return multimedia parameters."""
        return self.mm

    @classmethod
    def to_dict(cls, data):
        """Convert ControlPar instance to a dictionary."""
        control_par_dict = asdict(data)
        if isinstance(control_par_dict['mm'], MultimediaPar):
            control_par_dict['mm'] = asdict(control_par_dict['mm'])
        return control_par_dict


def read_control_par(filename: Path) -> ControlPar:
    """Read control parameters from file and return ControlPar object."""
    return ControlPar().from_file(filename)


def compare_control_par(c1: ControlPar, c2: ControlPar) -> bool:
    """Compare two ControlPar objects."""
    return (
        c1.num_cams == c2.num_cams
        and all(
            getattr(c1, attr) == getattr(c2, attr)
            for attr in ["img_base_name", "cal_img_base_name"]
        )
        and c1.hp_flag == c2.hp_flag
        and c1.all_cam_flag == c2.all_cam_flag
        and c1.tiff_flag == c2.tiff_flag
        and c1.imx == c2.imx
        and c1.imy == c2.imy
        and c1.pix_x == c2.pix_x
        and c1.pix_y == c2.pix_y
        and c1.chfield == c2.chfield
        and c1.mm == c2.mm
    )



@dataclass
class TargetPar(Parameters):
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

    # @classmethod
    # def from_dict(cls, data):
    #     """Read from target.par dictionary."""
    #     return cls(**data)

    # def to_dict(self):
    #     """Convert TargetPar instance to a dictionary."""
    #     return {
    #         'gvthresh': self.gvthresh,
    #         'discont': self.discont,
    #         'nnmin': self.nnmin,
    #         'nnmax': self.nnmax,
    #         'nxmin': self.nxmin,
    #         'nxmax': self.nxmax,
    #         'nymin': self.nymin,
    #         'nymax': self.nymax,
    #         'sumg_min': self.sumg_min,
    #         'cr_sz': self.cr_sz,
    #     }

    @classmethod
    def from_file(cls, filename: Path):
        """Read target parameters from file and returns target_par object.

        Reads target recognition parameters from a legacy detect_plate.par file,
        which holds one parameter per line. The arguments are read in this order:

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
        ret = cls()
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
            # return ret
        except IOError:
            print(f"Could not open target recognition parameters file {filename}.")
            # return None

        return ret

    def get_grey_thresholds(self):
        """Return the grey thresholds."""
        return self.gvthresh

    def get_pixel_count_bounds(self):
        """Return the pixel count bounds."""
        return (self.nnmin, self.nnmax)

    def get_xsize_bounds(self):
        """Return the xsize bounds."""
        return (self.nxmin, self.nxmax)

    def get_ysize_bounds(self):
        """Return the ysize bounds."""
        return (self.nymin, self.nymax)

    def get_min_sum_grey(self):
        """Return the sum grey bounds."""
        return self.sumg_min

def read_target_par(filename: Path) -> TargetPar:
    """Read target parameters from file and returns target_par object."""
    tpar = TargetPar()
    return tpar.from_file(filename)

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
            f"{targ.gvthresh[0]}\n{targ.gvthresh[1]}\n{targ.gvthresh[2]}\n{targ.gvthresh[3]}\n{targ.discont}\n{targ.nnmin}\n{targ.nnmax}\n{targ.nxmin}\n{targ.nymax}\n{targ.nymin}\n{targ.nymax}\n{targ.sumg_min}\n{targ.cr_sz}"
        )


@dataclass
class OrientPar(Parameters):
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

    @classmethod
    def from_file(cls, filename: Path):
        """Read orientation parameters from file and returns orient_par object."""
        ret = cls()
        try:
            with open(filename, "r", encoding="utf-8") as file:

                ret.useflag = int(file.readline().strip())  # /* use every point or every other pt */
                ret.ccflag = int(file.readline().strip())   # /* change back focal distance */
                ret.xhflag = int(file.readline().strip())   # /* change xh point, 1-yes, 0-no */
                ret.yhflag = int(file.readline().strip())
                ret.k1flag = int(file.readline().strip())
                ret.k2flag = int(file.readline().strip())
                ret.k3flag = int(file.readline().strip())
                ret.p1flag = int(file.readline().strip())
                ret.p2flag = int(file.readline().strip())
                ret.scxflag = int(file.readline().strip())  # /* scx - scaling  */
                ret.sheflag = int(file.readline().strip())  # /* she - shearing  */
                ret.interfflag = int(file.readline().strip())   # /* interface glass vector */

        except IOError:
            print(f"Could not open orientation parameters file {filename}.")

        return ret



@dataclass
class CalibrationPar(Parameters):
    """Calibration parameters."""

    fixp_name: str = ''
    img_name: list = field(default_factory=list)
    img_ori0: list = field(default_factory=list)
    tiff_flag: int = 0
    pair_flag: int = 0
    chfield: int = 0


    # def to_dict(self):
    #     """Convert CalibrationPar instance to a dictionary."""
    #     return {
    #         'fixp_name': self.fixp_name,
    #         'img_name': self.img_name,
    #         'img_ori0': self.img_ori0,
    #         'tiff_flag': self.tiff_flag,
    #         'pair_flag': self.pair_flag,
    #         'chfield': self.chfield,
    #     }

    @classmethod
    def from_file(cls, file_path: str, num_cams: int):
        """Read from cal_ori.par file."""
        with open(file_path, 'r', encoding="utf-8") as file:
            fixp_name = file.readline().strip()
            tmp = [file.readline().strip() for _ in range(num_cams*2)]
            # img_ori0 = [file.readline().strip() for _ in range(4)]
            img_name = tmp[0::2]
            img_ori0 = tmp[1::2]
            tiff_flag = int(file.readline().strip())
            pair_flag = int(file.readline().strip())
            chfield = int(file.readline().strip())

        return cls(fixp_name, img_name, img_ori0, tiff_flag, pair_flag, chfield)


def read_cal_ori_parameters(file_path: Path, num_cams: int) -> CalibrationPar:
    """Read from cal_ori.par file."""
    with open(file_path, 'r', encoding="utf-8") as file:
        fixp_name = file.readline().strip()
        tmp = [file.readline().strip() for _ in range(num_cams*2)]
        # img_ori0 = [file.readline().strip() for _ in range(4)]
        img_name = tmp[0::2]
        img_ori0 = tmp[1::2]
        tiff_flag = int(file.readline().strip())
        pair_flag = int(file.readline().strip())
        chfield = int(file.readline().strip())

    return CalibrationPar(fixp_name, img_name, img_ori0, tiff_flag, pair_flag, chfield)

@dataclass
class MultiPlanesPar(Parameters):
    """Multiplanes parameters."""

    num_planes: int = field(default_factory=int)
    filename: list = field(default_factory=list)

    @classmethod
    def from_file(cls, file_path: Path):
        """Read from multiplanes.par file."""
        with open(file_path, 'r', encoding="utf-8") as file:
            num_planes = int(file.readline().strip())
            filename = [file.readline().strip() for _ in range(num_planes)]
        return cls(num_planes, filename)

@dataclass
class ExaminePar(Parameters):
    """Examine parameters."""

    examine_flag: bool = False
    combine_flag: bool = False

    @classmethod
    def from_file(cls, file_path: Path):
        """Read from examine.par file."""
        with open(file_path, 'r', encoding="utf-8") as file:
            examine_flag = bool(int(file.readline().strip()))
            combine_flag = bool(int(file.readline().strip()))
        return cls(examine_flag, combine_flag)

def read_examine_par(file_path: Path) -> ExaminePar:
    """Read from examine.par file."""
    with open(file_path, 'r', encoding="utf-8") as file:
        examine_flag = bool(int(file.readline().strip()))
        combine_flag = bool(int(file.readline().strip()))
    return ExaminePar(examine_flag, combine_flag)

@dataclass
class PftVersionPar(Parameters):
    """Pft version parameters."""

    existing_target_flag: bool = False

    @classmethod
    def from_file(cls, file_path: Path):
        """Read from pft_version.par file."""
        with open(file_path, 'r', encoding="utf-8") as file:
            pft_version = bool(int(file.readline().strip()))
        return cls(pft_version)

    @classmethod
    def write(cls, file_path: Path):
        """Write to pft_version.par file."""
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write(f"{cls.existing_target_flag}\n")
