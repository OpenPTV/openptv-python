"""Parameters for OpenPTV-Python."""
import os
from pathlib import Path
from typing import List, Tuple

import yaml

# from numba.experimental import jitclass
from openptv_python.constants import TR_MAX_CAMS

# # @jitclass
# class Parameters:
#     """Base class for all parameters with a couple of methods."""

#     def __init__(self, **kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#     @classmethod
#     def from_dict(cls, data):
#         """Read Parameters from a dictionary."""
#         return cls(**data)

#     @classmethod
#     def to_dict(cls, instance):
#         """Convert Parameters instance to a dictionary."""
#         return {attr: getattr(instance, attr) for attr in dir(instance)
#                 if not callable(getattr(instance, attr)) and not attr.startswith("__")}

#     def to_yaml(self, file_path: Path):
#         """Write parameters to YAML file."""
#         with open(file_path, 'w', encoding='utf-8') as output_file:
#             yaml.dump(self.to_dict(self), output_file,
#                       default_flow_style=False)

#     @classmethod
#     def from_yaml(cls, file_path: Path):
#         """Read from YAML file."""
#         with open(file_path, 'r', encoding='utf-8') as file:
#             data_dict = yaml.safe_load(file)
#             return cls.from_dict(data_dict)


# @jitclass
class MultimediaPar:
    """Multimedia parameters."""

    def __init__(
        self,
        nlay: int = 1,
        n1: float = 1.0,
        n2: List[float] | None = None,
        d: List[float] | None = None,
        n3: float = 1.0,
    ):
        if n2 is None:
            n2 = [1.0]
        if d is None:
            d = [1.0]

        self.nlay = nlay
        self.n1 = n1
        self.n2 = n2
        self.d = d
        self.n3 = n3

        if len(self.n2) != len(self.d):
            raise ValueError("n2 and d must have the same length")

    def to_dict(self):
        """Convert MultimediaPar instance to a dictionary."""
        return {
            "nlay": self.nlay,
            "n1": self.n1,
            "n2": self.n2,
            "d": self.d,
            "n3": self.n3,
        }

    def from_dict(self, data: dict):
        """Read MultimediaPar from a dictionary."""
        return MultimediaPar(**data)

    def to_yaml(self, file_path: Path):
        """Write parameters to YAML file."""
        with open(file_path, "w", encoding="utf-8") as output_file:
            yaml.dump(self.to_dict(), output_file, default_flow_style=False)

    def from_yaml(self, file_path: Path):
        """Read from YAML file."""
        with open(file_path, "r", encoding="utf-8") as file:
            data_dict = yaml.safe_load(file)
            return MultimediaPar(**data_dict)

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
    return mm_np1.to_dict() == mm_np2.to_dict()


class SequencePar(object):
    """Sequence parameters."""

    def __init__(
        self,
        img_base_name: List[str] | None = None,
        first: int = 0,
        last: int = 0
    ):
        if img_base_name is None:
            img_base_name = []

        self.img_base_name = img_base_name
        self.first = first
        self.last = last

    def to_dict(self):
        """Convert SequencePar instance to a dictionary."""
        return {
            "img_base_name": self.img_base_name,
            "first": self.first,
            "last": self.last,
        }

    def from_dict(self, data):
        """Read SequencePar from a dictionary."""
        return SequencePar(**data)

    def to_yaml(self, file_path: Path):
        """Write parameters to YAML file."""
        with open(file_path, "w", encoding="utf-8") as output_file:
            yaml.dump(self.to_dict(), output_file, default_flow_style=False)

    def from_yaml(self, file_path: Path):
        """Read from YAML file."""
        with open(file_path, "r", encoding="utf-8") as file:
            data_dict = yaml.safe_load(file)
            return SequencePar(**data_dict)

    def set_img_base_name(self, new_name: List[str]):
        """Set the image base name for each camera."""
        self.img_base_name[:] = new_name

    def get_img_base_name(self, icam: int = 0):
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
    def from_file(cls, filename: str, num_cams: int):
        """Read sequence parameters from file."""
        if not Path(filename).exists():
            raise IOError("File {filename} does not exist.")

        ret = cls()
        with open(filename, "r", encoding="utf-8") as par_file:
            ret.img_base_name = []
            for _ in range(num_cams):
                ret.img_base_name.append(par_file.readline().strip())
            ret.first = int(par_file.readline().strip())
            ret.last = int(par_file.readline().strip())

        return ret

    def __eq__(self, other) -> bool:
        return self.to_dict() == other.to_dict()


def read_sequence_par(filename: str, num_cams: int = TR_MAX_CAMS) -> SequencePar:
    """Read sequence parameters from file and return SequencePar object."""
    return SequencePar().from_file(filename, num_cams)


class TrackPar:
    """Tracking parameters."""

    def __init__(
        self,
        dvxmin=0.0,
        dvxmax=0.0,
        dvymin=0.0,
        dvymax=0.0,
        dvzmin=0.0,
        dvzmax=0.0,
        dangle=0.0,
        dacc=0.0,
        add=0,
        dsumg=0.0,
        dn=0.0,
        dnx=0.0,
        dny=0.0,
    ):
        self.dvxmin = dvxmin
        self.dvxmax = dvxmax
        self.dvymin = dvymin
        self.dvymax = dvymax
        self.dvzmin = dvzmin
        self.dvzmax = dvzmax
        self.dangle = dangle
        self.dacc = dacc
        self.add = add
        self.dsumg = dsumg
        self.dn = dn
        self.dnx = dnx
        self.dny = dny

    def to_dict(self):
        """Convert TrackPar instance to a dictionary."""
        return {
            "dvxmin": self.dvxmin,
            "dvxmax": self.dvxmax,
            "dvymin": self.dvymin,
            "dvymax": self.dvymax,
            "dvzmin": self.dvzmin,
            "dvzmax": self.dvzmax,
            "dangle": self.dangle,
            "dacc": self.dacc,
            "add": self.add,
            "dsumg": self.dsumg,
            "dn": self.dn,
            "dnx": self.dnx,
            "dny": self.dny,
        }

    def from_dict(self, data):
        """Read TrackPar from a dictionary."""
        return TrackPar(**data)

    def to_yaml(self, file_path: Path):
        """Write parameters to YAML file."""
        with open(file_path, "w", encoding="utf-8") as output_file:
            yaml.dump(self.to_dict(), output_file, default_flow_style=False)

    def from_yaml(self, file_path: Path):
        """Read from YAML file."""
        with open(file_path, "r", encoding="utf-8") as file:
            data_dict = yaml.safe_load(file)
            return TrackPar(**data_dict)

    @classmethod
    def from_file(cls, filename: str):
        """Read tracking parameters from file and return TrackPar object.

        Note that the structure has 13 attributes, from which we read only 9
        dsumg, dn, dnx, dny are set to 0 automatically and used later in
        track.py

        """
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


def read_track_par(filename: str) -> TrackPar:
    """Read tracking parameters from file and return TrackPar object."""
    return TrackPar().from_file(filename)


def compare_track_par(t1: TrackPar, t2: TrackPar) -> bool:
    """Compare two TrackPar objects."""
    return all(getattr(t1, field) == getattr(t2, field) for field in t1.__annotations__)


class VolumePar:
    """Volume parameters."""

    def __init__(
        self,
        x_lay: List[float] = [0.0, 0.0],
        z_min_lay: List[float] = [0.0, 0.0],
        z_max_lay: List[float] = [0.0, 0.0],
        cn: float = 0.0,
        cnx: float = 0.0,
        cny: float = 0.0,
        csumg: float = 0.0,
        eps0: float = 0.0,
        corrmin: float = 0.0,
    ):
        self.x_lay = x_lay
        self.z_min_lay = z_min_lay
        self.z_max_lay = z_max_lay
        self.cn = cn
        self.cnx = cnx
        self.cny = cny
        self.csumg = csumg
        self.eps0 = eps0
        self.corrmin = corrmin

    def to_dict(self):
        """Convert VolumePar instance to a dictionary."""
        return {
            "x_lay": self.x_lay,
            "z_min_lay": self.z_min_lay,
            "z_max_lay": self.z_max_lay,
            "cn": self.cn,
            "cnx": self.cnx,
            "cny": self.cny,
            "csumg": self.csumg,
            "eps0": self.eps0,
            "corrmin": self.corrmin,
        }

    def from_dict(self, data):
        """Read VolumePar from a dictionary."""
        return VolumePar(**data)

    def to_yaml(self, file_path: Path):
        """Write parameters to YAML file."""
        with open(file_path, "w", encoding="utf-8") as output_file:
            yaml.dump(self.to_dict(), output_file, default_flow_style=False)

    def from_yaml(self, file_path: Path):
        """Read from YAML file."""
        with open(file_path, "r", encoding="utf-8") as file:
            data_dict = yaml.safe_load(file)
            return VolumePar(**data_dict)

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
    def from_file(cls, filename: str):
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
            cnx, cny, cn, csumg, corrmin, eps0 = [float(f.readline()) for _ in range(6)]

        return cls(x_lay, z_min_lay, z_max_lay, cn, cnx, cny, csumg, eps0, corrmin)


def read_volume_par(filename: str) -> VolumePar:
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


class ControlPar:
    """Control parameters."""

    def __init__(
        self,
        num_cams: int = 1,
        img_base_name: List[str] | None = None,
        cal_img_base_name: List[str] | None = None,
        hp_flag: int = 1,
        all_cam_flag: int = 0,
        tiff_flag: int = 1,
        imx: int = 0,
        imy: int = 0,
        pix_x: float = 0.0,
        pix_y: float = 0.0,
        chfield: int = 0,
        mm: MultimediaPar | None = None,
    ):
        self.num_cams = num_cams

        if mm is None:
            mm = MultimediaPar()

        if img_base_name is None:
            img_base_name = []

        if cal_img_base_name is None:
            cal_img_base_name = []

        self.imx = imx
        self.imy = imy
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.chfield = chfield

        self.mm = mm

        self.img_base_name = img_base_name
        self.cal_img_base_name = cal_img_base_name
        self.hp_flag = hp_flag
        self.all_cam_flag = all_cam_flag
        self.tiff_flag = tiff_flag

    def to_dict(self):
        """Convert ControlPar instance to a dictionary."""
        control_par_dict = {
            "num_cams": self.num_cams,
            "img_base_name": self.img_base_name,
            "cal_img_base_name": self.cal_img_base_name,
            "hp_flag": self.hp_flag,
            "all_cam_flag": self.all_cam_flag,
            "tiff_flag": self.tiff_flag,
            "imx": self.imx,
            "imy": self.imy,
            "pix_x": self.pix_x,
            "pix_y": self.pix_y,
            "chfield": self.chfield,
            "mm": self.mm.to_dict() if isinstance(self.mm, MultimediaPar) else self.mm,
        }
        return control_par_dict

    @classmethod
    def from_dict(cls, data: dict):
        """Create ControlPar instance from a dictionary."""
        mm_instance = MultimediaPar().from_dict(data["mm"])

        return ControlPar(
            num_cams=data.get("num_cams", 0),
            img_base_name=data.get("img_base_name", [""]),
            cal_img_base_name=data.get("cal_img_base_name", [""]),
            hp_flag=data.get("hp_flag", 1),
            all_cam_flag=data.get("all_cam_flag", 0),
            tiff_flag=data.get("tiff_flag", 1),
            imx=data.get("imx", 0),
            imy=data.get("imy", 0),
            pix_x=data.get("pix_x", 0.0),
            pix_y=data.get("pix_y", 0.0),
            chfield=data.get("chfield", 0),
            mm=mm_instance,
        )

    def from_yaml(self, file_path: Path):
        """Read from YAML file."""
        with open(file_path, "r", encoding="utf-8") as file:
            data_dict = yaml.safe_load(file)
            return self.from_dict(data_dict)

    def to_yaml(self, file_path: Path):
        """Write parameters to YAML file."""
        with open(file_path, "w", encoding="utf-8") as output_file:
            yaml.dump(self.to_dict(), output_file, default_flow_style=False)

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
        return self.all_cam_flag

    def get_tiff_flag(self):
        """Return tiff flag."""
        return self.tiff_flag

    @classmethod
    def from_file(cls, filename: str):
        """Read control parameters from file and return ControlPar object."""
        ret = cls()
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Could not open file {filename}")

        with open(filename, "r", encoding="utf-8") as par_file:
            ret.num_cams = int(par_file.readline().strip())

            for _ in range(ret.num_cams):
                ret.img_base_name.append(par_file.readline().strip())
                ret.cal_img_base_name.append(par_file.readline().strip())

            ret.hp_flag = int(par_file.readline().strip())
            ret.all_cam_flag = int(par_file.readline().strip())
            ret.tiff_flag = int(par_file.readline().strip())
            ret.imx = int(par_file.readline().strip())
            ret.imy = int(par_file.readline().strip())
            ret.pix_x = float(par_file.readline().strip())
            ret.pix_y = float(par_file.readline().strip())
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


def read_control_par(filename: str) -> ControlPar:
    """Read control parameters from file and return ControlPar object."""
    ret = ControlPar()
    ret.from_file(filename)
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
        and c1.all_cam_flag == c2.all_cam_flag
        and c1.tiff_flag == c2.tiff_flag
        and c1.imx == c2.imx
        and c1.imy == c2.imy
        and c1.pix_x == c2.pix_x
        and c1.pix_y == c2.pix_y
        and c1.chfield == c2.chfield
        and c1.mm == c2.mm
    )


# class TargetPar(Parameters):
#     """Target parameters."""

#     gvthresh: list[int] = field(default_factory=list)
#     discont: int = 100  # discontinuity
#     nnmin: int = 1
#     nnmax: int = 100
#     nxmin: int = 1
#     nxmax: int = 100
#     nymin: int = 1
#     nymax: int = 100
#     sumg_min: int = 10  # minimum sum of grey values
#     cr_sz: int = 1


class TargetPar:
    """Target parameters."""

    def __init__(
        self,
        gvthresh: List[int] | None = None,
        discont: int = 100,
        nnmin: int = 1,
        nnmax: int = 100,
        nxmin: int = 1,
        nxmax: int = 100,
        nymin: int = 1,
        nymax: int = 100,
        sumg_min: int = 10,
        cr_sz: int = 1,
    ):
        if gvthresh is None:
            gvthresh = [0, 0, 0, 0]

        self.gvthresh = gvthresh
        self.discont = discont
        self.nnmin = nnmin
        self.nnmax = nnmax
        self.nxmin = nxmin
        self.nxmax = nxmax
        self.nymin = nymin
        self.nymax = nymax
        self.sumg_min = sumg_min
        self.cr_sz = cr_sz

    @classmethod
    def from_dict(cls, data):
        """Read from target.par dictionary."""
        return cls(**data)

    def to_dict(self):
        """Convert TargetPar instance to a dictionary."""
        return {
            "gvthresh": self.gvthresh,
            "discont": self.discont,
            "nnmin": self.nnmin,
            "nnmax": self.nnmax,
            "nxmin": self.nxmin,
            "nxmax": self.nxmax,
            "nymin": self.nymin,
            "nymax": self.nymax,
            "sumg_min": self.sumg_min,
            "cr_sz": self.cr_sz,
        }

    def from_yaml(self, file_path: Path):
        """Write parameters to YAML file."""
        with open(file_path, "w", encoding="utf-8") as output_file:
            yaml.dump(self.to_dict(), output_file, default_flow_style=False)

    @classmethod
    def from_file(cls, filename: str):
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
                for i in range(TR_MAX_CAMS):  # todo - make it no. cameras
                    ret.gvthresh[i] = int(file.readline())

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


def read_target_par(filename: str) -> TargetPar:
    """Read target parameters from file and returns target_par object."""
    out = TargetPar()
    out.from_file(filename)
    return out


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


class OrientPar:
    """Orientation parameters."""

    def __init__(
        self,
        useflag: int = 0,
        ccflag: int = 0,
        xhflag: int = 0,
        yhflag: int = 0,
        k1flag: int = 0,
        k2flag: int = 0,
        k3flag: int = 0,
        p1flag: int = 0,
        p2flag: int = 0,
        scxflag: int = 0,
        sheflag: int = 0,
        interfflag: int = 0,
    ):
        self.useflag = useflag
        self.ccflag = ccflag
        self.xhflag = xhflag
        self.yhflag = yhflag
        self.k1flag = k1flag
        self.k2flag = k2flag
        self.k3flag = k3flag
        self.p1flag = p1flag
        self.p2flag = p2flag
        self.scxflag = scxflag
        self.sheflag = sheflag
        self.interfflag = interfflag

    def to_dict(self):
        """Convert OrientPar instance to a dictionary."""
        return {
            "useflag": self.useflag,
            "ccflag": self.ccflag,
            "xhflag": self.xhflag,
            "yhflag": self.yhflag,
            "k1flag": self.k1flag,
            "k2flag": self.k2flag,
            "k3flag": self.k3flag,
            "p1flag": self.p1flag,
            "p2flag": self.p2flag,
            "scxflag": self.scxflag,
            "sheflag": self.sheflag,
            "interfflag": self.interfflag,
        }

    @classmethod
    def from_dict(cls, data):
        """Create OrientPar instance from a dictionary."""
        return cls(**data)

    def to_yaml(self, file_path: Path):
        """Write parameters to YAML file."""
        with open(file_path, "w", encoding="utf-8") as output_file:
            yaml.dump(self.to_dict(), output_file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, file_path: Path):
        """Read from YAML file."""
        with open(file_path, "r", encoding="utf-8") as file:
            data_dict = yaml.safe_load(file)
            return cls(**data_dict)

    @classmethod
    def from_file(cls, filename: str):
        """Read orientation parameters from file and returns orient_par object."""
        try:
            with open(filename, "r", encoding="utf-8") as file:
                ret = cls()
                # /* use every point or every other pt */
                ret.useflag = int(file.readline().strip())
                # /* change back focal distance */
                ret.ccflag = int(file.readline().strip())
                # /* change xh point, 1-yes, 0-no */
                ret.xhflag = int(file.readline().strip())
                ret.yhflag = int(file.readline().strip())
                ret.k1flag = int(file.readline().strip())
                ret.k2flag = int(file.readline().strip())
                ret.k3flag = int(file.readline().strip())
                ret.p1flag = int(file.readline().strip())
                ret.p2flag = int(file.readline().strip())
                # /* scx - scaling  */
                ret.scxflag = int(file.readline().strip())
                # /* she - shearing  */
                ret.sheflag = int(file.readline().strip())
                # /* interface glass vector */
                ret.interfflag = int(file.readline().strip())
                return ret
        except IOError:
            print(f"Could not open orientation parameters file {filename}.")
            return None


class CalibrationPar:
    """Calibration parameters."""

    def __init__(
        self,
        fixp_name: str | None = None,
        img_name: List[str] | None = None,
        img_ori0: List[str] | None = None,
        tiff_flag: int = 0,
        pair_flag: int = 0,
        chfield: int = 0,
    ):
        if fixp_name is None:
            fixp_name = ""
        if img_name is None:
            img_name = [""]
        if img_ori0 is None:
            img_ori0 = [""]

        self.fixp_name = fixp_name
        self.img_name = img_name
        self.img_ori0 = img_ori0
        self.tiff_flag = tiff_flag
        self.pair_flag = pair_flag
        self.chfield = chfield

    def to_dict(self):
        """Convert CalibrationPar instance to a dictionary."""
        return {
            "fixp_name": self.fixp_name,
            "img_name": self.img_name,
            "img_ori0": self.img_ori0,
            "tiff_flag": self.tiff_flag,
            "pair_flag": self.pair_flag,
            "chfield": self.chfield,
        }

    def from_dict(self, data):
        """Read CalibrationPar from a dictionary."""
        return CalibrationPar(**data)

    @classmethod
    def from_file(cls, file_path: str, num_cams: int):
        """Read from cal_ori.par file."""
        with open(file_path, "r", encoding="utf-8") as file:
            fixp_name = file.readline().strip()
            tmp = [file.readline().strip() for _ in range(num_cams * 2)]
            # img_ori0 = [file.readline().strip() for _ in range(4)]
            img_name = tmp[0::2]
            img_ori0 = tmp[1::2]
            tiff_flag = int(file.readline().strip())
            pair_flag = int(file.readline().strip())
            chfield = int(file.readline().strip())

        return cls(fixp_name, img_name, img_ori0, tiff_flag, pair_flag, chfield)


def read_cal_ori_parameters(file_path: str, num_cams: int) -> CalibrationPar:
    """Read from cal_ori.par file."""
    with open(file_path, "r", encoding="utf-8") as file:
        fixp_name = file.readline().strip()
        tmp = [file.readline().strip() for _ in range(num_cams * 2)]
        # img_ori0 = [file.readline().strip() for _ in range(4)]
        img_name = tmp[0::2]
        img_ori0 = tmp[1::2]
        tiff_flag = int(file.readline().strip())
        pair_flag = int(file.readline().strip())
        chfield = int(file.readline().strip())

    return CalibrationPar(fixp_name, img_name, img_ori0, tiff_flag, pair_flag, chfield)


class MultiPlanesPar:
    """Multiplanes parameters."""

    def __init__(self,
                 num_planes: int = 0,
                 filename: List[str] | None = None):
        if filename is None:
            filename = []

        self.num_planes = num_planes
        self.filename = filename

    def to_dict(self):
        """Convert MultiPlanesPar instance to a dictionary."""
        return {
            "num_planes": self.num_planes,
            "filename": self.filename,
        }

    def from_dict(self, data):
        """Read MultiPlanesPar from a dictionary."""
        return MultiPlanesPar(**data)

    def to_yaml(self, file_path: Path):
        """Write parameters to YAML file."""
        with open(file_path, "w", encoding="utf-8") as output_file:
            yaml.dump(self.to_dict(), output_file, default_flow_style=False)

    def from_yaml(self, file_path: Path):
        """Read from YAML file."""
        with open(file_path, "r", encoding="utf-8") as file:
            data_dict = yaml.safe_load(file)
            return MultiPlanesPar(**data_dict)

    @classmethod
    def from_file(cls, file_path: str):
        """Read from multiplanes.par file."""
        with open(file_path, "r", encoding="utf-8") as file:
            num_planes = int(file.readline().strip())
            filename = [file.readline().strip() for _ in range(num_planes)]
        return cls(num_planes, filename)


class ExaminePar:
    """Examine parameters."""

    def __init__(self, examine_flag: bool = False, combine_flag: bool = False):
        self.examine_flag = examine_flag
        self.combine_flag = combine_flag

    def to_dict(self):
        """Convert ExaminePar instance to a dictionary."""
        return {
            "examine_flag": self.examine_flag,
            "combine_flag": self.combine_flag,
        }

    def from_dict(self, data):
        """Read ExaminePar from a dictionary."""
        return ExaminePar(**data)

    def to_yaml(self, file_path: Path):
        """Write parameters to YAML file."""
        with open(file_path, "w", encoding="utf-8") as output_file:
            yaml.dump(self.to_dict(), output_file, default_flow_style=False)

    def from_yaml(self, file_path: Path):
        """Read from YAML file."""
        with open(file_path, "r", encoding="utf-8") as file:
            data_dict = yaml.safe_load(file)
            return ExaminePar(**data_dict)

    @classmethod
    def from_file(cls, file_path: str):
        """Read from examine.par file."""
        with open(file_path, "r", encoding="utf-8") as file:
            examine_flag = bool(int(file.readline().strip()))
            combine_flag = bool(int(file.readline().strip()))
        return cls(examine_flag, combine_flag)


class PftVersionPar:
    """Pft version parameters."""

    def __init__(self, existing_target_flag: bool = False):
        self.existing_target_flag = existing_target_flag

    def to_dict(self):
        """Convert PftVersionPar instance to a dictionary."""
        return {
            "existing_target_flag": self.existing_target_flag,
        }

    def from_dict(self, data):
        """Read PftVersionPar from a dictionary."""
        return PftVersionPar(**data)

    def to_yaml(self, file_path: Path):
        """Write parameters to YAML file."""
        with open(file_path, "w", encoding="utf-8") as output_file:
            yaml.dump(self.to_dict(), output_file, default_flow_style=False)

    def from_yaml(self, file_path: Path):
        """Read from YAML file."""
        with open(file_path, "r", encoding="utf-8") as file:
            data_dict = yaml.safe_load(file)
            return PftVersionPar(**data_dict)

    @classmethod
    def from_file(cls, file_path: str):
        """Read from pft_version.par file."""
        with open(file_path, "r", encoding="utf-8") as file:
            pft_version = bool(file.readline().strip())
        return cls(pft_version)

    def write(self, file_path: str):
        """Write to pft_version.par file."""
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"{self.existing_target_flag}\n")
