"""Tracking run module."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

from openptv_python.calibration import Calibration
from openptv_python.tracking_frame_buf import FrameBuf

from ._native_compat import get_num_cams
from .multimed import volumedimension
from .parameters import (
    ControlPar,
    SequencePar,
    TrackParTuple,
    VolumePar,
    convert_track_par_to_tuple,
    read_control_par,
    read_sequence_par,
    read_track_par,
    read_volume_par,
)


def _tracking_param_value(tpar, name: str):
    """Read a tracking parameter from either attribute or getter access."""
    value = getattr(tpar, name, None)
    if value is not None:
        return value

    getter = getattr(tpar, f"get_{name}", None)
    if callable(getter):
        return getter()

    raise AttributeError(f"TrackingParams object does not expose {name}")


def _volume_param_value(vpar, name: str):
    """Read a volume parameter from either attribute or legacy getter access."""
    value = getattr(vpar, name, None)
    if value is not None:
        return value

    getter_name = {
        "x_lay": "get_X_lay",
        "z_min_lay": "get_Zmin_lay",
        "z_max_lay": "get_Zmax_lay",
    }.get(name, f"get_{name}")
    getter = getattr(vpar, getter_name, None)
    if callable(getter):
        return getter()

    raise AttributeError(f"VolumeParams object does not expose {name}")


def _set_volume_param_value(vpar, name: str, value) -> None:
    """Write a volume parameter through either a setter or direct attribute."""
    setter_name = {
        "x_lay": "set_X_lay",
        "z_min_lay": "set_Zmin_lay",
        "z_max_lay": "set_Zmax_lay",
    }.get(name, f"set_{name}")
    setter = getattr(vpar, setter_name, None)
    if callable(setter):
        setter(value)
        return

    setattr(vpar, name, value)


@dataclass
class TrackingRun:
    """A tracking run."""

    fb: FrameBuf
    seq_par: SequencePar
    tpar: TrackParTuple
    vpar: VolumePar
    cpar: ControlPar
    cal: List[Calibration]
    flatten_tol: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0
    lmax: float = 0.0
    npart: int = 0
    nlinks: int = 0

    def __init__(
        self,
        seq_par: SequencePar,
        tpar: TrackParTuple,
        vpar: VolumePar,
        cpar: ControlPar,
        buf_len: int,
        max_targets: int,
        corres_file_base: str,
        linkage_file_base: str,
        prio_file_base: str,
        cal: List[Calibration],
        flatten_tol: float,
    ):
        self.tpar = tpar
        self.vpar = vpar
        self.cpar = cpar
        self.seq_par = seq_par
        self.cal = cal
        self.flatten_tol = flatten_tol

        if hasattr(seq_par, "img_base_name"):
            img_base_names = seq_par.img_base_name
        else:
            img_base_names = [
                seq_par.get_img_base_name(cam_index)
                for cam_index in range(get_num_cams(cpar))
            ]

        self.fb = FrameBuf(
            buf_len,
            get_num_cams(cpar),
            max_targets,
            corres_file_base,
            linkage_file_base,
            prio_file_base,
            img_base_names,
        )

        self.lmax = math.sqrt(
            (_tracking_param_value(tpar, "dvxmin") - _tracking_param_value(tpar, "dvxmax")) ** 2
            + (_tracking_param_value(tpar, "dvymin") - _tracking_param_value(tpar, "dvymax")) ** 2
            + (_tracking_param_value(tpar, "dvzmin") - _tracking_param_value(tpar, "dvzmax")) ** 2
        )

        x_lay = list(_volume_param_value(vpar, "x_lay"))
        z_min_lay = list(_volume_param_value(vpar, "z_min_lay"))
        z_max_lay = list(_volume_param_value(vpar, "z_max_lay"))

        (
            x_lay[1],
            x_lay[0],
            self.ymax,
            self.ymin,
            z_max_lay[1],
            z_min_lay[0],
        ) = volumedimension(
            x_lay[1],
            x_lay[0],
            self.ymax,
            self.ymin,
            z_max_lay[1],
            z_min_lay[0],
            vpar,
            cpar,
            cal,
        )

        _set_volume_param_value(vpar, "x_lay", x_lay)
        _set_volume_param_value(vpar, "z_min_lay", z_min_lay)
        _set_volume_param_value(vpar, "z_max_lay", z_max_lay)

        self.npart = 0
        self.nlinks = 0


def tr_new(
    seq_par_fname: Path,
    tpar_fname: Path,
    vpar_fname: Path,
    cpar_fname: Path,
    buf_len: int,
    max_targets: int,
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    cal: List[Calibration],
    flatten_tol: float,
) -> TrackingRun:
    """Create a new tracking run from legacy files."""
    cpar = read_control_par(cpar_fname)
    seq_par = read_sequence_par(seq_par_fname, get_num_cams(cpar))
    tpar = convert_track_par_to_tuple(read_track_par(tpar_fname))
    vpar = read_volume_par(vpar_fname)

    tr = TrackingRun(
        seq_par,
        tpar,
        vpar,
        cpar,
        buf_len,
        max_targets,
        corres_file_base,
        linkage_file_base,
        prio_file_base,
        cal,
        flatten_tol,
    )

    return tr
