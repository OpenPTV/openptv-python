from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from openptv_python.calibration import Calibration
from openptv_python.tracking_frame_buf import FrameBuf, FrameBufBase

from .multimed import volumedimension
from .parameters import (
    ControlPar,
    SequencePar,
    TrackPar,
    VolumePar,
    read_control_par,
    read_sequence_par,
    read_track_par,
    read_volume_par,
)


@dataclass
class TrackingRun:
    """A tracking run."""

    fb: Optional[FrameBufBase] = None
    seq_par: Optional[SequencePar] = None
    tpar: Optional[TrackPar] = None
    vpar: Optional[VolumePar] = None
    cpar: Optional[ControlPar] = None
    cal: Optional[List[Calibration]] = None
    flatten_tol: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0
    lmax: float = 0.0
    npart: int = 0
    nlinks: int = 0

    def __init__(
        self,
        seq_par: SequencePar,
        tpar: TrackPar,
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

        self.fb = FrameBuf(
            buf_len,
            cpar.num_cams,
            max_targets,
            corres_file_base,
            linkage_file_base,
            prio_file_base,
            cpar.img_base_name,
        )

        self.lmax = np.linalg.norm(
            (tpar.dvxmin - tpar.dvxmax),
            (tpar.dvymin - tpar.dvymax),
            (tpar.dvzmin - tpar.dvzmax),
        )

        volumedimension(
            vpar.x_lay[1],
            vpar.x_lay[0],
            self.ymax,
            self.ymin,
            vpar.z_max_lay[1],
            vpar.z_min_lay[0],
            vpar,
            cpar,
            cal,
        )

        self.npart = 0
        self.nlinks = 0


def tr_new(
    seq_par: SequencePar,
    tpar: TrackPar,
    vpar: VolumePar,
    cpar: ControlPar,
    buf_len: int,
    max_targets: int,
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    cal: List[Calibration],
    flatten_tol: float,
) -> TrackingRun:
    """Create a new tracking run."""
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


def tr_new_legacy(
    seq_par_fname: str,
    tpar_fname: str,
    vpar_fname: str,
    cpar_fname: str,
    cal: List[Calibration],
) -> TrackingRun:
    """Create a new tracking run from legacy files."""
    cpar = read_control_par(cpar_fname)
    seq_par = read_sequence_par(seq_par_fname, cpar.num_cams)
    tpar = read_track_par(tpar_fname)
    vpar = read_volume_par(vpar_fname)

    tr = tr_new(
        seq_par,
        tpar,
        vpar,
        cpar,
        4,
        20000,
        "res/rt_is",
        "res/ptv_is",
        "res/added",
        cal,
        10000,
    )

    return tr


def track_forward_start(tr: TrackingRun):
    """Not implemented."""
    pass


def trackcorr_c_finish(tr: TrackingRun):
    """Not implemented."""
    pass


def trackcorr_c_loop(tr: TrackingRun):
    """Not implemented."""
    pass
