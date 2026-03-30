"""Conversion helpers between openptv_python objects and optv py_bind objects."""

from __future__ import annotations

from types import ModuleType
from typing import Iterable, List

import numpy as np

from ._native_compat import (
    HAS_NATIVE_CALIBRATION,
    HAS_NATIVE_TARGETS,
    HAS_OPTV,
    optv_calibration,
    optv_parameters,
    optv_tracking_framebuf,
)
from .calibration import Calibration
from .parameters import ControlPar, SequencePar, TargetPar, TrackPar, VolumePar
from .tracking_frame_buf import Target


def _require_optv_parameters() -> None:
    if not HAS_OPTV or optv_parameters is None:
        raise RuntimeError("optv py_bind parameters are not available")


def _optv_parameters_module() -> ModuleType:
    _require_optv_parameters()
    assert optv_parameters is not None
    return optv_parameters


def to_native_calibration(cal: Calibration):
    if not isinstance(cal, Calibration):
        return cal

    if not HAS_NATIVE_CALIBRATION or optv_calibration is None:
        raise RuntimeError("optv Calibration is not available")

    native = optv_calibration.Calibration()
    native.set_pos(np.asarray(cal.get_pos(), dtype=np.float64))
    native.set_angles(np.asarray(cal.get_angles(), dtype=np.float64))
    native.set_primary_point(np.asarray(cal.get_primary_point(), dtype=np.float64))
    native.set_radial_distortion(
        np.asarray(cal.get_radial_distortion(), dtype=np.float64)
    )
    native.set_decentering(np.asarray(cal.get_decentering(), dtype=np.float64))
    native.set_affine_trans(np.asarray(cal.get_affine(), dtype=np.float64))
    native.set_glass_vec(np.asarray(cal.get_glass_vec(), dtype=np.float64))
    return native


def from_native_calibration(calibration_obj):
    if isinstance(calibration_obj, Calibration):
        return calibration_obj

    try:
        converted = Calibration()
        converted.set_pos(np.array(calibration_obj.get_pos()))
        converted.set_angles(np.array(calibration_obj.get_angles()))
        converted.set_primary_point(np.array(calibration_obj.get_primary_point()))
        converted.set_radial_distortion(np.array(calibration_obj.get_radial_distortion()))
        converted.set_decentering(np.array(calibration_obj.get_decentering()))

        affine = np.array(calibration_obj.get_affine())
        if hasattr(converted, "set_affine_trans"):
            converted.set_affine_trans(affine)
        elif hasattr(converted, "set_affine_distortion"):
            converted.set_affine_distortion(affine)
        else:
            raise AttributeError("Calibration object does not support affine setters")

        converted.set_glass_vec(np.array(calibration_obj.get_glass_vec()))
        return converted
    except (TypeError, ValueError, AttributeError):
        return calibration_obj


def to_native_control_par(cpar: ControlPar):
    if not isinstance(cpar, ControlPar):
        return cpar

    parameters_module = _optv_parameters_module()

    flags = [
        flag_name
        for enabled, flag_name in (
            (bool(cpar.hp_flag), "hp"),
            (bool(cpar.all_cam_flag), "allcam"),
            (bool(cpar.tiff_flag), "headers"),
        )
        if enabled
    ]

    native = parameters_module.ControlParams(
        cpar.num_cams,
        flags=flags,
        image_size=(cpar.imx, cpar.imy),
        pixel_size=(cpar.pix_x, cpar.pix_y),
        cam_side_n=cpar.mm.n1,
        wall_ns=list(cpar.mm.n2),
        wall_thicks=list(cpar.mm.d),
        object_side_n=cpar.mm.n3,
    )
    native.set_chfield(cpar.chfield)

    for cam_index, img_base_name in enumerate(cpar.img_base_name):
        native.set_img_base_name(cam_index, img_base_name)

    for cam_index, cal_img_base_name in enumerate(cpar.cal_img_base_name):
        native.set_cal_img_base_name(cam_index, cal_img_base_name)

    return native


def to_native_volume_par(vpar: VolumePar):
    if not isinstance(vpar, VolumePar):
        return vpar

    parameters_module = _optv_parameters_module()
    native = parameters_module.VolumeParams()
    native.set_X_lay(list(vpar.get_X_lay()))
    native.set_Zmin_lay(list(vpar.get_Zmin_lay()))
    native.set_Zmax_lay(list(vpar.get_Zmax_lay()))
    native.set_cn(vpar.get_cn())
    native.set_cnx(vpar.get_cnx())
    native.set_cny(vpar.get_cny())
    native.set_csumg(vpar.get_csumg())
    native.set_corrmin(vpar.get_corrmin())
    native.set_eps0(vpar.get_eps0())
    return native


def to_native_sequence_par(spar: SequencePar):
    if not isinstance(spar, SequencePar):
        return spar

    parameters_module = _optv_parameters_module()

    try:
        native = parameters_module.SequenceParams(num_cams=spar.get_num_cams())
    except TypeError:
        native = parameters_module.SequenceParams()

    native.set_first(spar.get_first())
    native.set_last(spar.get_last())

    img_base_name = spar.get_img_base_name()
    if isinstance(img_base_name, (list, tuple)):
        try:
            native.set_img_base_name(list(img_base_name))
        except TypeError:
            for cam_index, base_name in enumerate(img_base_name):
                native.set_img_base_name(cam_index, base_name)

    return native


def to_native_track_par(tpar: TrackPar):
    if not isinstance(tpar, TrackPar):
        return tpar

    parameters_module = _optv_parameters_module()
    native = parameters_module.TrackingParams()
    for attr_name in (
        "dvxmin",
        "dvxmax",
        "dvymin",
        "dvymax",
        "dvzmin",
        "dvzmax",
        "dangle",
        "dacc",
        "add",
        "dsumg",
        "dn",
        "dnx",
        "dny",
    ):
        getter = getattr(tpar, f"get_{attr_name}", None)
        setter = getattr(native, f"set_{attr_name}", None)
        if callable(getter) and callable(setter):
            setter(getter())
    return native


def to_native_target_par(tpar: TargetPar):
    if not isinstance(tpar, TargetPar):
        return tpar

    parameters_module = _optv_parameters_module()

    thresholds = list(tpar.gvthresh)
    if len(thresholds) < 4:
        thresholds.extend([0] * (4 - len(thresholds)))

    return parameters_module.TargetParams(
        discont=tpar.discont,
        gvthresh=thresholds[:4],
        pixel_count_bounds=(tpar.nnmin, tpar.nnmax),
        xsize_bounds=(tpar.nxmin, tpar.nxmax),
        ysize_bounds=(tpar.nymin, tpar.nymax),
        min_sum_grey=tpar.sumg_min,
        cross_size=tpar.cr_sz,
    )


def to_native_target(target: Target):
    if not HAS_NATIVE_TARGETS or optv_tracking_framebuf is None:
        raise RuntimeError("optv Target is not available")

    if hasattr(target, "pnr") and callable(getattr(target, "pnr", None)):
        pnr = int(target.pnr())
    else:
        pnr = int(getattr(target, "pnr"))

    if hasattr(target, "pos") and callable(getattr(target, "pos", None)):
        x, y = target.pos()
    else:
        x = getattr(target, "x")
        y = getattr(target, "y")

    if hasattr(target, "count_pixels") and callable(getattr(target, "count_pixels", None)):
        n, nx, ny = target.count_pixels()
    else:
        n = getattr(target, "n")
        nx = getattr(target, "nx")
        ny = getattr(target, "ny")

    if hasattr(target, "sum_grey_value") and callable(getattr(target, "sum_grey_value", None)):
        sumg = target.sum_grey_value()
    else:
        sumg = getattr(target, "sumg")

    tnr = getattr(target, "tnr", -1)
    if callable(tnr):
        tnr = tnr()

    return optv_tracking_framebuf.Target(
        pnr=pnr,
        x=float(x),
        y=float(y),
        n=int(n),
        nx=int(nx),
        ny=int(ny),
        sumg=int(sumg),
        tnr=int(tnr),
    )


def to_native_target_array(targets: Iterable[Target]):
    if not HAS_NATIVE_TARGETS or optv_tracking_framebuf is None:
        raise RuntimeError("optv TargetArray is not available")

    target_count = len(targets)
    native_targets = optv_tracking_framebuf.TargetArray(target_count)
    for index in range(target_count):
        target = targets[index]
        converted_target = to_native_target(target)
        native_target = native_targets[index]
        native_target.set_pnr(int(converted_target.pnr()))
        native_target.set_tnr(int(converted_target.tnr()))
        native_target.set_pos(tuple(converted_target.pos()))
        native_target.set_pixel_counts(*converted_target.count_pixels())
        native_target.set_sum_grey_value(int(converted_target.sum_grey_value()))
    return native_targets


def from_native_target(native_target) -> Target:
    x, y = native_target.pos()
    n, nx, ny = native_target.count_pixels()
    return Target(
        pnr=int(native_target.pnr()),
        x=float(x),
        y=float(y),
        n=int(n),
        nx=int(nx),
        ny=int(ny),
        sumg=int(native_target.sum_grey_value()),
        tnr=int(native_target.tnr()),
    )


def from_native_target_array(native_targets: Iterable[object]) -> List[Target]:
    return [from_native_target(target) for target in native_targets]
