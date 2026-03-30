"""Headless/standalone calibration utilities.

This module is intentionally GUI-free (no Traits/Tk/Qt imports).
It is used by CLI scripts and tests to run calibration end-to-end.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from openptv_python._native_compat import should_use_native
from openptv_python.parameters import OrientPar
from openptv_python.calibration import Calibration as PythonCalibration

from ._backend import (
    Calibration,
    external_calibration,
    full_calibration,
    match_detection_to_ref,
    TargetArray,
)

from .parameter_manager import ParameterManager
from . import ptv


NAMES: list[str] = ["cc", "xh", "yh", "k1", "k2", "k3", "p1", "p2", "scale", "shear"]


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def load_parameter_manager(yaml_path: Path) -> ParameterManager:
    pm = ParameterManager()
    pm.from_yaml(yaml_path)
    return pm


def get_flags_from_yaml(pm: ParameterManager) -> list[str]:
    orient = pm.parameters.get("orient")
    if not isinstance(orient, dict):
        return []
    return [name for name in NAMES if _as_bool(orient.get(name, False))]


def _orient_par_from_flags(flags: Sequence[str]) -> OrientPar:
    orient = OrientPar()
    enabled = set(flags)

    orient.ccflag = int("cc" in enabled)
    orient.xhflag = int("xh" in enabled)
    orient.yhflag = int("yh" in enabled)
    orient.k1flag = int("k1" in enabled)
    orient.k2flag = int("k2" in enabled)
    orient.k3flag = int("k3" in enabled)
    orient.p1flag = int("p1" in enabled)
    orient.p2flag = int("p2" in enabled)
    orient.scxflag = int("scale" in enabled)
    orient.sheflag = int("shear" in enabled)

    return orient


def load_points_npz(npz_path: Path, *, num_cams: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load correspondences from NPZ.

    Returns (xyz, xy, pnr)
    - xyz: (M,3) float
    - xy: (C,N,2) float
    - pnr: (N,) int, mapping each xy point to xyz index
    """

    data = np.load(npz_path, allow_pickle=False)

    if "xyz" in data:
        xyz = np.asarray(data["xyz"], dtype=float)
    elif "XYZ" in data:
        xyz = np.asarray(data["XYZ"], dtype=float)
    else:
        raise KeyError("NPZ must contain 'xyz' or 'XYZ' array")

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N,3); got {xyz.shape}")

    pnr = None
    if "pnr" in data:
        pnr = np.asarray(data["pnr"], dtype=int)
        if pnr.ndim != 1:
            raise ValueError(f"pnr must be shape (N,); got {pnr.shape}")

    if "xy" in data:
        xy = np.asarray(data["xy"], dtype=float)
        if xy.ndim != 3 or xy.shape[2] != 2:
            raise ValueError(f"xy must have shape (C,N,2); got {xy.shape}")
    else:
        per_cam = []
        for cam in range(num_cams):
            key = f"xy_cam{cam}"
            if key not in data:
                break
            per_cam.append(np.asarray(data[key], dtype=float))
        if not per_cam:
            raise KeyError(
                "NPZ must contain 'xy' (C,N,2) or 'xy_cam0', 'xy_cam1', ... arrays"
            )
        xy = np.stack(per_cam, axis=0)

    if xy.shape[0] != num_cams:
        raise ValueError(
            f"xy camera dimension ({xy.shape[0]}) must match num_cams ({num_cams})"
        )

    n = xy.shape[1]
    if n == 0:
        raise ValueError("xy has zero points")

    if xyz.shape[0] < n and pnr is None:
        raise ValueError(
            f"xyz has only {xyz.shape[0]} points but xy provides {n}; provide 'pnr' mapping"
        )

    if pnr is None:
        pnr = np.arange(n, dtype=int)
    else:
        if pnr.shape[0] != n:
            raise ValueError(f"pnr length ({pnr.shape[0]}) must match xy N ({n})")
        if np.any(pnr < 0) or np.any(pnr >= xyz.shape[0]):
            raise ValueError("pnr contains indices out of bounds of xyz")

    return xyz, xy, pnr


def targets_from_xy(xy_cam: np.ndarray, pnr: np.ndarray) -> TargetArray:
    """Create an OpenPTV TargetArray from (N,2) pixel points."""

    xy_cam = np.asarray(xy_cam, dtype=float)
    if xy_cam.ndim != 2 or xy_cam.shape[1] != 2:
        raise ValueError(f"xy_cam must be (N,2); got {xy_cam.shape}")

    targs = TargetArray(len(xy_cam))
    for i, (pos, p) in enumerate(zip(xy_cam, pnr)):
        t = targs[i]
        t.set_pnr(int(p))
        t.set_pos([float(pos[0]), float(pos[1])])
    return targs


def _select_four_indices(mode: str, n: int) -> np.ndarray:
    if mode == "first4":
        if n < 4:
            raise ValueError("Need at least 4 points for external init")
        return np.arange(4, dtype=int)
    raise ValueError(f"Unknown init mode: {mode}")


def _select_manual_orientation_indices(pm: ParameterManager, cam: int, n: int) -> np.ndarray:
    man_ori = pm.parameters.get("man_ori")
    if isinstance(man_ori, dict):
        nr = man_ori.get("nr")
        if isinstance(nr, list) and len(nr) >= (cam + 1) * 4:
            cam_nr = nr[cam * 4 : cam * 4 + 4]
            indices = np.asarray([int(idx) - 1 for idx in cam_nr], dtype=int)
            if np.all((0 <= indices) & (indices < n)):
                return indices

    return _select_four_indices("first4", n)


def _load_or_init_calibration(ori_path: Path) -> Calibration:
    addpar_path = Path(str(ori_path).replace(".ori", ".addpar"))

    if ori_path.exists() and addpar_path.exists():
        if should_use_native("orientation"):
            cal = Calibration()
            cal.from_file(str(ori_path), str(addpar_path))
            return cal
        return PythonCalibration.from_file(str(ori_path), str(addpar_path))

    return Calibration() if should_use_native("orientation") else PythonCalibration()


def run_standalone_calibration(
    yaml_path: Path,
    xyz: np.ndarray,
    xy: np.ndarray,
    pnr: np.ndarray,
    *,
    flags: Sequence[str],
    init_external: str | None = "first4",
    write: bool = False,
) -> list[Calibration]:
    """Run calibration for all cameras and (optionally) write .ori/.addpar."""

    experiment_dir = yaml_path.parent
    previous_cwd = Path.cwd()

    try:
        os.chdir(experiment_dir)

        pm = load_parameter_manager(yaml_path)
        params = pm.parameters

        ptv_params = params.get("ptv")
        cal_ori = params.get("cal_ori")
        if not isinstance(ptv_params, dict) or not isinstance(cal_ori, dict):
            raise KeyError("YAML must contain 'ptv' and 'cal_ori' mappings")

        num_cams = int(pm.num_cams or params.get("num_cams") or xy.shape[0])
        if num_cams != xy.shape[0]:
            raise ValueError(f"num_cams ({num_cams}) != xy.shape[0] ({xy.shape[0]})")

        # Build ControlParams (cpar) from YAML
        cpar, *_rest = ptv.py_start_proc_c(pm)

        ori_files = cal_ori.get("img_ori")
        if not ori_files or len(ori_files) < num_cams:
            raise ValueError("cal_ori.img_ori must list one .ori path per camera")

        calibrations: list[Calibration] = []

        for cam in range(num_cams):
            ori_path = (
                (experiment_dir / ori_files[cam]).resolve()
                if not Path(ori_files[cam]).is_absolute()
                else Path(ori_files[cam])
            )
            targs = targets_from_xy(xy[cam], pnr)

            fallback_cal = _load_or_init_calibration(ori_path)
            cal = _load_or_init_calibration(ori_path)
            external_ok = True
            if init_external is not None:
                if init_external == "first4":
                    idx4 = _select_four_indices("first4", len(pnr))
                else:
                    idx4 = _select_four_indices(init_external, len(pnr))
                sel_xyz = np.asarray(xyz[pnr[idx4]], dtype=float)
                sel_xy = np.asarray(xy[cam][idx4], dtype=float)

                ok = external_calibration(cal, sel_xyz, sel_xy, cpar)
                if ok is False:
                    external_ok = False
                    print(
                        f"external_calibration failed for camera {cam}; "
                        "continuing with the loaded calibration"
                    )
                    cal = fallback_cal

            if external_ok:
                sorted_targs = match_detection_to_ref(
                    cal,
                    np.asarray(xyz, dtype=float),
                    targs,
                    cpar,
                )

                try:
                    residuals, targ_ix, err_est = full_calibration(
                        cal,
                        np.asarray(xyz, dtype=float),
                        sorted_targs,
                        cpar,
                        list(flags) if should_use_native("orientation") else _orient_par_from_flags(flags),
                    )
                except ValueError as exc:
                    print(f"full_calibration failed for camera {cam}; using loaded calibration: {exc}")
                    cal = fallback_cal
            else:
                residuals = np.zeros((len(targs), 2), dtype=float)
                targ_ix = np.asarray([t.pnr() if hasattr(t, "pnr") else -999 for t in targs], dtype=int)
                err_est = np.zeros(11, dtype=float)

            packed = np.array(
                [
                    cal.get_pos(),
                    cal.get_angles(),
                    cal.get_affine(),
                    cal.get_decentering(),
                    cal.get_radial_distortion(),
                ],
                dtype=object,
            )
            if np.any(np.isnan(np.hstack(packed))):
                cal = fallback_cal

            if write:
                addpar_path = Path(str(ori_path).replace(".ori", ".addpar"))
                ori_path.parent.mkdir(parents=True, exist_ok=True)
                cal.write(str(ori_path).encode(), str(addpar_path).encode())

            calibrations.append(cal)

        return calibrations
    finally:
        os.chdir(previous_cwd)
