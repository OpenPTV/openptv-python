"""Generate a deterministic synthetic case modeled after test_cavity."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, cast

import numpy as np

from .calibration import Calibration, read_calibration, write_calibration
from .imgcoord import image_coordinates
from .orientation import (
    external_calibration,
    full_calibration,
    initialize_bundle_adjustment_points,
    match_detection_to_ref,
)
from .parameters import ControlPar, OrientPar, VolumePar, read_volume_par
from .sortgrid import read_sortgrid_par
from .tracking_frame_buf import (
    Pathinfo,
    Target,
    n_tupel_dtype,
    write_path_frame,
    write_targets,
)
from .trafo import arr_metric_to_pixel

DEFAULT_SOURCE_CASE = Path("tests/testing_folder/test_cavity")
DEFAULT_OUTPUT_CASE = Path("tests/testing_folder/test_cavity_synthetic")
DEFAULT_SEED = 20260306
FRAME_NUMBERS = (10001, 10002)


@dataclass(frozen=True)
class SyntheticCaseSummary:
    """Metadata describing the generated synthetic case."""

    seed: int
    num_calibration_points: int
    num_frames: int
    particles_per_frame: int
    calibration_position_errors: List[float]
    calibration_angle_errors: List[float]


def clone_calibration(cal: Calibration) -> Calibration:
    """Return a detached calibration copy."""
    return Calibration(
        ext_par=cal.ext_par.copy(),
        int_par=cal.int_par.copy(),
        glass_par=cal.glass_par.copy(),
        added_par=cal.added_par.copy(),
        mmlut=cal.mmlut,
        mmlut_data=cal.mmlut_data,
    )


def make_target(x: float, y: float, pnr: int) -> Target:
    """Create a simple synthetic target record."""
    return Target(
        pnr=pnr,
        x=float(x),
        y=float(y),
        n=21,
        nx=5,
        ny=5,
        sumg=5000,
        tnr=-1,
    )


def camera_points_in_bounds(
    pixel_points: np.ndarray, cpar: ControlPar, margin: float = 8.0
) -> np.ndarray:
    """Return a mask of points lying safely inside the sensor bounds."""
    return (
        (pixel_points[:, 0] >= margin)
        & (pixel_points[:, 0] <= cpar.imx - margin)
        & (pixel_points[:, 1] >= margin)
        & (pixel_points[:, 1] <= cpar.imy - margin)
    )


def z_bounds_at_x(x_coord: float, vpar: VolumePar) -> tuple[float, float]:
    """Interpolate the admissible z range for one x coordinate."""
    x0, x1 = vpar.x_lay
    zmin0, zmin1 = vpar.z_min_lay
    zmax0, zmax1 = vpar.z_max_lay
    if x1 == x0:
        return zmin0, zmax0
    weight = (x_coord - x0) / (x1 - x0)
    z_min = zmin0 + weight * (zmin1 - zmin0)
    z_max = zmax0 + weight * (zmax1 - zmax0)
    return float(z_min), float(z_max)


def project_pixels(
    points_3d: np.ndarray, cals: Sequence[Calibration], cpar: ControlPar
) -> np.ndarray:
    """Project 3D points into all cameras in pixel coordinates."""
    observed = np.empty((points_3d.shape[0], len(cals), 2), dtype=np.float64)
    for cam_index, cal in enumerate(cals):
        observed[:, cam_index, :] = arr_metric_to_pixel(
            image_coordinates(points_3d, cal, cpar.mm),
            cpar,
        )
    return observed


def select_visible_points(
    candidate_points: np.ndarray,
    cals: Sequence[Calibration],
    cpar: ControlPar,
    count: int,
) -> np.ndarray:
    """Return the first points that project inside every camera image."""
    projected = project_pixels(candidate_points, cals, cpar)
    visibility = np.ones(candidate_points.shape[0], dtype=bool)
    for cam_index in range(len(cals)):
        visibility &= camera_points_in_bounds(projected[:, cam_index, :], cpar)
    visible_points = candidate_points[visibility]
    if visible_points.shape[0] < count:
        raise ValueError("Not enough visible points for the requested synthetic case")
    return visible_points[:count]


def build_calibration_body(
    vpar: VolumePar, cals: Sequence[Calibration], cpar: ControlPar
) -> np.ndarray:
    """Build a structured 3D calibration body visible in all cameras."""
    xs = np.linspace(vpar.x_lay[0] + 4.0, vpar.x_lay[1] - 4.0, 8)
    ys = np.linspace(-16.0, 16.0, 6)
    candidates = []
    for x_coord in xs:
        z_min, z_max = z_bounds_at_x(float(x_coord), vpar)
        if z_max - z_min < 10.0:
            continue
        zs = np.linspace(z_min + 4.0, z_max - 4.0, 4)
        for y_coord in ys:
            for z_coord in zs:
                candidates.append([float(x_coord), float(y_coord), float(z_coord)])
    candidate_points = np.asarray(candidates, dtype=np.float64)
    return select_visible_points(candidate_points, cals, cpar, count=48)


def generate_particle_cloud(
    rng: np.random.Generator,
    vpar: VolumePar,
    cals: Sequence[Calibration],
    cpar: ControlPar,
    count: int,
) -> np.ndarray:
    """Sample random 3D particles inside the observed volume and keep visible quadruplets."""
    accepted: List[np.ndarray] = []
    while len(accepted) < count:
        x_coord = float(rng.uniform(vpar.x_lay[0] + 2.0, vpar.x_lay[1] - 2.0))
        z_min, z_max = z_bounds_at_x(x_coord, vpar)
        y_coord = float(rng.uniform(-18.0, 18.0))
        z_coord = float(rng.uniform(z_min + 2.0, z_max - 2.0))
        point = np.asarray([[x_coord, y_coord, z_coord]], dtype=np.float64)
        projected = project_pixels(point, cals, cpar)[0]
        if all(
            camera_points_in_bounds(projected[None, cam_index, :], cpar)[0]
            for cam_index in range(len(cals))
        ):
            accepted.append(point[0])
    return np.asarray(accepted, dtype=np.float64)


def shuffled_targets_from_pixels(
    pixel_points: np.ndarray,
    rng: np.random.Generator,
    noise_sigma: float,
) -> tuple[List[Target], np.ndarray]:
    """Shuffle projected pixels into a target list and return point-to-target indices."""
    noisy_pixels = pixel_points + rng.normal(0.0, noise_sigma, size=pixel_points.shape)
    order = rng.permutation(pixel_points.shape[0])
    targets = [
        make_target(
            noisy_pixels[target_index, 0], noisy_pixels[target_index, 1], list_index
        )
        for list_index, target_index in enumerate(order)
    ]
    point_to_target = np.empty(pixel_points.shape[0], dtype=np.int32)
    for list_index, target_index in enumerate(order):
        point_to_target[target_index] = list_index
    return targets, point_to_target


def write_calibration_body_points(points: np.ndarray, output_file: Path) -> None:
    """Write a calblock-compatible calibration body file."""
    lines = [
        f"{index + 1:11d}{point[0]:11.3f}{point[1]:11.3f}{point[2]:11.3f}"
        for index, point in enumerate(points)
    ]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def perturb_calibration_for_recovery(
    cal: Calibration, camera_index: int
) -> Calibration:
    """Create a deterministic seed calibration for full_calibration recovery."""
    trial = clone_calibration(cal)
    position_deltas = [
        np.array([1.0, -0.8, 0.4]),
        np.array([-0.7, 0.5, -0.3]),
        np.array([0.6, 0.4, -0.5]),
        np.array([-0.5, -0.6, 0.4]),
    ]
    angle_deltas = [
        np.array([0.006, -0.004, 0.003]),
        np.array([-0.005, 0.004, -0.003]),
        np.array([0.004, 0.003, -0.002]),
        np.array([-0.004, -0.003, 0.002]),
    ]
    trial.set_pos(trial.get_pos() + position_deltas[camera_index])
    trial.set_angles(trial.get_angles() + angle_deltas[camera_index])
    return trial


def select_external_seed_subset(ref_points: np.ndarray) -> np.ndarray:
    """Pick a well-spread subset of reference points for external calibration."""
    mins = np.argmin(ref_points, axis=0)
    maxs = np.argmax(ref_points, axis=0)
    center = np.argmin(np.linalg.norm(ref_points - np.mean(ref_points, axis=0), axis=1))
    indices = []
    for index in np.concatenate([mins, maxs, np.array([center])]):
        if int(index) not in indices:
            indices.append(int(index))
    return np.asarray(indices[:6], dtype=np.int32)


def recover_calibrations_from_body(
    truth_cals: Sequence[Calibration],
    ref_points: np.ndarray,
    calibration_targets_dir: Path,
    output_cal_dir: Path,
    truth_cal_dir: Path,
    cpar: ControlPar,
    orient_par: OrientPar,
    sortgrid_eps: int,
    rng: np.random.Generator,
) -> tuple[List[Calibration], List[float], List[float]]:
    """Recover working calibrations from synthetic calibration-body targets."""
    recovered = []
    position_errors = []
    angle_errors = []

    seed_indices = select_external_seed_subset(ref_points)

    for camera_index, truth_cal in enumerate(truth_cals, start=1):
        projected = arr_metric_to_pixel(
            image_coordinates(ref_points, truth_cal, cpar.mm), cpar
        )
        targets, _ = shuffled_targets_from_pixels(projected, rng, noise_sigma=0.0)
        write_targets(
            targets,
            len(targets),
            str(calibration_targets_dir / f"cam{camera_index}.%05d"),
            1,
        )

        seed = perturb_calibration_for_recovery(truth_cal, camera_index - 1)
        external_calibration(
            seed,
            ref_points[seed_indices],
            projected[seed_indices],
            cpar,
        )
        sorted_targets = match_detection_to_ref(
            seed, ref_points, targets, cpar, sortgrid_eps
        )
        full_calibration(
            seed,
            ref_points,
            cast(np.ndarray, sorted_targets),
            cpar,
            orient_par,
        )
        recovered.append(seed)

        write_calibration(
            seed,
            output_cal_dir / f"cam{camera_index}.tif.ori",
            output_cal_dir / f"cam{camera_index}.tif.addpar",
        )
        write_calibration(
            truth_cal,
            truth_cal_dir / f"cam{camera_index}.tif.ori",
            truth_cal_dir / f"cam{camera_index}.tif.addpar",
        )

        position_errors.append(
            float(np.linalg.norm(seed.get_pos() - truth_cal.get_pos()))
        )
        angle_errors.append(
            float(np.linalg.norm(seed.get_angles() - truth_cal.get_angles()))
        )

    return recovered, position_errors, angle_errors


def build_frame_targets_and_paths(
    frame_points: np.ndarray,
    cals: Sequence[Calibration],
    cpar: ControlPar,
    rng: np.random.Generator,
) -> tuple[list[list[Target]], np.recarray, list[Pathinfo]]:
    """Create target files plus rt_is-compatible correspondences for one frame."""
    projected = project_pixels(frame_points, cals, cpar)
    per_camera_targets = []
    per_camera_mapping = []
    for cam_index in range(len(cals)):
        targets, point_to_target = shuffled_targets_from_pixels(
            projected[:, cam_index, :], rng, noise_sigma=0.08
        )
        per_camera_targets.append(targets)
        per_camera_mapping.append(point_to_target)

    observed_pixels = np.empty_like(projected)
    for point_index in range(frame_points.shape[0]):
        for cam_index in range(len(cals)):
            target_index = per_camera_mapping[cam_index][point_index]
            observed_pixels[point_index, cam_index, 0] = per_camera_targets[cam_index][
                target_index
            ].x
            observed_pixels[point_index, cam_index, 1] = per_camera_targets[cam_index][
                target_index
            ].y

    initial_points, _ = initialize_bundle_adjustment_points(
        observed_pixels, list(cals), cpar
    )
    cor_buf = np.recarray((frame_points.shape[0],), dtype=n_tupel_dtype)
    path_buf = [Pathinfo() for _ in range(frame_points.shape[0])]
    for point_index in range(frame_points.shape[0]):
        cor_buf[point_index].p = np.array(
            [
                per_camera_mapping[cam_index][point_index]
                for cam_index in range(len(cals))
            ],
            dtype=np.int32,
        )
        cor_buf[point_index].corr = 1.0
        path_buf[point_index].x = initial_points[point_index]
    return per_camera_targets, cor_buf, path_buf


def write_sequence_file(sequence_path: Path, frame_numbers: Sequence[int]) -> None:
    """Write a sequence.par file consistent with img_orig target files."""
    lines = [
        "img_orig/cam1.%05d",
        "img_orig/cam2.%05d",
        "img_orig/cam3.%05d",
        "img_orig/cam4.%05d",
        str(frame_numbers[0]),
        str(frame_numbers[-1]),
    ]
    sequence_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_output_case(source_case: Path, output_case: Path) -> None:
    """Copy the source structure and clear generated data directories."""
    shutil.copytree(source_case, output_case, dirs_exist_ok=True)
    for name in ("cal", "img_orig", "res_orig", "ground_truth", "calibration_targets"):
        path = output_case / name
        if path.exists():
            shutil.rmtree(path)
    (output_case / "cal").mkdir(parents=True, exist_ok=True)
    (output_case / "img_orig").mkdir(parents=True, exist_ok=True)
    (output_case / "res_orig").mkdir(parents=True, exist_ok=True)
    (output_case / "ground_truth" / "cal").mkdir(parents=True, exist_ok=True)
    (output_case / "ground_truth" / "particles").mkdir(parents=True, exist_ok=True)
    (output_case / "calibration_targets").mkdir(parents=True, exist_ok=True)


def write_case_readme(output_case: Path) -> None:
    """Document the generated synthetic case contents."""
    text = """# Synthetic Cavity Case

This case is generated deterministically from the geometry of `test_cavity`,
but all observations come from known ground truth.

Contents:

- `cal/`: working calibrations recovered from synthetic calibration-body targets using `full_calibration`.
- `ground_truth/cal/`: exact camera models used to project the synthetic data.
- `ground_truth/calibration_body_points.txt`: known 3D calibration-body points.
- `calibration_targets/`: synthetic target files for that calibration body.
- `img_orig/`: synthetic particle target files for two frames.
- `res_orig/`: synthetic `rt_is`, `ptv_is`, and `added` files for those frames.
- `ground_truth/particles/`: exact 3D particle coordinates per frame.
- `ground_truth/manifest.json`: generation seed and calibration-recovery errors.
"""
    (output_case / "README.md").write_text(text, encoding="utf-8")


def generate_synthetic_case(
    source_case: Path,
    output_case: Path,
    *,
    seed: int,
    particles_per_frame: int,
) -> SyntheticCaseSummary:
    """Generate the synthetic cavity-like case on disk."""
    rng = np.random.default_rng(seed)
    prepare_output_case(source_case, output_case)
    write_case_readme(output_case)

    control = ControlPar(4).from_file(source_case / "parameters/ptv.par")
    vpar = read_volume_par(source_case / "parameters/criteria.par")
    orient_par = OrientPar().from_file(source_case / "parameters/orient.par")
    sortgrid_eps = read_sortgrid_par(source_case / "parameters/sortgrid.par")
    truth_cals = [
        read_calibration(
            source_case / f"cal/cam{camera_index}.tif.ori",
            source_case / f"cal/cam{camera_index}.tif.addpar",
        )
        for camera_index in range(1, 5)
    ]

    calibration_body = build_calibration_body(vpar, truth_cals, control)
    write_calibration_body_points(
        calibration_body,
        output_case / "ground_truth/calibration_body_points.txt",
    )
    shutil.copy2(
        output_case / "ground_truth/calibration_body_points.txt",
        output_case / "cal/calblock.txt",
    )

    recovered_cals, position_errors, angle_errors = recover_calibrations_from_body(
        truth_cals,
        calibration_body,
        output_case / "calibration_targets",
        output_case / "cal",
        output_case / "ground_truth/cal",
        control,
        orient_par,
        sortgrid_eps,
        rng,
    )

    write_sequence_file(output_case / "parameters/sequence.par", FRAME_NUMBERS)

    for frame_number in FRAME_NUMBERS:
        frame_points = generate_particle_cloud(
            rng,
            vpar,
            truth_cals,
            control,
            particles_per_frame,
        )
        np.savetxt(
            output_case / "ground_truth/particles" / f"frame_{frame_number}.txt",
            frame_points,
            fmt="%.6f",
            header="x y z",
            comments="",
        )

        per_camera_targets, cor_buf, path_buf = build_frame_targets_and_paths(
            frame_points,
            recovered_cals,
            control,
            rng,
        )
        for camera_index, targets in enumerate(per_camera_targets, start=1):
            write_targets(
                targets,
                len(targets),
                str(output_case / "img_orig" / f"cam{camera_index}.%05d"),
                frame_number,
            )

        write_path_frame(
            cor_buf,
            path_buf,
            len(path_buf),
            str(output_case / "res_orig" / "rt_is"),
            str(output_case / "res_orig" / "ptv_is"),
            str(output_case / "res_orig" / "added"),
            frame_number,
        )

    manifest = SyntheticCaseSummary(
        seed=seed,
        num_calibration_points=int(calibration_body.shape[0]),
        num_frames=len(FRAME_NUMBERS),
        particles_per_frame=particles_per_frame,
        calibration_position_errors=position_errors,
        calibration_angle_errors=angle_errors,
    )
    (output_case / "ground_truth/manifest.json").write_text(
        json.dumps(manifest.__dict__, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for synthetic-case generation."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic bundle-adjustment case modeled after test_cavity.",
    )
    parser.add_argument(
        "--source-case",
        type=Path,
        default=DEFAULT_SOURCE_CASE,
        help="Empirical case used as the geometric template.",
    )
    parser.add_argument(
        "--output-case",
        type=Path,
        default=DEFAULT_OUTPUT_CASE,
        help="Destination case folder to populate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic particle generation.",
    )
    parser.add_argument(
        "--particles-per-frame",
        type=int,
        default=96,
        help="Number of fully observed particles to synthesize per frame.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Generate the synthetic case and print a compact summary."""
    args = parse_args(argv)
    summary = generate_synthetic_case(
        args.source_case.resolve(),
        args.output_case.resolve(),
        seed=args.seed,
        particles_per_frame=args.particles_per_frame,
    )
    print(f"Wrote synthetic case to {args.output_case.resolve()}")
    print(f"Calibration points: {summary.num_calibration_points}")
    print(f"Frames: {summary.num_frames}")
    print(f"Particles per frame: {summary.particles_per_frame}")
    print(
        "Calibration recovery position errors: "
        + ", ".join(f"{value:.6f}" for value in summary.calibration_position_errors)
    )
    print(
        "Calibration recovery angle errors: "
        + ", ".join(f"{value:.6f}" for value in summary.calibration_angle_errors)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
