from pathlib import Path

import numpy as np

from openptv_python.calibration import read_calibration
from openptv_python.demo_bundle_adjustment import load_case_observations
from openptv_python.generate_synthetic_cavity_case import (
    DEFAULT_OUTPUT_CASE,
    project_pixels,
)
from openptv_python.orientation import multi_camera_bundle_adjustment, reprojection_rms
from openptv_python.parameters import ControlPar, OrientPar, SequencePar


def test_synthetic_cavity_case_exists_and_is_coherent():
    case_dir = Path(DEFAULT_OUTPUT_CASE)
    assert case_dir.exists()
    assert (case_dir / "ground_truth/manifest.json").exists()
    assert (case_dir / "ground_truth/calibration_body_points.txt").exists()

    truth_cals = [
        read_calibration(
            case_dir / f"ground_truth/cal/cam{camera_index}.tif.ori",
            case_dir / f"ground_truth/cal/cam{camera_index}.tif.addpar",
        )
        for camera_index in range(1, 5)
    ]
    working_cals = [
        read_calibration(
            case_dir / f"cal/cam{camera_index}.tif.ori",
            case_dir / f"cal/cam{camera_index}.tif.addpar",
        )
        for camera_index in range(1, 5)
    ]

    position_errors = [
        np.linalg.norm(working.get_pos() - truth.get_pos())
        for working, truth in zip(working_cals, truth_cals)
    ]
    angle_errors = [
        np.linalg.norm(working.get_angles() - truth.get_angles())
        for working, truth in zip(working_cals, truth_cals)
    ]
    assert max(position_errors) < 0.2
    assert max(angle_errors) < 0.01

    control, observed_pixels, point_init = load_case_observations(
        case_dir,
        4,
        max_frames=1,
        max_points_per_frame=16,
    )
    assert observed_pixels.shape == (16, 4, 2)
    rms = reprojection_rms(observed_pixels, point_init, working_cals, control)
    assert rms < 0.5


def load_truth_particles(
    case_dir: Path,
    *,
    max_frames: int | None,
    max_points_per_frame: int | None,
) -> np.ndarray:
    """Load the exact synthetic particle coordinates in sequence order."""
    seq = SequencePar.from_file(case_dir / "parameters/sequence.par", 4)
    frames = list(range(seq.first, seq.last + 1))
    if max_frames is not None:
        frames = frames[:max_frames]

    truth_batches = []
    for frame in frames:
        frame_points = np.loadtxt(
            case_dir / "ground_truth/particles" / f"frame_{frame}.txt",
            skiprows=1,
        )
        if frame_points.ndim == 1:
            frame_points = frame_points.reshape(1, 3)
        if max_points_per_frame is not None:
            frame_points = frame_points[:max_points_per_frame]
        truth_batches.append(frame_points)

    return np.concatenate(truth_batches, axis=0)


def perturb_free_cameras(truth_cals):
    """Perturb only the cameras that remain free during BA."""
    deltas = {
        2: (np.array([0.9, -0.6, 0.4]), np.array([0.008, -0.006, 0.004])),
        3: (np.array([-0.7, 0.5, -0.3]), np.array([-0.007, 0.005, -0.004])),
    }
    perturbed = []
    for camera_index, cal in enumerate(truth_cals):
        trial = cal.__class__(
            ext_par=cal.ext_par.copy(),
            int_par=cal.int_par.copy(),
            glass_par=cal.glass_par.copy(),
            added_par=cal.added_par.copy(),
            mmlut=cal.mmlut,
            mmlut_data=cal.mmlut_data,
        )
        if camera_index in deltas:
            pos_delta, angle_delta = deltas[camera_index]
            trial.set_pos(trial.get_pos() + pos_delta)
            trial.set_angles(trial.get_angles() + angle_delta)
        perturbed.append(trial)
    return perturbed


def perturb_intrinsics(truth_cals):
    """Perturb a small subset of intrinsic parameters while keeping poses fixed."""
    perturbed = []
    for cal in truth_cals:
        trial = cal.__class__(
            ext_par=cal.ext_par.copy(),
            int_par=cal.int_par.copy(),
            glass_par=cal.glass_par.copy(),
            added_par=cal.added_par.copy(),
            mmlut=cal.mmlut,
            mmlut_data=cal.mmlut_data,
        )
        trial.added_par[0] += 2e-5
        trial.added_par[3] += 8e-5
        trial.added_par[4] -= 6e-5
        perturbed.append(trial)
    return perturbed


def test_synthetic_case_bundle_adjustment_recovers_ground_truth_from_controlled_perturbation():
    case_dir = Path(DEFAULT_OUTPUT_CASE)
    control = ControlPar(4).from_file(case_dir / "parameters/ptv.par")
    truth_cals = [
        read_calibration(
            case_dir / f"ground_truth/cal/cam{camera_index}.tif.ori",
            case_dir / f"ground_truth/cal/cam{camera_index}.tif.addpar",
        )
        for camera_index in range(1, 5)
    ]
    truth_points = load_truth_particles(
        case_dir,
        max_frames=1,
        max_points_per_frame=24,
    )
    observed_pixels = project_pixels(truth_points, truth_cals, control)

    start_cals = perturb_free_cameras(truth_cals)
    point_init = truth_points + np.array([0.35, -0.25, 0.18])

    initial_camera_errors = [
        np.linalg.norm(
            start_cals[camera_index].get_pos() - truth_cals[camera_index].get_pos()
        )
        for camera_index in (2, 3)
    ]
    initial_angle_errors = [
        np.linalg.norm(
            start_cals[camera_index].get_angles()
            - truth_cals[camera_index].get_angles()
        )
        for camera_index in (2, 3)
    ]
    initial_point_error = float(
        np.mean(np.linalg.norm(point_init - truth_points, axis=1))
    )

    refined_cals, refined_points, result = multi_camera_bundle_adjustment(
        observed_pixels,
        start_cals,
        control,
        OrientPar(),
        point_init=point_init,
        fixed_camera_indices=[0, 1],
        loss="linear",
        method="lm",
        prior_sigmas={
            "x0": 1.0,
            "y0": 1.0,
            "z0": 1.0,
            "omega": 0.01,
            "phi": 0.01,
            "kappa": 0.01,
        },
        max_nfev=100,
    )

    final_camera_errors = [
        np.linalg.norm(
            refined_cals[camera_index].get_pos() - truth_cals[camera_index].get_pos()
        )
        for camera_index in (2, 3)
    ]
    final_angle_errors = [
        np.linalg.norm(
            refined_cals[camera_index].get_angles()
            - truth_cals[camera_index].get_angles()
        )
        for camera_index in (2, 3)
    ]
    final_point_error = float(
        np.mean(np.linalg.norm(refined_points - truth_points, axis=1))
    )

    assert bool(result.success), result.message
    assert max(final_camera_errors) < max(initial_camera_errors) * 0.01
    assert max(final_camera_errors) < 0.01
    assert max(final_angle_errors) < max(initial_angle_errors) * 0.01
    assert max(final_angle_errors) < 1e-4
    assert final_point_error < initial_point_error * 1e-3
    assert final_point_error < 1e-3
    assert result["final_reprojection_rms"] < 1e-3


def test_synthetic_case_intrinsics_only_recovers_ground_truth_from_controlled_perturbation():
    case_dir = Path(DEFAULT_OUTPUT_CASE)
    control = ControlPar(4).from_file(case_dir / "parameters/ptv.par")
    truth_cals = [
        read_calibration(
            case_dir / f"ground_truth/cal/cam{camera_index}.tif.ori",
            case_dir / f"ground_truth/cal/cam{camera_index}.tif.addpar",
        )
        for camera_index in range(1, 5)
    ]
    truth_points = load_truth_particles(
        case_dir,
        max_frames=1,
        max_points_per_frame=24,
    )
    observed_pixels = project_pixels(truth_points, truth_cals, control)
    start_cals = perturb_intrinsics(truth_cals)

    intrinsics = OrientPar()
    intrinsics.k1flag = 1
    intrinsics.p1flag = 1
    intrinsics.p2flag = 1

    refined_cals, refined_points, result = multi_camera_bundle_adjustment(
        observed_pixels,
        start_cals,
        control,
        intrinsics,
        point_init=truth_points.copy(),
        fixed_camera_indices=[0, 1, 2, 3],
        optimize_extrinsics=False,
        optimize_points=False,
        loss="linear",
        method="trf",
        prior_sigmas={
            "k1": 5e-5,
            "p1": 1e-4,
            "p2": 1e-4,
        },
        parameter_bounds={
            "k1": (-5e-5, 5e-5),
            "p1": (-2e-4, 2e-4),
            "p2": (-2e-4, 2e-4),
        },
        max_nfev=40,
    )

    k1_errors = [
        abs(refined.added_par[0] - truth.added_par[0])
        for refined, truth in zip(refined_cals, truth_cals)
    ]
    p1_errors = [
        abs(refined.added_par[3] - truth.added_par[3])
        for refined, truth in zip(refined_cals, truth_cals)
    ]
    p2_errors = [
        abs(refined.added_par[4] - truth.added_par[4])
        for refined, truth in zip(refined_cals, truth_cals)
    ]

    assert bool(result.success), result.message
    assert np.max(np.linalg.norm(refined_points - truth_points, axis=1)) == 0.0
    for refined, start in zip(refined_cals, start_cals):
        np.testing.assert_allclose(refined.get_pos(), start.get_pos(), atol=1e-12)
        np.testing.assert_allclose(refined.get_angles(), start.get_angles(), atol=1e-12)
    assert max(k1_errors) < 1.5e-5
    assert max(p1_errors) < 1.5e-5
    assert max(p2_errors) < 3.0e-5
    assert result["final_reprojection_rms"] < result["initial_reprojection_rms"] * 0.2
