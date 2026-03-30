import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from openptv_python.calibration import read_calibration
from openptv_python.demo_bundle_adjustment import (
    ExperimentResult,
    all_fixed_camera_pairs,
    build_experiment_start_calibrations,
    build_known_point_constraints,
    calibration_body_projection_drift,
    default_experiments,
    format_quadruplet_sensitivity,
    load_calibrations,
    load_reference_geometry_points,
    normalize_staged_release_order,
    perturb_calibrations,
    run_experiment,
    summarize_epipolar_consistency,
    summarize_fixed_camera_diagnostics,
    summarize_quadruplet_sensitivity,
)
from openptv_python.imgcoord import image_coordinates
from openptv_python.parameters import ControlPar, read_volume_par
from openptv_python.tracking_frame_buf import read_path_frame, read_targets
from openptv_python.trafo import arr_metric_to_pixel


class TestBundleAdjustmentDemo(unittest.TestCase):
    def test_build_known_point_constraints_returns_copied_subset(self):
        point_init = np.arange(30, dtype=float).reshape(10, 3)

        known_points = build_known_point_constraints(point_init, 4)

        self.assertEqual(sorted(known_points), [0, 3, 6, 9])
        np.testing.assert_allclose(known_points[6], point_init[6])

        point_init[6, 0] = -999.0
        self.assertNotEqual(known_points[6][0], point_init[6, 0])

    def test_default_experiments_adds_known_point_presets(self):
        known_points = {0: np.array([0.0, 0.0, 0.0]), 5: np.array([1.0, 2.0, 3.0])}

        experiments = default_experiments(
            num_cams=4,
            known_points=known_points,
            known_point_sigmas=0.25,
        )

        names = [spec.name for spec in experiments]
        self.assertIn("intrinsics_only", names)
        self.assertIn("intrinsics_first_guarded_stagewise_release", names)
        self.assertIn("intrinsics_first_alternating_stagewise_release", names)
        self.assertIn("pose_trf_known_points", names)
        self.assertIn("guarded_two_step_known_points", names)
        self.assertIn("guarded_stagewise_release", names)
        self.assertIn("guarded_stagewise_release_known_points", names)

        intrinsics_only = next(
            spec for spec in experiments if spec.name == "intrinsics_only"
        )
        self.assertFalse(intrinsics_only.ba_kwargs["optimize_extrinsics"])
        self.assertFalse(intrinsics_only.ba_kwargs["optimize_points"])
        self.assertEqual(
            intrinsics_only.ba_kwargs["fixed_camera_indices"],
            [0, 1, 2, 3],
        )
        self.assertTrue(intrinsics_only.ba_kwargs["use_reference_cals"])
        self.assertTrue(intrinsics_only.ba_kwargs["perturb_intrinsics_only"])
        self.assertEqual(intrinsics_only.ba_kwargs["intrinsic_perturbation_scale"], 1.0)

        pose_spec = next(
            spec for spec in experiments if spec.name == "pose_trf_known_points"
        )
        guarded_spec = next(
            spec for spec in experiments if spec.name == "guarded_two_step_known_points"
        )
        intrinsics_first_spec = next(
            spec
            for spec in experiments
            if spec.name == "intrinsics_first_guarded_stagewise_release"
        )
        alternating_spec = next(
            spec
            for spec in experiments
            if spec.name == "intrinsics_first_alternating_stagewise_release"
        )
        staged_spec = next(
            spec for spec in experiments if spec.name == "guarded_stagewise_release"
        )
        staged_known_spec = next(
            spec
            for spec in experiments
            if spec.name == "guarded_stagewise_release_known_points"
        )
        self.assertIs(pose_spec.ba_kwargs["known_points"], known_points)
        self.assertEqual(pose_spec.ba_kwargs["known_point_sigmas"], 0.25)
        self.assertIs(guarded_spec.ba_kwargs["known_points"], known_points)
        self.assertEqual(guarded_spec.ba_kwargs["known_point_sigmas"], 0.25)
        self.assertEqual(staged_spec.ba_kwargs["fixed_camera_indices"], [1, 2, 3])
        self.assertEqual(
            staged_spec.ba_kwargs["pose_release_camera_order"], [0, 1, 2, 3]
        )
        self.assertEqual(staged_spec.ba_kwargs["pose_stage_ray_slack"], 0.0)
        self.assertEqual(len(staged_spec.ba_kwargs["pose_stage_configs"]), 3)
        self.assertFalse(
            staged_spec.ba_kwargs["pose_stage_configs"][0]["optimize_points"]
        )
        self.assertTrue(
            staged_spec.ba_kwargs["pose_stage_configs"][1]["optimize_points"]
        )
        self.assertTrue(
            staged_spec.ba_kwargs["pose_stage_configs"][2]["optimize_points"]
        )
        self.assertEqual(intrinsics_first_spec.mode, "intrinsics_then_guarded")
        self.assertFalse(
            intrinsics_first_spec.ba_kwargs["warmstart_optimize_extrinsics"]
        )
        self.assertFalse(intrinsics_first_spec.ba_kwargs["warmstart_optimize_points"])
        self.assertNotIn(
            "first_release_geometry_slack", intrinsics_first_spec.ba_kwargs
        )
        self.assertNotIn(
            "first_release_correspondence_slack",
            intrinsics_first_spec.ba_kwargs,
        )
        self.assertEqual(
            intrinsics_first_spec.ba_kwargs["pose_release_camera_order"],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            len(intrinsics_first_spec.ba_kwargs["pose_stage_configs"]),
            2,
        )
        self.assertFalse(
            intrinsics_first_spec.ba_kwargs["pose_stage_configs"][0]["optimize_points"]
        )
        self.assertTrue(
            intrinsics_first_spec.ba_kwargs["pose_stage_configs"][1]["optimize_points"]
        )
        self.assertEqual(alternating_spec.mode, "alternating")
        self.assertEqual(
            len(alternating_spec.ba_kwargs["pose_block_configs"]),
            6,
        )
        self.assertTrue(
            alternating_spec.ba_kwargs["pose_block_configs"][0]["optimize_points"]
        )
        self.assertEqual(
            [
                block["name"]
                for block in alternating_spec.ba_kwargs["pose_block_configs"]
            ],
            [
                "points_only",
                "omega_only",
                "phi_only",
                "kappa_only",
                "translation_only",
                "joint_pose_points",
            ],
        )
        self.assertEqual(
            alternating_spec.ba_kwargs["first_release_geometry_slack"], 0.35
        )
        self.assertEqual(
            alternating_spec.ba_kwargs["first_release_correspondence_slack"],
            0.02,
        )
        self.assertIs(staged_known_spec.ba_kwargs["known_points"], known_points)
        self.assertEqual(staged_known_spec.ba_kwargs["known_point_sigmas"], 0.25)
        self.assertTrue(
            all(
                stage_config["optimize_points"]
                for stage_config in guarded_spec.ba_kwargs["pose_stage_configs"]
            )
        )
        self.assertTrue(
            all(
                stage_config["optimize_points"]
                for stage_config in staged_known_spec.ba_kwargs["pose_stage_configs"]
            )
        )
        self.assertEqual(guarded_spec.ba_kwargs["geometry_guard_mode"], "off")
        self.assertIsNone(guarded_spec.ba_kwargs["geometry_guard_threshold"])
        self.assertEqual(guarded_spec.ba_kwargs["correspondence_guard_mode"], "off")
        self.assertIsNone(guarded_spec.ba_kwargs["correspondence_guard_threshold"])
        self.assertIsNone(guarded_spec.ba_kwargs["correspondence_guard_reference_rate"])

    def test_run_experiment_intrinsics_then_guarded_executes_warmstart_first(self):
        control = ControlPar(4).from_file(
            Path("tests/testing_folder/control_parameters/control.par")
        )
        add_file = Path("tests/testing_folder/calibration/cam1.tif.addpar")
        reference_cals = [
            read_calibration(
                Path(f"tests/testing_folder/calibration/sym_cam{cam_num}.tif.ori"),
                add_file,
            )
            for cam_num in range(1, 5)
        ]
        start_cals = perturb_calibrations(reference_cals, 1.0)
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
            ],
            dtype=float,
        )
        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(reference_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, control.mm),
                control,
            )

        spec = next(
            experiment
            for experiment in default_experiments(perturbation_scale=1.0)
            if experiment.name == "intrinsics_first_guarded_stagewise_release"
        )

        with (
            patch(
                "openptv_python.demo_bundle_adjustment.multi_camera_bundle_adjustment",
                return_value=(
                    reference_cals,
                    points.copy(),
                    {"success": True, "final_reprojection_rms": 3.8, "message": "warm"},
                ),
            ) as warmstart,
            patch(
                "openptv_python.demo_bundle_adjustment.guarded_two_step_bundle_adjustment",
                return_value=(
                    reference_cals,
                    points.copy(),
                    {
                        "accepted_stage": "intrinsics",
                        "final_reprojection_rms": 3.7,
                        "final_mean_ray_convergence": 0.2,
                    },
                ),
            ) as guarded,
        ):
            result = run_experiment(
                spec,
                observed_pixels=observed_pixels,
                point_init=points,
                control=control,
                start_cals=start_cals,
                reference_cals=reference_cals,
                reference_geometry_points=None,
                tracking_data=None,
                geometry_export_threshold=None,
                correspondence_export_threshold=None,
                source_case_dir=Path("tests/testing_folder"),
                output_dir=None,
            )

        self.assertEqual(warmstart.call_count, 1)
        self.assertEqual(guarded.call_count, 1)
        self.assertFalse(warmstart.call_args.kwargs["optimize_extrinsics"])
        self.assertFalse(warmstart.call_args.kwargs["optimize_points"])
        np.testing.assert_allclose(guarded.call_args.kwargs["point_init"], points)
        self.assertIn("warmstart_rms=3.800000", result.notes)
        self.assertIn("accepted_stage=intrinsics", result.notes)

    def test_run_experiment_guarded_mode_does_not_forward_alternating_slack(self):
        control = ControlPar(4).from_file(
            Path("tests/testing_folder/control_parameters/control.par")
        )
        add_file = Path("tests/testing_folder/calibration/cam1.tif.addpar")
        reference_cals = [
            read_calibration(
                Path(f"tests/testing_folder/calibration/sym_cam{cam_num}.tif.ori"),
                add_file,
            )
            for cam_num in range(1, 5)
        ]
        start_cals = perturb_calibrations(reference_cals, 1.0)
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
            ],
            dtype=float,
        )
        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(reference_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, control.mm),
                control,
            )

        spec = next(
            experiment
            for experiment in default_experiments(perturbation_scale=1.0)
            if experiment.name == "guarded_stagewise_release"
        )

        with patch(
            "openptv_python.demo_bundle_adjustment.guarded_two_step_bundle_adjustment",
            return_value=(
                reference_cals,
                points.copy(),
                {
                    "accepted_stage": "intrinsics",
                    "final_reprojection_rms": 3.7,
                    "final_mean_ray_convergence": 0.2,
                },
            ),
        ) as guarded:
            run_experiment(
                spec,
                observed_pixels=observed_pixels,
                point_init=points,
                control=control,
                start_cals=start_cals,
                reference_cals=reference_cals,
                reference_geometry_points=None,
                tracking_data=None,
                geometry_export_threshold=None,
                correspondence_export_threshold=None,
                source_case_dir=Path("tests/testing_folder"),
                output_dir=None,
            )

        self.assertEqual(guarded.call_count, 1)
        self.assertNotIn("first_release_geometry_slack", guarded.call_args.kwargs)
        self.assertNotIn(
            "first_release_correspondence_slack",
            guarded.call_args.kwargs,
        )

    def test_run_experiment_alternating_mode_uses_alternating_solver(self):
        control = ControlPar(4).from_file(
            Path("tests/testing_folder/control_parameters/control.par")
        )
        add_file = Path("tests/testing_folder/calibration/cam1.tif.addpar")
        reference_cals = [
            read_calibration(
                Path(f"tests/testing_folder/calibration/sym_cam{cam_num}.tif.ori"),
                add_file,
            )
            for cam_num in range(1, 5)
        ]
        start_cals = perturb_calibrations(reference_cals, 1.0)
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
            ],
            dtype=float,
        )
        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(reference_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, control.mm),
                control,
            )

        spec = next(
            experiment
            for experiment in default_experiments(perturbation_scale=1.0)
            if experiment.name == "intrinsics_first_alternating_stagewise_release"
        )

        with patch(
            "openptv_python.demo_bundle_adjustment.alternating_bundle_adjustment",
            return_value=(
                reference_cals,
                points.copy(),
                {
                    "warmstart_ok": True,
                    "accepted_stage": "warmstart",
                    "final_reprojection_rms": 3.9,
                    "final_mean_ray_convergence": 0.2,
                },
            ),
        ) as alternating:
            result = run_experiment(
                spec,
                observed_pixels=observed_pixels,
                point_init=points,
                control=control,
                start_cals=start_cals,
                reference_cals=reference_cals,
                reference_geometry_points=None,
                tracking_data=None,
                geometry_export_threshold=None,
                correspondence_export_threshold=None,
                source_case_dir=Path("tests/testing_folder"),
                output_dir=None,
            )

        self.assertEqual(alternating.call_count, 1)
        self.assertIn("warmstart_ok=True", result.notes)
        self.assertIn("accepted_stage=warmstart", result.notes)

    def test_default_experiments_accepts_geometry_guard_configuration(self):
        experiments = default_experiments(
            num_cams=4,
            perturbation_scale=1.0,
            staged_release_order=[2, 0, 1, 3],
            pose_stage_ray_slack=0.002,
            geometry_guard_mode="hard",
            geometry_guard_threshold=2.5,
            correspondence_guard_mode="hard",
            correspondence_guard_threshold=0.18,
            correspondence_guard_reference_rate=0.15625,
        )

        guarded_spec = next(
            spec for spec in experiments if spec.name == "guarded_two_step"
        )
        staged_spec = next(
            spec for spec in experiments if spec.name == "guarded_stagewise_release"
        )

        self.assertEqual(guarded_spec.ba_kwargs["geometry_guard_mode"], "hard")
        self.assertEqual(guarded_spec.ba_kwargs["geometry_guard_threshold"], 2.5)
        self.assertEqual(guarded_spec.ba_kwargs["correspondence_guard_mode"], "hard")
        self.assertEqual(guarded_spec.ba_kwargs["correspondence_guard_threshold"], 0.18)
        self.assertEqual(
            guarded_spec.ba_kwargs["correspondence_guard_reference_rate"],
            0.15625,
        )
        self.assertEqual(
            staged_spec.ba_kwargs["pose_release_camera_order"], [2, 0, 1, 3]
        )
        self.assertEqual(staged_spec.ba_kwargs["fixed_camera_indices"], [0, 1, 3])
        self.assertEqual(staged_spec.ba_kwargs["pose_stage_ray_slack"], 0.002)

    def test_normalize_staged_release_order_validates_camera_permutation(self):
        self.assertEqual(normalize_staged_release_order(None, 4), [0, 1, 2, 3])
        self.assertEqual(normalize_staged_release_order([2, 0, 1, 3], 4), [2, 0, 1, 3])

        with self.assertRaises(ValueError):
            normalize_staged_release_order([0, 1, 1, 3], 4)

        with self.assertRaises(ValueError):
            normalize_staged_release_order([0, 1, 2], 4)

    def test_pose_demo_keeps_fixed_cameras_on_reference_geometry(self):
        cavity_dir = Path("tests/testing_folder/test_cavity")
        control = ControlPar(4).from_file(cavity_dir / "parameters/ptv.par")
        true_cals = load_calibrations(cavity_dir, 4)
        start_cals = perturb_calibrations(true_cals, 1.0)
        spec = next(
            experiment
            for experiment in default_experiments(perturbation_scale=1.0)
            if experiment.name == "pose_trf_linear"
        )

        working_cals = build_experiment_start_calibrations(
            spec,
            start_cals=start_cals,
            reference_cals=true_cals,
        )
        geometry_points = load_reference_geometry_points(cavity_dir, 4)
        drift = calibration_body_projection_drift(
            true_cals,
            working_cals,
            control,
            geometry_points,
        )

        self.assertIsNotNone(drift)
        np.testing.assert_allclose(working_cals[0].get_pos(), true_cals[0].get_pos())
        np.testing.assert_allclose(working_cals[1].get_pos(), true_cals[1].get_pos())
        np.testing.assert_allclose(
            working_cals[0].get_angles(),
            true_cals[0].get_angles(),
        )
        np.testing.assert_allclose(
            working_cals[1].get_angles(),
            true_cals[1].get_angles(),
        )
        self.assertLess(drift[0].max_distance, 1e-9)
        self.assertLess(drift[1].max_distance, 1e-9)
        self.assertGreater(drift[2].max_distance, 0.1)

    def test_all_fixed_camera_pairs_returns_all_unique_pairs(self):
        self.assertEqual(
            all_fixed_camera_pairs(4),
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        )

    def test_summarize_fixed_camera_diagnostics_orders_by_fixed_drift_then_rms(self):
        results = [
            ExperimentResult(
                name="run_b",
                description="",
                duration_sec=0.2,
                success=True,
                initial_rms=4.0,
                final_rms=1.3,
                baseline_ray_convergence=2.0,
                final_ray_convergence=1.1,
                notes="b",
                cal_dir=None,
                fixed_camera_indices=(0, 2),
                camera_position_shifts=[0.0, 0.8, 0.0, 0.6],
                camera_angle_shifts=[0.0, 0.02, 0.0, 0.01],
                refined_cals=None,
                refined_points=None,
            ),
            ExperimentResult(
                name="run_a",
                description="",
                duration_sec=0.1,
                success=True,
                initial_rms=4.0,
                final_rms=1.2,
                baseline_ray_convergence=2.0,
                final_ray_convergence=1.0,
                notes="a",
                cal_dir=None,
                fixed_camera_indices=(0, 1),
                camera_position_shifts=[0.0, 0.0, 0.5, 0.4],
                camera_angle_shifts=[0.0, 0.0, 0.01, 0.02],
                refined_cals=None,
                refined_points=None,
            ),
        ]

        diagnostics = summarize_fixed_camera_diagnostics(results)

        self.assertEqual(diagnostics[0].fixed_camera_indices, (0, 1))
        self.assertEqual(diagnostics[1].fixed_camera_indices, (0, 2))
        self.assertEqual(diagnostics[0].fixed_position_shift, 0.0)
        self.assertAlmostEqual(diagnostics[0].mean_free_position_shift, 0.45)

    def test_epipolar_and_quadruplet_diagnostics_detect_perturbation(self):
        cavity_dir = Path("tests/testing_folder/test_cavity")
        control = ControlPar(4).from_file(cavity_dir / "parameters/ptv.par")
        vpar = read_volume_par(cavity_dir / "parameters/criteria.par")
        cals = [
            read_calibration(
                cavity_dir / f"cal/cam{cam_num}.tif.ori",
                cavity_dir / f"cal/cam{cam_num}.tif.addpar",
            )
            for cam_num in range(1, 5)
        ]

        cor_buf, path_buf = read_path_frame(
            str(cavity_dir / "res_orig/rt_is"), "", "", 10001
        )
        targets = [
            read_targets(str(cavity_dir / f"img_orig/cam{cam_num}.%05d"), 10001)
            for cam_num in range(1, 5)
        ]
        subset = [
            pt_num for pt_num, corres in enumerate(cor_buf) if np.all(corres.p >= 0)
        ][:8]
        observed_pixels = np.full((len(subset), 4, 2), np.nan, dtype=float)
        for out_num, pt_num in enumerate(subset):
            for cam in range(4):
                target_index = cor_buf[pt_num].p[cam]
                observed_pixels[out_num, cam, 0] = targets[cam][target_index].x
                observed_pixels[out_num, cam, 1] = targets[cam][target_index].y

        baseline_epipolar = summarize_epipolar_consistency(
            observed_pixels,
            cals,
            control,
            vpar,
            num_curve_points=16,
        )
        baseline_quad = summarize_quadruplet_sensitivity(observed_pixels, cals, control)

        perturbed = observed_pixels.copy()
        perturbed[0, 1, 0] += 25.0
        perturbed[0, 1, 1] -= 15.0
        perturbed_epipolar = summarize_epipolar_consistency(
            perturbed,
            cals,
            control,
            vpar,
            num_curve_points=16,
        )
        perturbed_quad = summarize_quadruplet_sensitivity(perturbed, cals, control)

        self.assertGreater(
            max(item.max_distance for item in perturbed_epipolar),
            max(item.max_distance for item in baseline_epipolar),
        )
        self.assertGreater(perturbed_quad.max_spread, baseline_quad.max_spread)
        self.assertIn(
            "baseline", format_quadruplet_sensitivity(baseline_quad, perturbed_quad)
        )


if __name__ == "__main__":
    unittest.main()
