import unittest
from pathlib import Path

import numpy as np

from openptv_python.calibration import (
    Calibration,
)
from openptv_python.imgcoord import flat_image_coordinates, image_coordinates
from openptv_python.orientation import (
    match_detection_to_ref,
    point_positions,
    weighted_dumbbell_precision,
)
from openptv_python.parameters import ControlPar, MultimediaPar, OrientPar, VolumePar
from openptv_python.tracking_frame_buf import Target
from openptv_python.trafo import arr_metric_to_pixel


class TestOrientation(unittest.TestCase):
    """Test the orientation module."""

    def setUp(self):
        """Set up the test."""
        filepath = Path("tests") / "testing_folder"
        self.input_ori_file_name = filepath / "calibration/cam1.tif.ori"
        self.input_add_file_name = filepath / "calibration/cam2.tif.addpar"
        self.control_file_name = filepath / "control_parameters/control.par"
        self.volume_file_name = filepath / "corresp/criteria.par"
        self.orient_par_file_name = filepath / "corresp/orient.par"

        self.control = ControlPar(4).from_file(self.control_file_name)
        self.calibration = Calibration().from_file(
            self.input_ori_file_name, self.input_add_file_name
        )
        self.vpar = VolumePar().from_file(self.volume_file_name)
        self.orient_par = OrientPar().from_file(self.orient_par_file_name)

    def test_match_detection_to_ref(self):
        """Match detection to reference (sortgrid)."""
        xyz_input = np.array(
            [
                (10, 10, 10),
                (200, 200, 200),
                (600, 800, 100),
                (20, 10, 2000),
                (30, 30, 30),
            ],
            dtype=float,
        )
        coords_count = len(xyz_input)

        xy_img_pts_metric = image_coordinates(
            xyz_input, self.calibration, self.control.mm
        )
        xy_img_pts_pixel = arr_metric_to_pixel(xy_img_pts_metric, self.control)

        # convert to TargetArray object
        # targets = TargetArray(coords_count)
        targets = [Target() for _ in range(coords_count)]

        for i in range(coords_count):
            targets[i].pnr = i
            targets[i].x = xy_img_pts_pixel[i][0]
            targets[i].y = xy_img_pts_pixel[i][1]

            # set_pos((xy_img_pts_pixel[i][0], xy_img_pts_pixel[i][1]))

        # create randomized target array
        indices = np.arange(coords_count)
        shuffled_indices = np.arange(coords_count)

        while np.all(indices == shuffled_indices):
            np.random.shuffle(shuffled_indices)

        # rand_targ_array = TargetArray(coords_count)
        rand_targ_array = [Target() for _ in range(coords_count)]
        for i in range(coords_count):
            rand_targ_array[shuffled_indices[i]].x = targets[i].x
            rand_targ_array[shuffled_indices[i]].y = targets[i].y
            rand_targ_array[shuffled_indices[i]].pnr = targets[i].pnr

        # match detection to reference
        matched_target_array = match_detection_to_ref(
            cal=self.calibration,
            ref_pts=xyz_input,
            img_pts=rand_targ_array,
            cparam=self.control,
        )

        # assert target array is as before
        for i in range(coords_count):
            if matched_target_array[i] != targets[i]:
                self.fail()

        # pass ref_pts and img_pts with non-equal lengths
        # with self.assertRaises(ValueError):
        #     match_detection_to_ref(
        #         cal=self.calibration,
        #         ref_pts=xyz_input,
        #         img_pts=TargetArray(coords_count - 1),
        #         cparam=self.control,
        #     )

    def test_point_positions(self):
        """Point positions."""
        # prepare MultimediaParams
        mult_params = MultimediaPar()

        mult_params.set_n1(1.0)
        mult_params.set_layers([1.0], [1.0])
        mult_params.set_n3(1.0)

        # 3d point
        points = np.atleast_2d(np.array([17, 42, 0], dtype=float))

        num_cams = 4
        ori_tmpl = "tests/testing_folder/calibration/sym_cam{cam_num}.tif.ori"
        add_file = Path("tests/testing_folder/calibration/cam1.tif.addpar")
        calibs = []
        targs_plain = []
        targs_jigged = []

        jigg_amp = 0.5

        # read calibration for each camera from files
        for cam in range(num_cams):
            ori_name = Path(ori_tmpl.format(cam_num=cam + 1))
            new_cal = Calibration().from_file(ori_file=ori_name, add_file=add_file)
            calibs.append(new_cal)

        for cam_num, cam_cal in enumerate(calibs):
            new_plain_targ = image_coordinates(points, cam_cal, mult_params)
            targs_plain.append(new_plain_targ)

            if (cam_num % 2) == 0:
                jigged_points = points - np.r_[0, jigg_amp, 0]
            else:
                jigged_points = points + np.r_[0, jigg_amp, 0]

            new_jigged_targs = image_coordinates(jigged_points, cam_cal, mult_params)
            targs_jigged.append(new_jigged_targs)

        targs_plain = np.array(targs_plain).transpose(1, 0, 2)
        targs_jigged = np.array(targs_jigged).transpose(1, 0, 2)
        skew_dist_plain = point_positions(targs_plain, mult_params, calibs, self.vpar)
        skew_dist_jigged = point_positions(targs_jigged, mult_params, calibs, self.vpar)

        if np.any(skew_dist_plain[1] > 1e-10):
            self.fail(
                ("skew distance of target#{targ_num} " + "is more than allowed").format(
                    targ_num=np.nonzero(skew_dist_plain[1] > 1e-10)[0][0]
                )
            )

        if np.any(np.linalg.norm(points - skew_dist_plain[0], axis=1) > 1e-6):
            self.fail("Rays converge on wrong position.")

        if np.any(skew_dist_jigged[1] > 0.7):
            self.fail(
                ("skew distance of target#{targ_num} " + "is more than allowed").format(
                    targ_num=np.nonzero(skew_dist_jigged[1] > 1e-10)[0][0]
                )
            )
        if np.any(np.linalg.norm(points - skew_dist_jigged[0], axis=1) > 0.1):
            self.fail("Rays converge on wrong position after jigging.")

    def test_single_camera_point_positions(self):
        """Point positions for a single camera case."""
        num_cams = 1
        # prepare MultimediaParams
        cpar_file = Path("tests/testing_folder/single_cam/parameters/ptv.par")
        vpar_file = Path("tests/testing_folder/single_cam/parameters/criteria.par")
        cpar = ControlPar(num_cams).from_file(cpar_file)
        # mm_params = cpar.get_multimedia_params()

        vpar = VolumePar().from_file(vpar_file)

        ori_name = Path("tests/testing_folder/single_cam/calibration/cam_1.tif.ori")
        add_name = Path("tests/testing_folder/single_cam/calibration/cam_1.tif.addpar")
        calibs = []

        # read calibration for each camera from files
        new_cal = Calibration().from_file(ori_file=ori_name, add_file=add_name)
        calibs.append(new_cal)

        # 3d point
        points = np.array([[1, 1, 0], [-1, -1, 0]], dtype=float)

        targs_plain = []
        targs_jigged = []

        jigg_amp = 0.4

        new_plain_targ = image_coordinates(points, calibs[0], cpar.mm)
        targs_plain.append(new_plain_targ)

        # print(f"new_plain_targ: {new_plain_targ}")

        jigged_points = points - np.r_[0, jigg_amp, 0]

        new_jigged_targs = image_coordinates(jigged_points, calibs[0], cpar.mm)
        targs_jigged.append(new_jigged_targs)

        # print(f"new_jigged_targs: {new_jigged_targs}")

        targs_plain = np.array(targs_plain).transpose(1, 0, 2)

        targs_jigged = np.array(targs_jigged).transpose(1, 0, 2)

        # print(f"targs_plain: {targs_plain}")
        # print(f"targs_jigged: {targs_jigged}")

        skew_dist_plain = point_positions(targs_plain, cpar.mm, calibs, vpar)
        skew_dist_jigged = point_positions(targs_jigged, cpar.mm, calibs, vpar)

        # print(f"skew_dist_plain: {skew_dist_plain}")
        # print(f"skew_dist_jigged: {skew_dist_jigged}")

        if np.sum(np.linalg.norm(points - skew_dist_plain[0], axis=1)) > 1e-6:
            self.fail("Rays converge on wrong position.")

        if np.sum(np.linalg.norm(jigged_points - skew_dist_jigged[0], axis=1)) > 1e-6:
            self.fail("Rays converge on wrong position after jigging.")

    def test_dumbbell(self):
        """Point positions for a dumbbell case."""
        # prepare MultimediaParams
        mult_params = self.control.get_multimedia_params()
        mult_params.set_n1(1.0)
        mult_params.set_layers([1.0], [1.0])
        mult_params.set_n3(1.0)

        # 3d point
        points = np.array([[17.5, 42, 0], [-17.5, 42, 0]], dtype=float)

        num_cams = 4
        ori_tmpl = "tests/testing_folder/dumbbell/cam{cam_num}.tif.ori"
        add_file = Path("tests/testing_folder/calibration/cam1.tif.addpar")
        calibs = []
        targs_plain = []

        # read calibration for each camera from files
        for cam in range(num_cams):
            ori_name = Path(ori_tmpl.format(cam_num=cam + 1))
            new_cal = Calibration().from_file(ori_file=ori_name, add_file=add_file)
            calibs.append(new_cal)

        for cam_cal in calibs:
            new_plain_targ = flat_image_coordinates(points, cam_cal, mult_params)
            targs_plain.append(new_plain_targ)

        targs_plain = np.array(targs_plain).transpose(1, 0, 2)

        # The cameras are not actually fully calibrated, so the result is not
        # an exact 0. The test is that changing the expected distance changes
        # the measure.

        tf = weighted_dumbbell_precision(targs_plain, mult_params, calibs, 35.0, 0.0)
        self.assertAlmostEqual(tf, 0.0, 5)  # just a regression test

        # As we check the db length, the measure increases...
        tf_len = weighted_dumbbell_precision(
            targs_plain, mult_params, calibs, 35.0, 1.0
        )
        self.assertTrue(tf_len > tf)

        # ...but not as much as when giving the wrong length.
        tf_too_long = weighted_dumbbell_precision(
            targs_plain, mult_params, calibs, 25.0, 1.0
        )
        self.assertTrue(tf_too_long > tf_len > tf)


if __name__ == "__main__":
    unittest.main()
