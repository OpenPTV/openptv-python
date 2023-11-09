#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tests for the Tracker class.

Created on Mon Apr 24 10:57:01 2017

@author: yosef
"""

import unittest
from math import isclose

import numpy as np

from openptv_python.parameters import (
    ControlPar,
    TrackPar,
)
from openptv_python.track import (
    angle_acc,
    candsearch_in_pix,
    candsearch_in_pix_rest,
    pos3d_in_bounds,
    predict,
    search_volume_center_moving,
)
from openptv_python.tracking_frame_buf import Target
from openptv_python.vec_utils import vec_scalar_mul


class TestPredict(unittest.TestCase):
    """Test the predict function."""

    def test_predict(self):
        """Test the predict function."""
        prev_pos = [1.1, 0.6]
        curr_pos = [2.0, -0.8]
        result = [2.9, -2.2]
        EPS = 1e-9  # You can adjust this epsilon value as needed

        c = [0.0, 0.0]
        predict(prev_pos, curr_pos, c)

        self.assertTrue(
            isclose(c[0], result[0], rel_tol=EPS), f"Expected 2.9 but found {c[0]}"
        )
        self.assertTrue(
            isclose(c[1], result[1], rel_tol=EPS), f"Expected -2.2 but found {c[1]}"
        )


class TestSearchVolumeCenterMoving(unittest.TestCase):
    """Test the search_volume_center_moving function."""

    def test_search_volume_center_moving(self):
        """Test the search_volume_center_moving function."""
        prev_pos = [1.1, 0.6, 0.1]
        curr_pos = [2.0, -0.8, 0.2]
        result = [2.9, -2.2, 0.3]
        EPS = 1e-9  # You can adjust this epsilon value as needed

        c = [0.0, 0.0, 0.0]
        search_volume_center_moving(prev_pos, curr_pos, c)

        self.assertTrue(
            isclose(c[0], result[0], rel_tol=EPS), f"Expected 2.9 but found {c[0]}"
        )
        self.assertTrue(
            isclose(c[1], result[1], rel_tol=EPS), f"Expected -2.2 but found {c[1]}"
        )
        self.assertTrue(
            isclose(c[2], result[2], rel_tol=EPS), f"Expected 0.3 but found {c[2]}"
        )


class TestPos3dInBounds(unittest.TestCase):
    """Test the pos3d_in_bounds function."""

    def test_pos3d_in_bounds(self):
        """Test the pos3d_in_bounds function."""
        inside = [1.0, -1.0, 0.0]
        outside = [2.0, -0.8, 2.1]

        bounds = TrackPar(
            0.4, 120, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1
        )

        result = pos3d_in_bounds(inside, bounds)

        self.assertEqual(result, 1, "Expected True but found %s" % result)

        result = pos3d_in_bounds(outside, bounds)

        self.assertEqual(result, 0, "Expected False but found %s" % result)


class TestAngleAcc(unittest.TestCase):
    def test_angle_acc(self):
        start = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, 1.0, 1.0])
        cand = np.array([1.1, 1.0, 1.0])
        EPS = 1e-7  # You can adjust this epsilon value as needed

        angle = 0.0
        acc = 0.0

        angle, acc = angle_acc(start, pred, cand)
        self.assertTrue(
            isclose(angle, 2.902234, rel_tol=EPS),
            f"Expected 2.902234 but found {angle}",
        )
        self.assertTrue(isclose(acc, 0.1, rel_tol=EPS), f"Expected 0.1 but found {acc}")

        angle = 0.0
        acc = 0.0

        angle, acc = angle_acc(start, pred, pred)
        self.assertTrue(
            isclose(angle, 0.0, rel_tol=EPS), f"Expected 0.0 but found {angle}"
        )
        self.assertTrue(isclose(acc, 0.0, rel_tol=EPS), f"Expected 0.0 but found {acc}")

        cand = vec_scalar_mul(pred, -1)

        angle, acc = angle_acc(start, pred, cand)
        self.assertTrue(
            isclose(angle, 200.0, rel_tol=EPS), f"Expected 200.0 but found {angle}"
        )


class TestCandSearchInPix(unittest.TestCase):
    def test_candsearch_in_pix(self):
        test_targets = [
            Target(0, 0.0, -0.2, 5, 1, 2, 10, -999),
            Target(6, 0.2, 0.2, 10, 8, 1, 20, -999),
            Target(3, 0.2, 0.3, 10, 3, 3, 30, -999),
            Target(4, 0.2, 1.0, 10, 3, 3, 40, -999),
            Target(1, -0.7, 1.2, 10, 3, 3, 50, -999),
            Target(7, 1.2, 1.3, 10, 3, 3, 60, -999),
            Target(5, 10.4, 2.1, 10, 3, 3, 70, -999),
        ]
        num_targets = len(test_targets)

        cent_x = 0.2
        cent_y = 0.2
        dl = dr = du = dd = 0.1

        num_cams = 4
        p = [-1] * num_cams  # Initialize p with zeros

        test_cpar = ControlPar(num_cams=num_cams)
        img_format = "cam{}"
        cal_format = "cal/cam{}.tif"

        test_cpar.img_base_name = [""] * num_cams
        test_cpar.cal_img_base_name = [""] * num_cams

        for cam in range(num_cams):
            test_cpar.img_base_name[cam] = img_format.format(cam + 1)
            test_cpar.cal_img_base_name[cam] = cal_format.format(cam + 1)

        test_cpar.hp_flag = 1
        test_cpar.allCam_flag = 0
        test_cpar.tiff_flag = 1
        test_cpar.imx = 1280
        test_cpar.imy = 1024
        test_cpar.pix_x = 0.02
        #  20 micron pixel size
        test_cpar.pix_y = 0.02
        test_cpar.chfield = 0
        test_cpar.mm.n1 = 1
        test_cpar.mm.n2[0] = 1.49
        test_cpar.mm.n3 = 1.33
        test_cpar.mm.d[0] = 5

        counter = candsearch_in_pix(
            test_targets, num_targets, cent_x, cent_y, dl, dr, du, dd, p, test_cpar
        )
        # print(f"p = {p}, counter = {counter}")

        self.assertEqual(counter, 2)

        cent_x = 0.5
        cent_y = 0.3
        dl = dr = du = dd = 10.2
        p = [-1] * num_cams  # Initialize p with zeros

        counter = candsearch_in_pix(
            test_targets, num_targets, cent_x, cent_y, dl, dr, du, dd, p, test_cpar
        )
        # print(f"p = {p}, counter = {counter}")
        self.assertEqual(counter, 4)

        test_targets = [
            Target(0, 0.0, -0.2, 5, 1, 2, 10, 0),
            Target(6, 100.0, 100.0, 10, 8, 1, 20, -1),
            Target(3, 102.0, 102.0, 10, 3, 3, 30, -1),
            Target(4, 103.0, 103.0, 10, 3, 3, 40, 2),
            Target(1, -0.7, 1.2, 10, 3, 3, 50, 5),
            Target(7, 1.2, 1.3, 10, 3, 3, 60, 7),
            Target(5, 1200, 201.1, 10, 3, 3, 70, 11),
        ]
        num_targets = len(test_targets)

        cent_x = cent_y = 98.9
        dl = dr = du = dd = 3
        p = [-1] * num_cams  # Initialize p

        counter = candsearch_in_pix_rest(
            test_targets, num_targets, cent_x, cent_y, dl, dr, du, dd, p, test_cpar
        )

        # print(f"p = {p}, counter = {counter}")
        self.assertEqual(counter, 1)
        self.assertTrue(
            isclose(test_targets[p[0]].x, 100.0, rel_tol=1e-9),
            f"Expected 100.0 but found {test_targets[p[0]].x}",
        )


if __name__ == "__main__":
    unittest.main()
