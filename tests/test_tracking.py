#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tests for the Tracker class.

Created on Mon Apr 24 10:57:01 2017

@author: yosef
"""

import shutil
import unittest
from math import isclose
from pathlib import Path

import numpy as np

from openptv_python.calibration import Calibration, read_calibration
from openptv_python.constants import MAX_CANDS, TR_UNUSED
from openptv_python.parameters import (
    ControlPar,
    TrackPar,
)
from openptv_python.track import (
    Foundpix,
    angle_acc,
    candsearch_in_pix,
    candsearch_in_pix_rest,
    copy_foundpix_array,
    pos3d_in_bounds,
    predict,
    reset_foundpix_array,
    search_volume_center_moving,
    searchquader,
    sort,
    sort_candidates_by_freq,
)
from openptv_python.tracking_frame_buf import Target
from openptv_python.vec_utils import vec_scalar_mul


def copy_directory(source_path, destination_path):
    """Copy the contents of the source directory to the destination directory."""
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    # Create the destination directory if it doesn't exist
    destination_path.mkdir(parents=True, exist_ok=True)

    # Copy the contents from the source to the destination
    for item in source_path.iterdir():
        if item.is_dir():
            shutil.copytree(item, destination_path / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination_path / item.name)


def read_all_calibration(num_cams: int = 4) -> list[Calibration]:
    """Read all calibration files."""
    ori_tmpl = "tests/testing_fodder/track/cal/cam%d.tif.ori"
    added_tmpl = "tests/testing_fodder/track/cal/cam%d.tif.addpar"

    calib = []

    for cam in range(num_cams):
        ori_name = ori_tmpl % (cam + 1)
        added_name = added_tmpl % (cam + 1)
        calib.append(read_calibration(ori_name, added_name))

    return calib


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

        self.assertEqual(result, 1, f"Expected True but found {result}")

        result = pos3d_in_bounds(outside, bounds)

        self.assertEqual(result, 0, f"Expected False but found {result}")


class TestAngleAcc(unittest.TestCase):
    """Test the angle_acc function."""

    def test_angle_acc(self):
        """Test the angle_acc function."""
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

        p = candsearch_in_pix(
            test_targets, num_targets, cent_x, cent_y, dl, dr, du, dd, test_cpar
        )
        counter = len(p) - p.count(-1)

        print(f"p = {p}, counter = {counter}")

        self.assertEqual(counter, 2)

        cent_x = 0.5
        cent_y = 0.3
        dl = dr = du = dd = 10.2
        p = [-1] * num_cams  # Initialize p with zeros

        p = candsearch_in_pix(
            test_targets, num_targets, cent_x, cent_y, dl, dr, du, dd, test_cpar
        )
        counter = len(p) - p.count(-1)
        print(f"p = {p}, counter = {counter}")

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


class TestSort(unittest.TestCase):
    def test_sort(self):
        test_array = np.array([1.0, 2200.2, 0.3, -0.8, 100.0], dtype=float)
        ix_array = np.array([0, 5, 13, 2, 124], dtype=int)
        len_array = 5

        sort(test_array, ix_array)

        self.assertTrue(
            isclose(test_array[0], -0.8, rel_tol=1e-9),
            f"Expected -0.8 but found { test_array[0] } ",
        )
        self.assertNotEqual(
            ix_array[len_array - 1],
            1,
            f"Expected not to be 1 but found { ix_array[len_array - 1] }",
        )

    def test_copy_foundpix_array(self):
        """Test the copy_foundpix_array function."""
        src = [Foundpix(1, 1, [1, 0]), Foundpix(2, 5, [1, 1])]
        arr_len = 2
        num_cams = 2

        dest = [
            Foundpix(TR_UNUSED, 0, [0] * num_cams) for _ in range(num_cams * MAX_CANDS)
        ]

        reset_foundpix_array(dest, num_cams * MAX_CANDS, num_cams)

        self.assertEqual(
            dest[1].ftnr, -1, f"Expected dest[1].ftnr == -1 but found {dest[1].ftnr}"
        )
        self.assertEqual(
            dest[0].freq, 0, f"Expected dest[0].freq == 0 but found {dest[0].freq}"
        )
        self.assertEqual(
            dest[1].whichcam[0], 0, f"Expected 0 but found {dest[1].whichcam[0]}"
        )

        copy_foundpix_array(dest, src, arr_len, num_cams)

        self.assertEqual(
            dest[1].ftnr, 2, f"Expected dest[1].ftnr == 2 but found {dest[1].ftnr}"
        )

        # print("Destination foundpix array:")
        # for i in range(arr_len):
        #     print(f"ftnr = {dest[i].ftnr} freq = {dest[i].freq} whichcam = {dest[i].whichcam}")


class TestSearchQuader(unittest.TestCase):
    def setUp(self):
        self.cpar = ControlPar()
        self.cpar.from_file("tests/testing_fodder/track/parameters/ptv.par")
        self.cpar.mm.n2[0] = 1.0
        self.cpar.mm.n3 = 1.0

        # self.calib = [None] * self.cpar.num_cams
        self.calib = read_all_calibration(self.cpar.num_cams)

    def test_searchquader(self):
        """Test the searchquader function."""
        point = np.array([185.5, 3.2, 203.9])

        #  print(f"cpar = {self.cpar}")

        tpar = TrackPar(
            0.4, 120, 0.2, -0.2, 0.1, -0.1, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 1
        )
        xr, xl, yd, yu = searchquader(point, tpar, self.cpar, self.calib)

        # print(f"xr = {xr}, xl = {xl}, yd = {yd}, yu = {yu}")

        self.assertTrue(
            isclose(xr[0], 0.560048, rel_tol=1e-6),
            f"Expected 0.560048 but found {xr[0]}",
        )
        self.assertTrue(
            isclose(yu[1], 0.437303, rel_tol=1e-6),
            f"Expected 0.437303 but found {yu[1]}",
        )

        # Let's test with just one camera to check borders
        self.cpar.num_cams = 1
        tpar1 = TrackPar(
            0.4, 120, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 1
        )
        xr, xl, yd, yu = searchquader(point, tpar1, self.cpar, self.calib)

        # print(f"xr = {xr}, xl = {xl}, yd = {yd}, yu = {yu}")

        self.assertTrue(
            isclose(xr[0], 0.0, rel_tol=1e-9), f"Expected 0.0 but found {xr[0]}"
        )

        # Test with infinitely large values of tpar that should return about half the image size
        tpar2 = TrackPar(
            0.4,
            120,
            1000.0,
            -1000.0,
            1000.0,
            -1000.0,
            1000.0,
            -1000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1,
        )

        xr, xl, yd, yu = searchquader(point, tpar2, self.cpar, self.calib)

        # print(f"xr = {xr}, xl = {xl}, yd = {yd}, yu = {yu}")

        self.assertTrue(
            isclose(xr[0] + xl[0], self.cpar.imx, rel_tol=1e-9),
            f"Expected image size but found {xr[0] + xl[0]}",
        )
        self.assertTrue(
            isclose(yd[0] + yu[0], self.cpar.imy, rel_tol=1e-9),
            f"Expected {self.cpar.imy} but found {yd[0] + yu[0]}",
        )


class TestSortCandidatesByFreq(unittest.TestCase):
    """Test the sort_candidates_by_freq function."""

    def test_sort_candidates_by_freq(self):
        """Test the sort_candidates_by_freq function."""
        src = [Foundpix(1, 0, [1, 0]), Foundpix(2, 0, [1, 1])]
        num_cams = 2

        # allocate
        dest = [
            Foundpix(TR_UNUSED, 0, [0] * num_cams) for _ in range(num_cams * MAX_CANDS)
        ]

        # sortwhatfound freaks out if the array is not reset before
        reset_foundpix_array(dest, 2, 2)
        copy_foundpix_array(dest, src, 2, 2)
        # print(f"src = {src}")
        # print(f"dest = {dest}")

        # test simple sort of a small foundpix array
        sort_candidates_by_freq(dest, num_cams)

        # print(f"num_parts = {num_parts}")
        # self.assertEqual(num_parts, 1)
        self.assertEqual(dest[0].ftnr, 2)
        self.assertEqual(dest[0].freq, 2)
        self.assertEqual(dest[1].freq, 0)


if __name__ == "__main__":
    unittest.main()
