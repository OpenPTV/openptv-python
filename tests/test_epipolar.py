#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test the epipolar curve code, at least for simple cases.

Created on Thu Mar 23 16:12:21 2017

@author: yosef
"""

import unittest

import numpy as np

from openptv_python.calibration import Calibration
from openptv_python.constants import MAXCAND
from openptv_python.epi import Candidate, Coord2d, epipolar_curve
from openptv_python.find_candidate import find_candidate
from openptv_python.parameters import (
    ControlPar,
    VolumePar,
    read_control_par,
    read_volume_par,
)
from openptv_python.tracking_frame_buf import Target


class TestEpipolarCurve(unittest.TestCase):
    """Test the epipolar curve code."""

    def test_two_cameras(self):
        """Test the epipolar curve code for two cameras."""
        cam_num = 1
        ori_tmpl = f"tests/testing_folder/calibration/sym_cam{cam_num}.tif.ori"
        add_file = "tests/testing_folder/calibration/cam1.tif.addpar"

        orig_cal = Calibration()
        orig_cal.from_file(ori_tmpl, add_file)

        proj_cal = Calibration()
        cam_num = 3
        ori_tmpl = f"tests/testing_folder/calibration/sym_cam{cam_num}.tif.ori"
        proj_cal.from_file(ori_tmpl, add_file)

        # reorient cams:
        orig_cal.set_angles(np.r_[0.0, -np.pi / 4.0, 0.0])
        proj_cal.set_angles(np.r_[0.0, 3 * np.pi / 4.0, 0.0])

        cpar = ControlPar(4)
        cpar = read_control_par("tests/testing_folder/corresp/control.par")
        sens_size = cpar.get_image_size()

        vpar = read_volume_par("tests/testing_folder/corresp/criteria.par")
        vpar.set_Zmin_lay([-10, -10])
        vpar.set_Zmax_lay([10, 10])

        mult_params = cpar.mm
        mult_params.set_n1(1.0)
        mult_params.set_layers(np.array([1.0]), np.array([1.0]))
        mult_params.set_n3(1.0)

        # Central point translates to central point because cameras point
        # directly at each other.
        mid = np.r_[sens_size] / 2.0
        line = epipolar_curve(mid, orig_cal, proj_cal, 5, cpar, vpar)
        self.assertTrue(np.all(np.abs(line - mid) < 1e-4))  # we need to improve this

        # An equatorial point draws a latitude.
        line = epipolar_curve(
            mid - np.r_[100.0, 0.0], orig_cal, proj_cal, 5, cpar, vpar
        )
        np.testing.assert_array_equal(np.argsort(line[:, 0]), np.arange(5)[::-1])
        self.assertTrue(np.all(abs(line[:, 1] - mid[1]) < 1e-6))


class TestFindCandidate(unittest.TestCase):
    def test_find_candidate(self):
        """Test the find_candidate function."""
        crd = [Coord2d(0, 0, 0), Coord2d(1, 1, 1), Coord2d(2, 2, 2)]
        pix = [Target(10, 20, 30, 40), Target(15, 25, 35, 45), Target(20, 30, 40, 50)]
        num = len(crd)
        xa, ya, xb, yb = -1, -1, 3, 3
        n, nx, ny, sumg = 15, 25, 35, 45
        cand = [Candidate() for _ in range(3)]
        vpar = VolumePar()
        cpar = ControlPar()
        cal = Calibration()

        # Expected output
        expected = 2

        # Test the function
        output = find_candidate(
            crd, pix, num, xa, ya, xb, yb, n, nx, ny, sumg, cand, vpar, cpar, cal
        )

        # Check the output
        assert output == expected

        # Test when there are more candidates than MAXCAND
        crd = [Coord2d(x, x, x) for x in range(MAXCAND + 1)]
        pix = [Target(10, 20, 30, 40) for _ in range(MAXCAND + 1)]
        num = len(crd)
        expected = -1
        output = find_candidate(
            crd, pix, num, xa, ya, xb, yb, n, nx, ny, sumg, cand, vpar, cpar, cal
        )
        assert output == expected


if __name__ == "__main__":
    unittest.main()
