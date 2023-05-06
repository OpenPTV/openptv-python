#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test the epipolar curve code, at least for simple cases.

Created on Thu Mar 23 16:12:21 2017

@author: yosef
"""

import unittest
from math import isclose

import numpy as np

from openptv_python.calibration import Calibration, Exterior, Glass, Interior, ap_52
from openptv_python.epi import Candidate, Coord2d, epipolar_curve
from openptv_python.find_candidate import find_candidate
from openptv_python.parameters import (
    ControlPar,
    MultimediaPar,
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
        # we need to improve this
        self.assertTrue(np.all(np.abs(line - mid) < 1e-4))

        # An equatorial point draws a latitude.
        line = epipolar_curve(
            mid - np.r_[100.0, 0.0], orig_cal, proj_cal, 5, cpar, vpar
        )
        np.testing.assert_array_equal(np.argsort(line[:, 0]), np.arange(5)[::-1])
        self.assertTrue(np.all(abs(line[:, 1] - mid[1]) < 1e-6))


class TestFindCandidate(unittest.TestCase):
    """Test the find_candidate function."""

    def test_find_candidate(self):
        """Test the find_candidate function."""
        # set of particles to choose from
        test_pix = [
            [0, 0.0, -0.2, 5, 1, 2, 10, -999],
            [6, 0.1, 0.1, 10, 8, 1, 20, -999],
            [3, 0.2, 0.8, 10, 3, 3, 30, -999],
            [4, 0.4, -1.1, 10, 3, 3, 40, -999],
            [1, 0.7, -0.1, 10, 3, 3, 50, -999],
            [2, 1.2, 0.3, 10, 3, 3, 60, -999],
            [5, 10.4, 0.1, 10, 3, 3, 70, -999],
        ]

        test_pix = [Target(*x) for x in test_pix]

        # length of the test_pix
        num_pix = len(test_pix)

        # coord_2d is int pnr, double x,y
        # note that it's x-sorted by construction
        test_crd = [
            [6, 0.1, 0.1],
            [3, 0.2, 0.8],
            [4, 0.4, -1.1],
            [1, 0.7, -0.1],
            [2, 1.2, 0.3],
            [0, 0.0, 0.0],
            [5, 10.4, 0.1],
        ]

        test_crd = [Coord2d(*x) for x in test_crd]

        # parameters of the particle for which we look for the candidates
        n = 10
        nx = 3
        ny = 3
        sumg = 100

        test_Ex = Exterior(x0=0.0, y0=0.0, z0=100.0, omega=0.0, phi=0.0, kappa=0.0)
        test_I = Interior(0.0, 0.0, 100.0)
        test_G = Glass(0.0, 0.0, 50.0)
        test_addp = ap_52(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        test_cal = Calibration(test_Ex, test_I, test_G, test_addp)
        test_mm = MultimediaPar(1, 1.0, [1.49, 0.0, 0.0], [5.0, 0.0, 0.0], 1.33)
        test_vpar = VolumePar(
            [-250.0, 250.0],
            [-100.0, -100.0],
            [100.0, 100.0],
            0.01,
            0.3,
            0.3,
            0.01,
            1.0,
            33,
        )
        test_cpar = ControlPar(4)
        test_cpar.set_image_size([1280, 1024])
        test_cpar.set_pixel_size([0.02, 0.02])
        test_cpar.mm = test_mm

        # the result is that the sensor size is 12.8 mm x 10.24 mm

        #  epipolar line
        xa = -10.0
        ya = -10.0
        xb = 10.0
        yb = 10.0

        print(test_vpar.eps0)

        test_cand = find_candidate(
            test_crd,
            test_pix,
            num_pix,
            xa,
            ya,
            xb,
            yb,
            n,
            nx,
            ny,
            sumg,
            test_vpar,
            test_cpar,
            test_cal,
        )

        # for t in test_cand:
        #     print(t)

        # output of the check_epi.c from liboptv/tests
        # 13: candidate 0: pnr 0, corr 1156.000000, tol 0.000000
        # 13: candidate 1: pnr 1, corr 784.000000, tol 0.424264
        # 13: candidate 2: pnr 3, corr 421.000000, tol 0.565685
        # 13: candidate 3: pnr 4, corr 676.000000, tol 0.636396
        # 13: candidate 4: pnr 5, corr 264.000000, tol 0.000000

        expected = [
            Candidate(pnr=0, corr=1156.0, tol=0.0),
            Candidate(pnr=1, corr=784.0, tol=0.424264),
            Candidate(pnr=3, corr=421.0, tol=0.565685),
            Candidate(pnr=4, corr=676.000000, tol=0.636396),
            Candidate(pnr=5, corr=264.000000, tol=0.000000),
        ]

        self.assertTrue(len(test_cand) == len(expected))
        for t, e in zip(test_cand, expected):
            self.assertTrue(t.pnr == e.pnr)
            self.assertTrue(t.corr == e.corr)
            print(t.tol, e.tol)
            self.assertTrue(isclose(t.tol, e.tol, abs_tol=1e-5))


if __name__ == "__main__":
    unittest.main()
