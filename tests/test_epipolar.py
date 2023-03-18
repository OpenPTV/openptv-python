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
from openptv_python.epipolar import epipolar_curve
from openptv_python.parameters import ControlParams, VolumeParams


class TestEpipolarCurve(unittest.TestCase):
    def test_two_cameras(self):
        ori_tmpl = "testing_folder/calibration/sym_cam{cam_num}.tif.ori"
        add_file = "testing_folder/calibration/cam1.tif.addpar"

        orig_cal = Calibration()
        orig_cal.from_file(ori_tmpl.format(cam_num=1).encode(), add_file.encode())
        proj_cal = Calibration()
        proj_cal.from_file(ori_tmpl.format(cam_num=3).encode(), add_file.encode())

        # reorient cams:
        orig_cal.set_angles(np.r_[0.0, -np.pi / 4.0, 0.0])
        proj_cal.set_angles(np.r_[0.0, 3 * np.pi / 4.0, 0.0])

        cpar = ControlParams(4)
        cpar.read_control_par(b"testing_folder/corresp/control.par")
        sens_size = cpar.get_image_size()

        vpar = VolumeParams()
        vpar.read_volume_par(b"testing_folder/corresp/criteria.par")
        vpar.set_Zmin_lay([-10, -10])
        vpar.set_Zmax_lay([10, 10])

        mult_params = cpar.get_multimedia_params()
        mult_params.set_n1(1.0)
        mult_params.set_layers(np.array([1.0]), np.array([1.0]))
        mult_params.set_n3(1.0)

        # Central point translates to central point because cameras point
        # directly at each other.
        mid = np.r_[sens_size] / 2.0
        line = epipolar_curve(mid, orig_cal, proj_cal, 5, cpar, vpar)
        self.assertTrue(np.all(abs(line - mid) < 1e-6))

        # An equatorial point draws a latitude.
        line = epipolar_curve(
            mid - np.r_[100.0, 0.0], orig_cal, proj_cal, 5, cpar, vpar
        )
        np.testing.assert_array_equal(np.argsort(line[:, 0]), np.arange(5)[::-1])
        self.assertTrue(np.all(abs(line[:, 1] - mid[1]) < 1e-6))
