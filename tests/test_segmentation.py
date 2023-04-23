# -*- coding: utf-8 -*-
"""
Tests for the segmentation module. Basically recreates the C tests.

Created on Thu Aug 18 16:52:36 2016

@author: yosef
"""

import unittest

import numpy as np

from openptv_python.parameters import ControlPar, TargetPar
from openptv_python.segmentation import target_recognition


class TestTargRec(unittest.TestCase):
    def test_single_target(self):
        img = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 255, 255, 255, 0],
                [0, 255, 255, 255, 0],
                [0, 255, 255, 255, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        cpar = ControlPar(4)
        cpar.set_image_size = (5, 5)
        tpar = TargetPar(
            gvthresh=[250, 100, 20, 20],
            discont=5,
            nnmin=1,
            nnmax=10,
            sumg_min=12,
            nxmin=1,
            nxmax=10,
            nymin=1,
            nymax=10,
        )

        targs = target_recognition(img, tpar, 0, cpar)

        self.assertEqual(len(targs), 1)
        self.assertEqual(targs[0].count_pixels(), (9, 3, 3))

    def test_two_targets(self):
        img = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 255, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 251, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        cpar = ControlPar(4)
        cpar.set_image_size = (5, 5)
        tpar = TargetPar(
            gvthresh=[250, 100, 20, 20],
            discont=5,
            nnmin=1,
            nnmax=10,
            sumg_min=12,
            nxmin=1,
            nxmax=10,
            nymin=1,
            nymax=10,
        )

        targs = target_recognition(img, tpar, 0, cpar)

        self.assertEqual(len(targs), 2)
        self.assertEqual(targs[0].count_pixels(), (1, 1, 1))

        # Exclude the first target and try again:
        tpar.gvthresh = [252, 100, 20, 20]
        targs = target_recognition(img, tpar, 0, cpar)

        self.assertEqual(len(targs), 1)
        self.assertEqual(targs[0].count_pixels(), (1, 1, 1))

    # the following is a test code
    def test_one_targets2(self):
        img = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 255, 250, 250, 0],
                [0, 251, 253, 251, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        cpar = ControlPar(4)
        cpar.set_image_size = (5, 5)
        tpar = TargetPar(
            gvthresh=[250, 100, 20, 20],
            discont=5,
            nnmin=1,
            nnmax=10,
            sumg_min=12,
            nxmin=1,
            nxmax=10,
            nymin=1,
            nymax=10,
        )

        targs = target_recognition(img, tpar, 0, cpar)

        self.assertEqual(len(targs), 1)
        self.assertEqual(targs[0].count_pixels(), (4, 3, 2))


if __name__ == "__main__":
    unittest.main()
