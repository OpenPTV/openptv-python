#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tests for the Tracker class.

Created on Mon Apr 24 10:57:01 2017

@author: yosef
"""

import os
import shutil
import unittest

import yaml

from openptv_python.calibration import Calibration
from openptv_python.parameters import (
    ControlPar,
    SequenceParams,
    TrackingParams,
    VolumeParams,
)
from openptv_python.tracker import Tracker

framebuf_naming = {
    "corres": "tests/testing_folder/track/res/particles",
    "linkage": "tests/testing_folder/track/res/linkage",
    "prio": "tests/testing_folder/track/res/whatever",
}


class TestTracker(unittest.TestCase):
    def setUp(self):
        with open("tests/testing_folder/track/conf.yaml") as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        seq_cfg = yaml_conf["sequence"]

        cals = []
        img_base = []
        print((yaml_conf["cameras"]))
        for cix, cam_spec in enumerate(yaml_conf["cameras"]):
            cam_spec.setdefault(b"addpar_file", None)
            cal = Calibration()
            cal.from_file(
                cam_spec["ori_file"].encode(), cam_spec["addpar_file"].encode()
            )
            cals.append(cal)
            img_base.append(seq_cfg["targets_template"].format(cam=cix + 1))

        cpar = ControlPar(len(yaml_conf["cameras"]), **yaml_conf["scene"])
        vpar = VolumeParams(**yaml_conf["correspondences"])
        tpar = TrackingParams(**yaml_conf["tracking"])
        spar = SequenceParams(
            image_base=img_base, frame_range=(seq_cfg["first"], seq_cfg["last"])
        )

        self.tracker = Tracker(cpar, vpar, tpar, spar, cals, framebuf_naming)

    def test_forward(self):
        """Manually running a full forward tracking run."""
        shutil.copytree("testing_folder/track/res_orig/", "testing_folder/track/res/")

        self.tracker.restart()
        last_step = 10001
        while self.tracker.step_forward():
            # print(f"step is {self.tracker.current_step()}\n")
            # print(self.tracker.current_step() > last_step)
            self.assertTrue(self.tracker.current_step() > last_step)
            with open("testing_folder/track/res/linkage.%d" % last_step) as f:
                lines = f.readlines()
                # print(last_step,lines[0])
                if last_step == 10003:
                    self.assertTrue(lines[0] == "-1\n")
                else:
                    self.assertTrue(lines[0] == "1\n")
            last_step += 1
        self.tracker.finalize()

    def test_full_forward(self):
        """Automatic full forward tracking run."""
        shutil.copytree("testing_folder/track/res_orig/", "testing_folder/track/res/")
        self.tracker.full_forward()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def test_full_backward(self):
        """Automatic full backward correction phase."""
        shutil.copytree("testing_folder/track/res_orig/", "testing_folder/track/res/")
        self.tracker.full_forward()
        self.tracker.full_backward()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def tearDown(self):
        if os.path.exists("testing_folder/track/res/"):
            shutil.rmtree("testing_folder/track/res/")


if __name__ == "__main__":
    unittest.main()
