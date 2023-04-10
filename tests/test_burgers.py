# -*- coding: utf-8 -*-
import os
import shutil
import unittest

import yaml

from .calibration import Calibration
from .parameters import (
    ControlParams,
    SequenceParams,
    TrackingParams,
    VolumeParams,
)
from .tracker import Tracker

framebuf_naming = {
    "corres": b"testing_folder/burgers/res/rt_is",
    "linkage": b"testing_folder/burgers/res/ptv_is",
    "prio": b"testing_folder/burgers/res/whatever",
}


class TestTracker(unittest.TestCase):
    def setUp(self):
        with open(b"testing_folder/burgers/conf.yaml") as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        seq_cfg = yaml_conf["sequence"]

        cals = []
        img_base = []
        print(yaml_conf["cameras"])
        for cix, cam_spec in enumerate(yaml_conf["cameras"]):
            cam_spec.setdefault(b"addpar_file", None)
            cal = Calibration()
            cal.from_file(
                cam_spec["ori_file"].encode(), cam_spec["addpar_file"].encode()
            )
            cals.append(cal)
            img_base.append(seq_cfg["targets_template"].format(cam=cix + 1))

        cpar = ControlParams(len(yaml_conf["cameras"]), **yaml_conf["scene"])
        vpar = VolumeParams(**yaml_conf["correspondences"])
        tpar = TrackingParams(**yaml_conf["tracking"])
        spar = SequenceParams(
            image_base=img_base, frame_range=(seq_cfg["first"], seq_cfg["last"])
        )

        self.tracker = Tracker(cpar, vpar, tpar, spar, cals, framebuf_naming)

    def test_forward(self):
        """Manually running a full forward tracking run."""
        # path = 'testing_folder/burgers/res'
        # try:
        #     os.mkdir(path)
        # except OSError:
        #     print("Creation of the directory %s failed" % path)
        # else:
        #     print("Successfully created the directory %s " % path)

        shutil.copytree(
            "testing_folder/burgers/res_orig/", "testing_folder/burgers/res/"
        )
        shutil.copytree(
            "testing_folder/burgers/img_orig/", "testing_folder/burgers/img/"
        )

        self.tracker.restart()
        last_step = 10001
        while self.tracker.step_forward():
            self.failUnless(self.tracker.current_step() > last_step)
            with open("testing_folder/burgers/res/rt_is.%d" % last_step) as f:
                lines = f.readlines()
                # print(last_step,lines[0])
                # print(lines)
                if last_step == 10003:
                    self.failUnless(lines[0] == "4\n")
                else:
                    self.failUnless(lines[0] == "5\n")
            last_step += 1
        self.tracker.finalize()

    def test_full_forward(self):
        """Automatic full forward tracking run."""
        # os.mkdir('testing_folder/burgers/res')
        shutil.copytree(
            "testing_folder/burgers/res_orig/", "testing_folder/burgers/res/"
        )
        shutil.copytree(
            "testing_folder/burgers/img_orig/", "testing_folder/burgers/img/"
        )
        self.tracker.full_forward()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def test_full_backward(self):
        """Automatic full backward correction phase."""
        shutil.copytree(
            "testing_folder/burgers/res_orig/", "testing_folder/burgers/res/"
        )
        shutil.copytree(
            "testing_folder/burgers/img_orig/", "testing_folder/burgers/img/"
        )
        self.tracker.full_forward()
        self.tracker.full_backward()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def tearDown(self):
        if os.path.exists("testing_folder/burgers/res/"):
            shutil.rmtree("testing_folder/burgers/res/")
        if os.path.exists("testing_folder/burgers/img/"):
            shutil.rmtree("testing_folder/burgers/img/")
            # print("there is a /res folder\n")
            # pass


if __name__ == "__main__":
    unittest.main()
