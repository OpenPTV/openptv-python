#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tests for the Tracker class.

Created on Mon Apr 24 10:57:01 2017

@author: yosef
"""

import shutil
import unittest
from pathlib import Path

from openptv_python.calibration import Calibration, read_calibration
from openptv_python.parameters import (
    read_control_par,
)
from openptv_python.tracking_run import (
    tr_new_legacy,
    track_forward_start,
    trackcorr_c_finish,
    trackcorr_c_loop,
)


def remove_directory(directory_path):
    # Convert the input to a Path object
    path = Path(directory_path)

    # Iterate over all items in the directory
    for item in path.iterdir():
        if item.is_file():
            # Remove files
            item.unlink()
        elif item.is_dir():
            # Remove subdirectories and their contents
            remove_directory(item)

    # Remove the directory itself
    path.rmdir()


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


class TestTrackCorrNoAdd(unittest.TestCase):
    def test_trackcorr_no_add(self):
        EPS = 1e-9
        """Test tracking without adding particles."""
        copy_directory(
            "tests/testing_fodder/track/res_orig/", "tests/testing_fodder/track/res/"
        )
        copy_directory(
            "tests/testing_fodder/track/img_orig/", "tests/testing_fodder/track/img/"
        )

        print("----------------------------")
        print("Test tracking multiple files 2 cameras, 1 particle")
        cpar = read_control_par("tests/testing_fodder/track/parameters/ptv.par")

        calib = read_all_calibration(cpar.num_cams)

        run = tr_new_legacy(
            "tests/testing_fodder/track/parameters/sequence.par",
            "tests/testing_fodder/track/parameters/track.par",
            "tests/testing_fodder/track/parameters/criteria.par",
            "tests/testing_fodder/track/parameters/ptv.par",
            calib,
        )
        run.tpar.add = 0

        print(f"run.tpar = {run.tpar}")

        track_forward_start(run)
        trackcorr_c_loop(run, run.seq_par.first)

        for step in range(run.seq_par.first + 1, run.seq_par.last):
            trackcorr_c_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        remove_directory("tests/testing_fodder/track/res/")
        remove_directory("tests/testing_fodder/track/img/")

        range_val = run.seq_par.last - run.seq_par.first
        npart = run.npart / range_val
        nlinks = run.nlinks / range_val

        self.assertAlmostEqual(npart, 0.8, delta=EPS)
        self.assertAlmostEqual(nlinks, 0.8, delta=EPS)


if __name__ == "__main__":
    unittest.main()