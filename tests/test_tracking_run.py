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
from pathlib import Path

import numpy as np

from openptv_python.calibration import Calibration, read_calibration
from openptv_python.parameters import (
    read_control_par,
)
from openptv_python.track import (
    track_forward_start,
    trackback_c,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv_python.tracking_run import (
    tr_new,
)
from openptv_python.vec_utils import vec_norm

EPS = 1e-9


def remove_directory(directory_path):
    """Remove a directory and its contents."""
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
    ori_tmpl = "cal/cam%d.tif.ori"
    added_tmpl = "cal/cam%d.tif.addpar"

    calib = []

    for cam in range(num_cams):
        ori_name = ori_tmpl % (cam + 1)
        added_name = added_tmpl % (cam + 1)
        calib.append(read_calibration(ori_name, added_name))

    return calib


class TestTrackCorrNoAdd(unittest.TestCase):
    """Test tracking without adding particles."""

    def test_trackcorr_no_add(self):
        """Test tracking without adding particles."""
        # import os

        current_directory = Path.cwd()
        print(f"working from {current_directory}")
        directory = Path("tests/testing_fodder/track")

        os.chdir(directory)

        print(os.path.abspath(os.curdir))
        print(Path.cwd())

        if Path("res/").exists():
            remove_directory("res/")

        if Path("img/").exists():
            remove_directory("img/")


        copy_directory("res_orig/", "res/")
        copy_directory("img_orig/", "img/")

        print("----------------------------")
        print("Test tracking multiple files 2 cameras, 1 particle")
        cpar = read_control_par("parameters/ptv.par")

        calib = read_all_calibration(cpar.num_cams)

        run = tr_new(
            "parameters/sequence.par",
            "parameters/track.par",
            "parameters/criteria.par",
            "parameters/ptv.par",
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            calib,
            10000.0,
        )
        run.tpar.add = 0
        print(f"run.seq_par.first = {run.seq_par.first} run.seq_par.last = {run.seq_par.last}")

        track_forward_start(run)
        trackcorr_c_loop(run, run.seq_par.first)

        for step in range(run.seq_par.first + 1, run.seq_par.last):
            # print(f"step = {step}")
            trackcorr_c_loop(run, step)
            # print(f"run.npart = {run.npart}")
            # print(f"run.nlinks = {run.nlinks}")

        trackcorr_c_finish(run, run.seq_par.last)


        range_val = run.seq_par.last - run.seq_par.first
        npart = run.npart / range_val
        nlinks = run.nlinks / range_val

        self.assertAlmostEqual(npart, 0.9, delta=EPS)
        self.assertAlmostEqual(nlinks, 0.8, delta=EPS)

        remove_directory("res/")
        remove_directory("img/")

        os.chdir(current_directory)

    def test_trackcorr_add(self):
        """Test tracking without adding particles."""
        # import os

        current_directory = Path.cwd()
        print(f"working from {current_directory}")
        directory = Path("tests/testing_fodder/track")

        os.chdir(directory)

        print(os.path.abspath(os.curdir))
        print(Path.cwd())

        if Path("res/").exists():
            remove_directory("res/")

        if Path("img/").exists():
            remove_directory("img/")


        copy_directory("res_orig/", "res/")
        copy_directory("img_orig/", "img/")

        print("----------------------------")
        print("Test tracking multiple files 2 cameras, 1 particle")
        cpar = read_control_par("parameters/ptv.par")

        calib = read_all_calibration(cpar.num_cams)
        # calib.append(Calibration())

        run = tr_new(
            "parameters/sequence.par",
            "parameters/track.par",
            "parameters/criteria.par",
            "parameters/ptv.par",
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            calib,
            10000.0,
        )

        run.seq_par.first = 10240
        run.seq_par.last = 10250
        run.tpar.add = 1

        track_forward_start(run)
        trackcorr_c_loop(run, run.seq_par.first)

        for step in range(run.seq_par.first + 1, run.seq_par.last):
            trackcorr_c_loop(run, step)

        trackcorr_c_finish(run, run.seq_par.last)

        remove_directory("res/")
        remove_directory("img/")

        range_val = run.seq_par.last - run.seq_par.first
        npart = run.npart / range_val
        nlinks = run.nlinks / range_val

        self.assertTrue(
            abs(npart - 1.0) < EPS,
            f"Was expecting npart == 208/210 but found {npart}"
        )
        self.assertTrue(
            abs(nlinks - 0.7) < EPS,
            f"Was expecting nlinks == 328/210 but found {nlinks}"
        )


        os.chdir(current_directory)

class TestTrackback(unittest.TestCase):

    def test_trackback(self):

        current_directory = Path.cwd()
        print(f"working from {current_directory}")
        directory = Path("tests/testing_fodder/track")

        os.chdir(directory)

        print(os.path.abspath(os.curdir))
        print(Path.cwd())

        if Path("res/").exists():
            remove_directory("res/")

        if Path("img/").exists():
            remove_directory("img/")


        copy_directory("res_orig/", "res/")
        copy_directory("img_orig/", "img/")

        print("----------------------------")
        print("Test tracking multiple files 2 cameras, 1 particle")
        cpar = read_control_par("parameters/ptv.par")

        calib = read_all_calibration(cpar.num_cams)
        # calib.append(Calibration())

        run = tr_new(
            "parameters/sequence.par",
            "parameters/track.par",
            "parameters/criteria.par",
            "parameters/ptv.par",
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            calib,
            10000.0,
        )

        run.seq_par.first = 10240
        run.seq_par.last = 10250
        run.tpar.add = 1

        track_forward_start(run)
        trackcorr_c_loop(run, run.seq_par.first)

        for step in range(run.seq_par.first + 1, run.seq_par.last):
            trackcorr_c_loop(run, step)

        trackcorr_c_finish(run, run.seq_par.last)


        run.tpar.dvxmin = run.tpar.dvymin = run.tpar.dvzmin = -50
        run.tpar.dvxmax = run.tpar.dvymax = run.tpar.dvzmax = 50


        run.lmax = vec_norm(
            np.r_[
            (run.tpar.dvxmin - run.tpar.dvxmax),
            (run.tpar.dvymin - run.tpar.dvymax),
            (run.tpar.dvzmin - run.tpar.dvzmax)
            ]
            )

        nlinks = trackback_c(run)

        print(f"nlinks = {nlinks}")

        # Uncomment the following lines to add assertions
        self.assertTrue(
            abs(nlinks - 1.0) < EPS,
            f"Was expecting nlinks to be 1.0 but found {nlinks}"
        )

        remove_directory("res/")
        remove_directory("img/")

        os.chdir(current_directory)

if __name__ == "__main__":
    unittest.main()
