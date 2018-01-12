# Regression tests for the orienbtation.

import glob
import os
import shutil
import sys
import unittest

import numpy as np

sys.path.append('../../src_c')

from ptv import py_start_proc_c, py_init_proc_c, py_prepare_eval, get_pix_crd
from ptv import py_calibration, get_xy_calib

class TestOrient(unittest.TestCase):
    def setUp(self):
        os.chdir("testing_fodder/")

        if os.path.exists("res/"):
            shutil.rmtree("res/")
        if os.path.exists("scene83_event1/"):
            shutil.rmtree("scene83_event1/")
        
    def tearDown(self):
        if os.path.exists("res/"):
            shutil.rmtree("res/")
        
        shutil.rmtree("scene83_event1/")
        
        if os.path.exists("parameters/sequence_scene.par"):
            shutil.copyfile("parameters/sequence_scene.par",
                "parameters/sequence.par")
            os.remove("parameters/sequence_scene.par")
        
        os.chdir("../")
    
    def test_prepare_eval(self):
        shutil.copytree("db_targ/", "scene83_event1/")
        shutil.copytree("db_res/", "res/")
        shutil.copyfile("parameters/sequence.par", "parameters/sequence_scene.par")
        shutil.copyfile("parameters/sequence_db.par", "parameters/sequence.par")
        
        py_init_proc_c()
        py_start_proc_c()
        
        pix, crd = py_prepare_eval(4)
        
        #np.savetxt('db_res/pix_regress.dat', pix.reshape(-1, 2))
        #np.savetxt('db_res/crd_regress.dat', crd.reshape(-1, 3))
        np.testing.assert_array_equal(pix.reshape(-1, 2),
            np.loadtxt('db_res/pix_regress.dat'))
        np.testing.assert_array_equal(crd.reshape(-1, 3),
            np.loadtxt('db_res/crd_regress.dat'))
    
    def test_ori_from_particles(self):
        """Orientation preparation in ori from particles."""
        # Note: this result set isn't great for testing shaking - it is only
        # used for testing the prepare_eval part!
        shutil.copytree("db_targ/", "scene83_event1/")
        shutil.copytree("shaking_res/", "res/")
        shutil.copyfile("parameters/sequence.par", "parameters/sequence_scene.par")
        shutil.copyfile("parameters/sequence_db.par", "parameters/sequence.par")
        
        if os.path.exists("cal_bk/"):
            shutil.rmtree("cal_bk/")
        shutil.copytree("cal/", "cal_bk/")
        
        py_init_proc_c(n_cams = 4)
        py_start_proc_c(n_cams = 4)
        
        py_calibration(10)
        pix, crd = get_pix_crd(4)

        #np.savetxt('shaking_res/pix_regress.dat', pix.reshape(-1, 2))
        #np.savetxt('shaking_res/crd_regress.dat', crd.reshape(-1, 3))
        np.testing.assert_array_almost_equal(pix.reshape(-1, 2),
            np.loadtxt('shaking_res/pix_regress.dat'))
        np.testing.assert_array_almost_equal(crd.reshape(-1, 3),
            np.loadtxt('shaking_res/crd_regress.dat'))

        shutil.rmtree("cal/")
        shutil.copytree("cal_bk/", "cal/")
        shutil.rmtree("cal_bk/")
        for file in glob.glob("safety_*"):
            os.remove(file)
        
    def test_raw_orient(self):
        """Check that raw_orient doesn't ruin the results for just_plot()."""
        shutil.copytree("db_targ/", "scene83_event1/")
        
        py_init_proc_c(n_cams = 4)
        py_start_proc_c(n_cams = 4)
        
        py_calibration(9)
        calib = get_xy_calib(4)

        #np.savetxt('calib.dat', calib.reshape(-1, 2))
        np.testing.assert_array_almost_equal(calib.reshape(-1, 2),
            np.loadtxt('calib.dat'))
        
        for file in glob.glob("safety_*"):
            os.remove(file)

if __name__ == '__main__':
    unittest.main()

