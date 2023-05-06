import unittest

from openptv_python.calibration import Calibration
from openptv_python.correspondences import (
    match_pairs,
    Correspond,
    safely_allocate_adjacency_lists
)
from openptv_python.epi import Coord2d
from openptv_python.parameters import (
    ControlPar,
    VolumePar,
    read_control_par,
    read_volume_par
)
from openptv_python.tracking_frame_buf import (
    Frame,
)
from test_corresp import generate_test_set, correct_frame, read_all_calibration

class TestMatchPairs(unittest.TestCase):
    def setUp(self):
        self.cpar = read_control_par("tests/testing_fodder/parameters/ptv.par")
        self.vpar = read_volume_par("tests/testing_fodder/parameters/criteria.par")

        self.cpar.num_cams = 2
        self.cpar.mm.n2[0] = 1.0
        self.cpar.mm.n3 = 1.0
        
        
        # self.vpar.Zmin_lay[0] = -1
        # self.vpar.Zmin_lay[1] = -1
        # self.vpar.Zmax_lay[0] = 1
        # self.vpar.Zmax_lay[1] = 1

        self.calib = read_all_calibration(self.cpar)
        self.frm = generate_test_set(self.calib, self.cpar)

        self.corrected = correct_frame(self.frm, self.calib, self.cpar, 0.0001)
        
        # initialize test inputs
        self.corr_lists = safely_allocate_adjacency_lists(self.cpar.num_cams, self.frm.num_targets)

    def test_match_pairs(self):
        """ Test that match_pairs() correctly fills in the Correspond objects in corr_lists """
        match_pairs(self.corr_lists, self.corrected, self.frm, self.vpar, self.cpar, self.calib)

        # check that all Correspond objects in corr_lists have been filled in with data
        for i1 in range(self.cpar.num_cams - 1):
            for i2 in range(i1 + 1, self.cpar.num_cams):
                for i in range(self.frm.num_targets[i1]):
                    for j in range(len(self.corr_lists[i1][i2][i].p2)):
                        self.assertIsNotNone(self.corr_lists[i1][i2][i].p2[j])
                        self.assertIsNotNone(self.corr_lists[i1][i2][i].corr[j])
                        self.assertIsNotNone(self.corr_lists[i1][i2][i].dist[j])
                        self.assertIsInstance(self.corr_lists[i1][i2][i].p2[j], int)
                        self.assertIsInstance(self.corr_lists[i1][i2][i].corr[j], float)
                        self.assertIsInstance(self.corr_lists[i1][i2][i].dist[j], float)

    def test_match_pairs_empty_input(self):
        """ Test that match_pairs() does not add any Correspond objects to corr_lists when the inputs are empty """
        empty_corr_lists = []
        empty_corrected = []
        empty_frm = Frame(num_cams=1, max_targets=10)
        empty_vpar = VolumePar()
        empty_cpar = ControlPar(num_cams=1)
        empty_calib = []

        match_pairs(empty_corr_lists, empty_corrected, empty_frm, empty_vpar, empty_cpar, empty_calib)

        # check that no Correspond objects were added to corr_lists
        self.assertEqual(len(empty_corr_lists), 0)

    def test_match_pairs_single_camera(self):
        """ Test that match_pairs() does not add any Correspond objects to corr_lists when there is only one camera """
        # test with only one camera
        corr_lists = [[[Correspond() for _ in range(1)] for _ in range(1)] for _ in range(1)]
        corrected = [[Coord2d(1.0, 2.0) for _ in range(1)] for _ in range(1)]
        frm = Frame(num_cams=1, max_targets=1)
        vpar = VolumePar()
        cpar = ControlPar(num_cams=1)
        calib = [Calibration() for _ in range(1)]

        match_pairs(corr_lists, corrected, frm, vpar, cpar, calib)

        # check that no Correspond objects were added to corr_lists
        self.assertEqual(len(corr_lists[0][0]), 1)
        self.assertEqual(len(corr_lists[0][0][0].p2), 0)
        
        
        

if __name__ == "__main__":
    unittest.main()
