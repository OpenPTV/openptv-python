"""Unit tests for the correspondence code."""
import unittest

import numpy as np

from openptv_python.calibration import Calibration
from openptv_python.correspondences import (
    consistent_pair_matching,
    correspondences,
    match_pairs,
    py_correspondences,
    safely_allocate_adjacency_lists,
    safely_allocate_target_usage_marks,
)
from openptv_python.epi import Coord2d
from openptv_python.imgcoord import img_coord
from openptv_python.parameters import (
    ControlPar,
    read_control_par,
    read_volume_par,
)
from openptv_python.tracking_frame_buf import (
    Frame,
    MatchedCoords,
    TargetArray,
    n_tupel,
    read_targets,
)
from openptv_python.trafo import dist_to_flat, metric_to_pixel, pixel_to_metric


def read_all_calibration(cpar):
    """Read all calibration files."""
    ori_tmpl = "tests/testing_fodder/cal/sym_cam%d.tif.ori"
    added_name = "tests/testing_fodder/cal/cam1.tif.addpar"

    calib = []
    for cam in range(cpar.num_cams):
        ori_name = ori_tmpl % (cam + 1)
        cal = Calibration()
        cal.from_file(ori_name, added_name)
        calib.append(cal)

    return calib


def correct_frame(frm, calib, cpar, tol):
    """
    Perform the transition from pixel to metric to flat coordinates.

    and x-sorting as required by the correspondence code.

    Arguments:
    ---------
    frm - target information for all cameras.
    cpar - parameters of image size, pixel size etc.
    tol - tolerance parameter for iterative flattening phase, see
        trafo.h:correct_brown_affine_exact().
    """
    corrected = []
    for cam in range(cpar.num_cams):
        corrected.append([])
        for part in range(frm.num_targets[cam]):
            x, y = pixel_to_metric(
                frm.targets[cam][part].x, frm.targets[cam][part].y, cpar
            )
            x, y = dist_to_flat(x, y, calib[cam], tol)

            corrected[cam].append(Coord2d(x, y))
            corrected[cam][part].pnr = frm.targets[cam][part].pnr

        # This is expected by find_candidate()
        corrected[cam].sort(key=lambda coord: coord.x)

    return corrected


def generate_test_set(calib, cpar):
    """
    Generate data for targets on 4 cameras.

    The targets are organized on a 4x4 grid, 10 mm apart.
    """
    frm = Frame(cpar.num_cams, 16)

    # Four cameras on 4 quadrants looking down into a calibration target.
    # Calibration taken from an actual experimental setup
    for cam in range(cpar.num_cams):
        frm.num_targets[cam] = 16

        # Construct a scene representing a calibration target, generate
        # targets for it, then use them to reconstruct correspondences.
        for cpt_horz in range(4):
            for cpt_vert in range(4):
                cpt_ix = cpt_horz * 4 + cpt_vert
                if cam % 2:
                    cpt_ix = 15 - cpt_ix  # Avoid symmetric case

                targ = frm.targets[cam][cpt_ix]
                targ.pnr = cpt_ix

                tmp = np.r_[cpt_vert * 10, cpt_horz * 10, 0]
                targ.x, targ.y = img_coord(tmp, calib[cam], cpar.mm)
                targ.x, targ.y = metric_to_pixel(targ.x, targ.y, cpar)

                # These values work in check_epi, so used here too
                targ.n = 25
                targ.nx = targ.ny = 5
                targ.sumg = 10

    return frm


class TestReadControlPar(unittest.TestCase):
    """Test the read_control_par function."""

    def test_file_not_found(self):
        """Read a nonexistent control.par file."""
        with self.assertRaises(FileNotFoundError):
            read_control_par("nonexistent_file.txt")

    def test_valid_file(self):
        """Read a valid control.par file."""
        expected = ControlPar(num_cams=4)
        expected.img_base_name = [
            "dumbbell/cam1_Scene77_4085",
            "dumbbell/cam2_Scene77_4085",
            "dumbbell/cam3_Scene77_4085",
            "dumbbell/cam4_Scene77_4085",
        ]
        expected.cal_img_base_name = [
            "cal/cam1.tif",
            "cal/cam2.tif",
            "cal/cam3.tif",
            "cal/cam4.tif",
        ]
        expected.hp_flag = 1
        expected.allCam_flag = 0
        expected.tiff_flag = 1
        expected.imx = 1280
        expected.imy = 1024
        expected.pix_x = 0.017
        expected.pix_y = 0.017
        expected.chfield = 0
        expected.mm.n1 = 1.49
        expected.mm.n2 = 1.33
        expected.mm.n3 = 5.0
        expected.mm.d = 0.0

        result = read_control_par("tests/testing_folder/corresp/valid.par")
        self.assertEqual(result, expected)

    def test_instantiate(self):
        """Creating a MatchedCoords object."""
        cal = Calibration()
        cal.from_file(
            "tests/testing_folder/calibration/cam1.tif.ori",
            "tests/testing_folder/calibration/cam2.tif.addpar",
        )
        cpar = read_control_par("tests/testing_folder/corresp/control.par")
        targs = read_targets("tests/testing_folder/frame/cam1.", 333)

        mc = MatchedCoords(targs, cpar, cal)
        pos, pnr = mc.as_arrays()

        # x sorted?
        assert np.all(pos[1:, 0] > pos[:-1, 0])

        # Manually verified order for the loaded data:
        np.testing.assert_array_equal(
            pnr, np.r_[6, 11, 10, 8, 1, 4, 7, 0, 2, 9, 5, 3, 12]
        )

    def test_full_corresp(self):
        """Full scene correspondences."""
        cpar = read_control_par("tests/testing_folder/corresp/control.par")
        vpar = read_volume_par("tests/testing_folder/corresp/criteria.par")

        # Cameras are at so high angles that opposing cameras don't see each
        # other in the normal air-glass-water setting.
        cpar.mm.set_layers([1.0001], [1.0])
        cpar.mm.n3 = 1.0001

        cals = []
        img_pts = []
        corrected = []
        for c in range(cpar.num_cams):
            cal = Calibration()
            cal.from_file(
                f"tests/testing_folder/calibration/sym_cam{c+1:d}.tif.ori",
                "tests/testing_folder/calibration/cam1.tif.addpar",
            )
            cals.append(cal)

            # Generate test targets.
            ta = TargetArray(16)
            for row, col in np.ndindex(4, 4):
                targ_ix = row * 4 + col
                # Avoid symmetric case:
                if c % 2:
                    targ_ix = 15 - targ_ix
                targ = ta[targ_ix]

                pos3d = 10 * np.array([col, row, 0], dtype=np.float64)
                x, y = img_coord(pos3d, cal, cpar.mm)
                x, y = metric_to_pixel(x, y, cpar)
                targ.set_pos((x, y))

                targ.set_pnr(targ_ix)
                targ.set_pixel_counts(25, 5, 5)
                targ.set_sum_grey_value(10)

            img_pts.append(ta)
            mc = MatchedCoords(ta, cpar, cal)
            corrected.append(mc)

        _, _, num_targs = correspondences(img_pts, corrected, vpar, cpar, cals, mc)
        assert num_targs == 16

    def test_single_cam_corresp(self):
        """Single camera correspondence."""
        cpar = read_control_par("tests/testing_folder/single_cam/parameters/ptv.par")
        vpar = read_volume_par(
            "tests/testing_folder/single_cam/parameters/criteria.par"
        )

        # Cameras are at so high angles that opposing cameras don't see each
        # other in the normal air-glass-water setting.
        cpar.mm.set_layers([1.0], [1.0])
        cpar.n3 = 1.0

        cals = []
        img_pts = []
        corrected = []
        cal = Calibration()
        cal.from_file(
            "tests/testing_folder/single_cam/calibration/cam_1.tif.ori",
            "tests/testing_folder/single_cam/calibration/cam_1.tif.addpar",
        )
        cals.append(cal)

        # Generate test targets.
        targs = TargetArray(9)
        for row, col in np.ndindex(3, 3):
            targ_ix = row * 3 + col
            targ = targs[targ_ix]

            pos3d = 10 * np.r_[col, row, 0]
            x, y = img_coord(pos3d, cal, cpar.mm)
            x, y = metric_to_pixel(x, y, cpar)
            targ.set_pos((x, y))

            targ.set_pnr(targ_ix)
            targ.set_pixel_counts(25, 5, 5)
            targ.set_sum_grey_value(10)

        img_pts.append(targs)
        mc = MatchedCoords(targs, cpar, cal)
        corrected.append(mc)

        sorted_pos, sorted_corresp, num_targs = py_correspondences(
            img_pts, corrected, cals, vpar, cpar
        )

        self.assertEqual(len(sorted_pos), 1)  # 1 camera
        self.assertEqual(sorted_pos[0].shape, (1, 9, 2))
        np.testing.assert_array_equal(
            sorted_corresp[0][0], np.r_[6, 3, 0, 7, 4, 1, 8, 5, 2]
        )
        self.assertEqual(num_targs, 9)

    def test_two_camera_matching(self):
        """Setup is the same as the 4-camera test, targets are darkened in two cameras to get 16 pairs."""
        calib = [None] * 4
        lists = [[None] * 4 for _ in range(4)]

        cpar = read_control_par("tests/testing_fodder/parameters/ptv.par")
        vpar = read_volume_par("tests/testing_fodder/parameters/criteria.par")

        cpar.mm.n2[0] = 1.0001
        cpar.mm.n3 = 1.0001
        vpar.Zmin_lay[0] = -1
        vpar.Zmin_lay[1] = -1
        vpar.Zmax_lay[0] = 1
        vpar.Zmax_lay[1] = 1

        calib = read_all_calibration(cpar)
        frm = generate_test_set(calib, cpar)

        cpar.num_cams = 2
        corrected = correct_frame(frm, calib, cpar, 0.0001)
        # lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)
        match_pairs(corrected, frm, vpar, cpar, calib)

        con = [n_tupel() for _ in range(4 * 16)]
        tusage = safely_allocate_target_usage_marks(cpar.num_cams)

        # high accept corr bcz of closeness to epipolar lines.
        matched = consistent_pair_matching(
            lists, 2, frm.num_targets, 10000.0, con, 4 * 16, tusage
        )

        assert matched == 16

    def test_correspondences(self):
        """Test correspondences function."""
        frm = None
        match_counts = [0] * 4

        cpar = read_control_par("tests/testing_fodder/parameters/ptv.par")
        vpar = read_volume_par("tests/testing_fodder/parameters/criteria.par")

        # Cameras are at so high angles that opposing cameras don't see each other
        # in the normal air-glass-water setting.
        cpar.mm.n2[0] = 1.0001
        cpar.mm.n3 = 1.0001

        calib = read_all_calibration(cpar)
        frm = generate_test_set(calib, cpar)
        corrected = correct_frame(frm, calib, cpar, 0.0001)
        _ = correspondences(frm, corrected, vpar, cpar, calib, match_counts)

        # The example set is built to have all 16 quadruplets.
        assert match_counts[0] == 16
        assert match_counts[1] == 0
        assert match_counts[2] == 0
        assert match_counts[3] == 16  # last element is the sum of matches


class TestSafelyAllocateAdjacencyLists(unittest.TestCase):
    def test_correct_list_size(self):
        num_cams = 5
        target_counts = [3, 5, 2, 4, 1]
        lists = safely_allocate_adjacency_lists(num_cams, target_counts)
        self.assertEqual(len(lists), num_cams)
        for i in range(num_cams):
            self.assertEqual(len(lists[i]), num_cams)
            for j in range(num_cams):
                if i < j:
                    self.assertEqual(len(lists[i][j]), target_counts[i])

    def test_memory_error(self):
        """Memory stress test."""
        # available_memory = 8GB = 8 * 1024 * 1024 * 1024 bytes
        # overhead = 200MB = 200 * 1024 * 1024 bytes
        # item_size = 4 bytes (for integers)

        # max_items = (8 * 1024 * 1024 * 1024 - 200 * 1024 * 1024) // 4 = 1,995,116,800

        num_cams = 4
        target_counts = [10000, 10000, 10000, 10000]
        # with self.assertRaises(MemoryError):
        _ = safely_allocate_adjacency_lists(num_cams, target_counts)

        # target_counts = [int(1e3), int(1e3), int(1e3), int(1e10)]
        # with self.assertRaises(MemoryError):
        #     safely_allocate_adjacency_lists(num_cams, target_counts)


if __name__ == "__main__":
    unittest.main()
