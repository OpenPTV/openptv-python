import unittest

import numpy as np

from openptv_python.calibration import Calibration
from openptv_python.correspondences import MatchedCoords, correspondences
from openptv_python.imgcoord import img_coord
from openptv_python.parameters import ControlPar, VolumePar
from openptv_python.tracking_framebuf import TargetArray, read_targets
from openptv_python.transforms import convert_arr_metric_to_pixel


class TestMatchedCoords(unittest.TestCase):
    def test_instantiate(self):
        """Creating a MatchedCoords object."""
        cal = Calibration()
        cpar = ControlPar

        cal.from_file(
            "tests/testing_folder/calibration/cam1.tif.ori",
            "tests/testing_folder/calibration/cam2.tif.addpar",
        )
        cpar.read_control_par("tests/testing_folder/corresp/control.par")
        targs = read_targets("tests/testing_folder/frame/cam1.", 333)

        mc = MatchedCoords(targs, cpar, cal)
        pos, pnr = mc.as_arrays()

        # x sorted?
        self.assertTrue(np.all(pos[1:, 0] > pos[:-1, 0]))

        # Manually verified order for the loaded data:
        np.testing.assert_array_equal(
            pnr, np.r_[6, 11, 10, 8, 1, 4, 7, 0, 2, 9, 5, 3, 12]
        )


class TestCorresp(unittest.TestCase):
    def test_full_corresp(self):
        """Full scene correspondences."""
        cpar = ControlPar
        cpar.read_control_par("tests/testing_folder/corresp/control.par")
        vpar = VolumePar()
        vpar.read_volume_par("tests/testing_folder/corresp/criteria.par")

        # Cameras are at so high angles that opposing cameras don't see each
        # other in the normal air-glass-water setting.
        cpar.get_multimedia_params().set_layers([1.0001], [1.0])
        cpar.get_multimedia_params().set_n3(1.0001)

        cals = []
        img_pts = []
        corrected = []
        for c in range(4):
            cal = Calibration()
            cal.from_file(
                "tests/testing_folder/calibration/sym_cam%d.tif.ori" % (c + 1),
                "tests/testing_folder/calibration/cam1.tif.addpar",
            )
            cals.append(cal)

            # Generate test targets.
            targs = TargetArray(16)
            for row, col in np.ndindex(4, 4):
                targ_ix = row * 4 + col
                # Avoid symmetric case:
                if c % 2:
                    targ_ix = 15 - targ_ix
                targ = targs[targ_ix]

                pos3d = 10 * np.array([[col, row, 0]], dtype=np.float64)
                pos2d = img_coord(pos3d, cal, cpar.mmlut)
                targ.set_pos(convert_arr_metric_to_pixel(pos2d, cpar)[0])

                targ.set_pnr(targ_ix)
                targ.set_pixel_counts(25, 5, 5)
                targ.set_sum_grey_value(10)

            img_pts.append(targs)
            mc = MatchedCoords(targs, cpar, cal)
            corrected.append(mc)

        _, _, num_targs = correspondences(img_pts, corrected, cals, vpar, cpar)
        self.assertEqual(num_targs, 16)

    def test_single_cam_corresp(self):
        """Single camera correspondence."""
        cpar = ControlPar
        cpar.read_control_par("tests/testing_folder/single_cam/parameters/ptv.par")
        vpar = VolumePar()
        vpar.read_volume_par("tests/testing_folder/single_cam/parameters/criteria.par")

        # Cameras are at so high angles that opposing cameras don't see each
        # other in the normal air-glass-water setting.
        cpar.get_multimedia_params().set_layers([1.0], [1.0])
        cpar.get_multimedia_params().set_n3(1.0)

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

            pos3d = 10 * np.array([[col, row, 0]], dtype=np.float64)
            pos2d = img_coord(pos3d, cal, cpar.get_multimedia_params())
            targ.set_pos(convert_arr_metric_to_pixel(pos2d, cpar)[0])

            targ.set_pnr(targ_ix)
            targ.set_pixel_counts(25, 5, 5)
            targ.set_sum_grey_value(10)

            img_pts.append(targs)
            mc = MatchedCoords(targs, cpar, cal)
            corrected.append(mc)

        _, _, num_targs = correspondences(img_pts, corrected, cals, vpar, cpar)

        self.assertEqual(num_targs, 9)


if __name__ == "__main__":
    import os

    print((os.path.abspath(os.curdir)))
    unittest.main()