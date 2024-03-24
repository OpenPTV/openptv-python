import unittest
from pathlib import Path

import numpy as np

from openptv_python.calibration import (
    Calibration,
)
from openptv_python.imgcoord import image_coordinates
from openptv_python.orientation import (
    external_calibration,
    full_calibration,
)
from openptv_python.parameters import OrientPar, read_control_par
from openptv_python.trafo import arr_metric_to_pixel


class TestGradientDescent(unittest.TestCase):
    # Based on the C tests in liboptv/tests/check_orientation.c

    def setUp(self):
        control_file_name = Path("tests/testing_folder/corresp/control.par")
        # self.control = ControlPar(4)
        self.control = read_control_par(control_file_name)

        self.orient_par_file_name = "tests/testing_folder/corresp/orient.par"
        self.orient_par = OrientPar().from_file(Path(self.orient_par_file_name))

        self.cal = Calibration().from_file(
            Path("tests/testing_folder/calibration/cam1.tif.ori"),
            Path("tests/testing_folder/calibration/cam1.tif.addpar"),
        )
        self.orig_cal = Calibration().from_file(
            Path("tests/testing_folder/calibration/cam1.tif.ori"),
            Path("tests/testing_folder/calibration/cam1.tif.addpar"),
        )


    def test_external_calibration(self):
        """External calibration using clicked points."""
        ref_pts = np.array(
            [
                [-40.0, -25.0, 8.0],
                [40.0, -15.0, 0.0],
                [40.0, 15.0, 0.0],
                [40.0, 0.0, 8.0],
            ]
        )

        # Fake the image points by back-projection
        targets = arr_metric_to_pixel(
            image_coordinates(ref_pts, self.cal, self.control.mm),
            self.control,
        )

        # Jigg the fake detections to give raw_orient some challenge.
        targets[:, 1] -= 0.1

        self.assertTrue(external_calibration(self.cal, ref_pts, targets, self.control))
        np.testing.assert_array_almost_equal(
            self.cal.get_angles(), self.orig_cal.get_angles(), decimal=3
        )
        np.testing.assert_array_almost_equal(
            self.cal.get_pos(), self.orig_cal.get_pos(), decimal=3
        )

    def test_full_calibration(self):
        """Full calibration using clicked points."""
        ref_pts = np.array(
            [
                a.flatten()
                for a in np.meshgrid(np.r_[-60:-30:4j], np.r_[0:15:4j], np.r_[0:15:4j])
            ]
        ).T

        # Fake the image points by back-projection
        targets = arr_metric_to_pixel(
            image_coordinates(ref_pts, self.cal, self.control.get_multimedia_params()),
            self.control,
        )

        # # Full calibration works with TargetArray objects, not NumPy.
        # targets = TargetArray(len(targets))
        # for i, trgt in enumerate(targets):
        #     trgt.set_pnr(i)
        #     trgt.set_pos(targets[i])

        # Perturb the calibration object, then compore result to original.
        self.cal.set_pos(self.cal.get_pos() + np.r_[15.0, -15.0, 15.0])
        self.cal.set_angles(self.cal.get_angles() + np.r_[-0.5, 0.5, -0.5])


        self.orient_par.ccflag=0
        self.orient_par.xhflag=0
        self.orient_par.yhflag=0
        print(f"Calibrating with the following flags: {self.orient_par}")

        _, _, _ = full_calibration(
            self.cal,
            ref_pts,
            targets,
            self.control,
            self.orient_par
            )

        np.testing.assert_array_almost_equal(
            self.cal.get_angles(), self.orig_cal.get_angles(), decimal=4
        )
        np.testing.assert_array_almost_equal(
            self.cal.get_pos(), self.orig_cal.get_pos(), decimal=3
        )

        print(f"{self.cal.get_pos()}")
        print(f"{self.cal.get_angles()}")
        print(f"{self.cal.added_par}")

        # Perturb the calibration object, then compore result to original.
        self.cal.set_pos(self.cal.get_pos() + np.r_[1.0, -1.0, 1.0])
        self.cal.set_angles(self.cal.get_angles() + np.r_[-0.1, 0.1, -0.1])

        self.orient_par.ccflag=1
        self.orient_par.xhflag=1
        self.orient_par.yhflag=1
        print(f"Calibrating with the following flags: {self.orient_par}")

        _, _, _ = full_calibration(
            self.cal,
            ref_pts,
            targets,
            self.control,
            self.orient_par
            )

        np.testing.assert_array_almost_equal(
            self.cal.get_angles(), self.orig_cal.get_angles(), decimal=4
        )
        np.testing.assert_array_almost_equal(
            self.cal.get_pos(), self.orig_cal.get_pos(), decimal=3
        )

        print(f"{self.cal.get_pos()}")
        print(f"{self.cal.get_angles()}")
        print(f"{self.cal.added_par}")

        # Perturb the calibration object, then compore result to original.
        # self.cal.set_pos(self.cal.get_pos() + np.r_[1.0, -1.0, 1.0])
        # self.cal.set_angles(self.cal.get_angles() + np.r_[-0.1, 0.1, -0.1])

        self.orient_par.ccflag=0
        self.orient_par.xhflag=0
        self.orient_par.yhflag=0
        self.orient_par.k1flag=0
        self.orient_par.k2flag=0
        self.orient_par.k3flag=0
        self.orient_par.scxflag=0
        self.orient_par.sheflag=0
        print(f"Calibrating with the following flags: {self.orient_par}")

        _, _, _ = full_calibration(
            self.cal,
            ref_pts,
            targets,
            self.control,
            self.orient_par
            )

        np.testing.assert_array_almost_equal(
            self.cal.get_angles(), self.orig_cal.get_angles(), decimal=4
        )
        np.testing.assert_array_almost_equal(
            self.cal.get_pos(), self.orig_cal.get_pos(), decimal=3
        )
        print(f"{self.cal.get_pos()}")
        print(f"{self.cal.get_angles()}")
        print(f"{self.cal.added_par}")

        self.orient_par.ccflag=0
        self.orient_par.xhflag=0
        self.orient_par.yhflag=0
        self.orient_par.k1flag=0
        self.orient_par.k2flag=0
        self.orient_par.k3flag=0
        self.orient_par.scxflag=1
        self.orient_par.sheflag=0
        print(f"Calibrating with the following flags: {self.orient_par}")

        _, _, _ = full_calibration(
            self.cal,
            ref_pts,
            targets,
            self.control,
            self.orient_par
            )

        np.testing.assert_array_almost_equal(
            self.cal.get_angles(), self.orig_cal.get_angles(), decimal=4
        )
        np.testing.assert_array_almost_equal(
            self.cal.get_pos(), self.orig_cal.get_pos(), decimal=3
        )
        print(f"{self.cal.get_pos()}")
        print(f"{self.cal.get_angles()}")
        print(f"{self.cal.added_par}")

        self.orient_par.ccflag=0
        self.orient_par.xhflag=0
        self.orient_par.yhflag=0
        self.orient_par.k1flag=0
        self.orient_par.k2flag=0
        self.orient_par.k3flag=1
        self.orient_par.scxflag=0
        self.orient_par.sheflag=0
        print(f"Calibrating with the following flags: {self.orient_par}")

        _, _, _ = full_calibration(
            self.cal,
            ref_pts,
            targets,
            self.control,
            self.orient_par
            )

        np.testing.assert_array_almost_equal(
            self.cal.get_angles(), self.orig_cal.get_angles(), decimal=4
        )
        np.testing.assert_array_almost_equal(
            self.cal.get_pos(), self.orig_cal.get_pos(), decimal=3
        )
        print(f"{self.cal.get_pos()}")
        print(f"{self.cal.get_angles()}")
        print(f"{self.cal.added_par}")

        self.orient_par.ccflag=0
        self.orient_par.xhflag=0
        self.orient_par.yhflag=0
        self.orient_par.k1flag=0
        self.orient_par.k2flag=1
        self.orient_par.k3flag=0
        self.orient_par.scxflag=0
        self.orient_par.sheflag=0
        print(f"Calibrating with the following flags: {self.orient_par}")

        _, _, _ = full_calibration(
            self.cal,
            ref_pts,
            targets,
            self.control,
            self.orient_par
            )

        np.testing.assert_array_almost_equal(
            self.cal.get_angles(), self.orig_cal.get_angles(), decimal=4
        )
        np.testing.assert_array_almost_equal(
            self.cal.get_pos(), self.orig_cal.get_pos(), decimal=3
        )
        print(f"{self.cal.get_pos()}")
        print(f"{self.cal.get_angles()}")
        print(f"{self.cal.added_par}")

        self.orient_par.ccflag=0
        self.orient_par.xhflag=0
        self.orient_par.yhflag=0
        self.orient_par.k1flag=1
        self.orient_par.k2flag=0
        self.orient_par.k3flag=0
        self.orient_par.scxflag=0
        self.orient_par.sheflag=0
        print(f"Calibrating with the following flags: {self.orient_par}")

        _, _, _ = full_calibration(
            self.cal,
            ref_pts,
            targets,
            self.control,
            self.orient_par
            )

        np.testing.assert_array_almost_equal(
            self.cal.get_angles(), self.orig_cal.get_angles(), decimal=4
        )
        np.testing.assert_array_almost_equal(
            self.cal.get_pos(), self.orig_cal.get_pos(), decimal=3
        )
        print(f"{self.cal.get_pos()}")
        print(f"{self.cal.get_angles()}")
        print(f"{self.cal.added_par}")

        self.orient_par.ccflag=0
        self.orient_par.xhflag=0
        self.orient_par.yhflag=0
        self.orient_par.k1flag=0
        self.orient_par.k2flag=0
        self.orient_par.k3flag=1
        self.orient_par.scxflag=0
        self.orient_par.sheflag=0
        self.orient_par.p1flag=1
        self.orient_par.p2flag=0

        print(f"Calibrating with the following flags: {self.orient_par}")

        _, _, _ = full_calibration(
            self.cal,
            ref_pts,
            targets,
            self.control,
            self.orient_par
            )

        np.testing.assert_array_almost_equal(
            self.cal.get_angles(), self.orig_cal.get_angles(), decimal=4
        )
        np.testing.assert_array_almost_equal(
            self.cal.get_pos(), self.orig_cal.get_pos(), decimal=3
        )
        print(f"{self.cal.get_pos()}")
        print(f"{self.cal.get_angles()}")
        print(f"{self.cal.added_par}")

        self.orient_par.ccflag=1
        self.orient_par.xhflag=0
        self.orient_par.yhflag=0
        self.orient_par.k1flag=0
        self.orient_par.k2flag=0
        self.orient_par.k3flag=0
        self.orient_par.scxflag=0
        self.orient_par.sheflag=0
        self.orient_par.p1flag=1
        self.orient_par.p2flag=1
        print(f"Calibrating with the following flags: {self.orient_par}")

        _, _, _ = full_calibration(
            self.cal,
            ref_pts,
            targets,
            self.control,
            self.orient_par
            )

        np.testing.assert_array_almost_equal(
            self.cal.get_angles(), self.orig_cal.get_angles(), decimal=4
        )
        np.testing.assert_array_almost_equal(
            self.cal.get_pos(), self.orig_cal.get_pos(), decimal=3
        )
        print(f"{self.cal.get_pos()}")
        print(f"{self.cal.get_angles()}")
        print(f"{self.cal.added_par}")

if __name__ == "__main__":
    unittest.main()
