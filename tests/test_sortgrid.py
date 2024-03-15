import unittest
from pathlib import Path

import numpy as np

from openptv_python.calibration import read_calibration
from openptv_python.constants import SORTGRID_EPS
from openptv_python.parameters import read_control_par
from openptv_python.sortgrid import (
    nearest_neighbour_pix,
    read_calblock,
    read_sortgrid_par,
    sortgrid,
)
from openptv_python.tracking_frame_buf import Target, read_targets


class TestSortgrid(unittest.TestCase):
    def test_nearest_neighbour_pix(self):
        """Test finding the nearest neighbour pixel."""
        targets = [Target(0, 1127.0000, 796.0000, 13320, 111, 120, 828903, 1)]
        pnr = nearest_neighbour_pix(targets, 1128.0, 795.0, 0.0)
        self.assertEqual(pnr, -999)

        pnr = nearest_neighbour_pix(targets, 1128.0, 795.0, -1.0)
        self.assertEqual(pnr, -999)

        pnr = nearest_neighbour_pix(targets, -1127.0, -796.0, 1e3)
        self.assertEqual(pnr, -999)

        pnr = nearest_neighbour_pix(targets, 1127.0, 796.0, 1e-5)
        self.assertEqual(pnr, 0)

    def test_read_sortgrid_par(self):
        """Test reading sortgrid.par file."""
        eps = read_sortgrid_par("tests/testing_fodder/parameters/sortgrid.par")
        self.assertEqual(eps, 25)

        eps = read_sortgrid_par(
            "tests/testing_fodder/parameters/sortgrid_corrupted.par"
        )
        self.assertEqual(eps, SORTGRID_EPS)

    def test_read_calblock(self):
        """Test reading calblock.txt file."""
        correct_num_points = 5
        calblock_file = Path("tests/testing_fodder/cal/calblock.txt")

        # with self.assertRaises(FileNotFoundError):
        assert calblock_file.exists()

        cal_points = read_calblock(str(calblock_file))
        self.assertEqual(len(cal_points), correct_num_points)

    def test_sortgrid(self):
        """Test sorting the grid points according to the image coordinates."""
        nfix, eps, correct_eps = 5, 25, 25

        test_path = Path("tests") / "testing_fodder"
        eps = read_sortgrid_par(test_path / "parameters" / "sortgrid.par")
        self.assertEqual(eps, correct_eps)

        file_base = "tests/testing_fodder/sample_%04d"
        frame_num = 42

        targets = read_targets(file_base, frame_num)
        self.assertEqual(len(targets), 2)


        ori_file = test_path / "cal"/ "cam1.tif.ori"
        add_file = test_path / "cal" / "cam1.tif.addpar"

        cal = read_calibration(ori_file, add_file)
        cpar = read_control_par(test_path / "parameters/ptv.par")
        cal_points = read_calblock(test_path / "cal/calblock.txt")

        self.assertEqual(nfix, 5)

        # sortgrid expects only x,y,z
        # fix = np.array([vec_set(p.x, p.y, p.z) for p in cal_points])
        fix = np.c_[cal_points.x, cal_points.y, cal_points.z]

        sorted_pix = sortgrid(cal, cpar, nfix, fix, eps, targets)
        self.assertEqual(sorted_pix[0]['pnr'], -999)
        self.assertEqual(sorted_pix[1]['pnr'], -999)

        sorted_pix = sortgrid(cal, cpar, nfix, fix, 120, targets)
        self.assertEqual(sorted_pix[1]['pnr'], 1)
        self.assertEqual(sorted_pix[1].x, 796)


if __name__ == "__main__":
    unittest.main()
