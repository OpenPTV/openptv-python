import unittest

from openptv_python.calibration import Calibration, read_calibration
from openptv_python.parameters import ControlPar, read_control_par
from openptv_python.sortgrid import (
    nearest_neighbour_pix,
    read_calblock,
    read_sortgrid_par,
    sortgrid,
)
from openptv_python.tracking_frame_buf import Target, read_targets


class TestSortgrid(unittest.TestCase):
    def test_nearest_neighbour_pix(self):
        target = Target(0, 1127.0000, 796.0000, 13320, 111, 120, 828903, 1)
        pnr = nearest_neighbour_pix(target, 1, 1128.0, 795.0, 0.0)
        self.assertEqual(pnr, -999)

        pnr = nearest_neighbour_pix(target, 1, 1128.0, 795.0, -1.0)
        self.assertEqual(pnr, -999)

        pnr = nearest_neighbour_pix(target, 1, -1127.0, -796.0, 1e3)
        self.assertEqual(pnr, -999)

        pnr = nearest_neighbour_pix(target, 1, 1127.0, 796.0, 1e-5)
        self.assertEqual(pnr, 0)

    def test_read_sortgrid_par(self):
        eps = read_sortgrid_par("testing_fodder/parameters/sortgrid.par")
        self.assertEqual(eps, 25)

        eps = read_sortgrid_par("testing_fodder/parameters/sortgrid_corrupted.par")
        self.assertEqual(eps, 0)

    def test_read_calblock(self):
        num_points, correct_num_points = 5, 5
        calblock_file = "testing_fodder/cal/calblock.txt"

        with self.assertRaises(FileNotFoundError):
            calblock_file.exists()

        read_calblock(num_points, calblock_file)
        self.assertEqual(num_points, correct_num_points)

    def test_sortgrid(self):
        calibration = Calibration()
        control_par = ControlPar()
        target = Target()
        sorted_pix = Target()
        nfix, i, eps, correct_eps = 5, 0, 25, 25

        eps = read_sortgrid_par("testing_fodder/parameters/sortgrid.par")
        self.assertEqual(eps, correct_eps)

        file_base = "testing_fodder/sample_"
        frame_num = 42
        targets_read = 0

        targets_read = read_targets(file_base, frame_num)
        self.assertEqual(targets_read, 2)

        ori_file = "testing_fodder/cal/cam1.tif.ori"
        add_file = "testing_fodder/cal/cam1.tif.addpar"

        with self.assertRaises(ValueError):
            calibration = read_calibration(ori_file, add_file, None)

        with self.assertRaises(ValueError):
            control_par = read_control_par("testing_fodder/parameters/ptv.par")

        with self.assertRaises(ValueError):
            fix = read_calblock(nfix, "testing_fodder/cal/calblock.txt")

        self.assertEqual(nfix, 5)

        sorted_pix = sortgrid(
            calibration, control_par, nfix, fix, targets_read, eps, target
        )
        self.assertEqual(sorted_pix[0].pnr, -999)
        self.assertEqual(sorted_pix[1].pnr, -999)

        sorted_pix = sortgrid(
            calibration, control_par, nfix, fix, targets_read, 120, target
        )
        self.assertEqual(sorted_pix[1].pnr, 1)
        self.assertEqual(sorted_pix[1].x, 796)


if __name__ == "__main__":
    unittest.main()
