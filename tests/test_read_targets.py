import os
import unittest

from openptv_python.tracking_frame_buf import (
    Target,
    TargetArray,
    read_targets,
    write_targets,
)


class TestTargets(unittest.TestCase):
    """Test the Target class."""

    def test_fill_target(self):
        """Test filling a target."""
        t = Target(pnr=1, tnr=2, x=1.5, y=2.5, n=20, nx=4, ny=5, sumg=30)
        self.assertEqual(t.pnr, 1)
        self.assertEqual(t.tnr, 2)
        self.assertEqual(t.pos(), (1.5, 2.5))
        self.assertEqual(t.count_pixels(), (20, 4, 5))
        self.assertEqual(t.sum_grey_value(), 30)

    def test_fill_target_array(self):
        """Test filling a target array."""
        tarr = TargetArray(num_targets=2)
        tarr[0].set_pos((1.5, 2.5))
        tarr[1].set_pos((3.5, 4.5))

        self.assertEqual(tarr[0].pos(), (1.5, 2.5))
        self.assertEqual(tarr[1].pos(), (3.5, 4.5))

    def test_read_targets(self):
        """Reading a targets file from Python."""
        targs = read_targets("tests/testing_folder/sample_%04d", 42)

        self.assertEqual(len(targs), 2)
        self.assertEqual([targ.tnr for targ in targs], [1, 0])
        self.assertEqual([targ.pos()[0] for targ in targs], [1127.0, 796.0])
        self.assertEqual([targ.pos()[1] for targ in targs], [796.0, 809.0])

    def test_sort_y(self):
        """Sorting on the Y coordinate in place."""
        targs = read_targets("tests/testing_folder/frame/cam1.%04d", 333)
        revs = read_targets("tests/testing_folder/frame/cam1_reversed.%04d", 333)
        revs.sort(key=lambda x: x.pos()[1])

        for targ, rev in zip(targs, revs):
            self.assertTrue(targ.pos(), rev.pos())

    def test_write_targets(self):
        """Round-trip test of writing targets."""
        targs = read_targets("tests/testing_folder/sample_%04d", 42)
        write_targets(targs, len(targs), "tests/testing_folder/round_trip.", 1)
        tback = read_targets("tests/testing_folder/round_trip.%04d", 1)

        self.assertEqual(len(targs), len(tback))
        self.assertEqual([targ.tnr for targ in targs], [targ.tnr for targ in tback])
        self.assertEqual(
            [targ.pos()[0] for targ in targs], [targ.pos()[0] for targ in tback]
        )
        self.assertEqual(
            [targ.pos()[1] for targ in targs], [targ.pos()[1] for targ in tback]
        )

    def tearDown(self):
        filename = "tests/testing_folder/round_trip.0001_targets"
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    unittest.main()
