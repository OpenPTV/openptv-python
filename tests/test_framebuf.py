import os
import unittest

import numpy as np

from openptv_python.tracking_frame_buf import Frame, Target, TargetArray, read_targets


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
        tarr = TargetArray(2)
        tarr[0].set_pos((1.5, 2.5))
        tarr[1].set_pos((3.5, 4.5))

        self.assertEqual(tarr[0].pos(), (1.5, 2.5))
        self.assertEqual(tarr[1].pos(), (3.5, 4.5))

    def test_read_targets(self):
        """Reading a targets file from Python."""
        targs = read_targets("../../liboptv/tests/testing_folder/sample_", 42)

        self.assertEqual(len(targs), 2)
        self.assertEqual([targ.tnr() for targ in targs], [1, 0])
        self.assertEqual([targ.pos()[0] for targ in targs], [1127.0, 796.0])
        self.assertEqual([targ.pos()[1] for targ in targs], [796.0, 809.0])

    def test_sort_y(self):
        """Sorting on the Y coordinate in place."""
        targs = read_targets("testing_folder/frame/cam1.", 333)
        revs = read_targets("testing_folder/frame/cam1_reversed.", 333)
        revs.sort_y()

        for targ, rev in zip(targs, revs):
            self.assertTrue(targ.pos(), rev.pos())

    def test_write_targets(self):
        """Round-trip test of writing targets."""
        targs = read_targets("../../liboptv/tests/testing_folder/sample_", 42)
        targs.write("tests/testing_folder/round_trip.", 1)
        tback = read_targets("testing_folder/round_trip.", 1)

        self.assertEqual(len(targs), len(tback))
        self.assertEqual([targ.tnr() for targ in targs], [targ.tnr() for targ in tback])
        self.assertEqual(
            [targ.pos()[0] for targ in targs], [targ.pos()[0] for targ in tback]
        )
        self.assertEqual(
            [targ.pos()[1] for targ in targs], [targ.pos()[1] for targ in tback]
        )

    def tearDown(self):
        filename = "testing_folder/round_trip.0001_targets"
        if os.path.exists(filename):
            os.remove(filename)


class TestFrame(unittest.TestCase):
    def test_read_frame(self):
        """Reading a frame.

        num_cams: int = field(default_factory=int)
        max_targets: int = MAX_TARGETS
        path_info: Pathinfo | None = None
        correspond: Correspond | None = None
        targets: List[List[Target]] | None = None
        num_parts: int = 0
        num_targets: List[int] | None = None

            corres_file_base: Any,
            linkage_file_base: Any,
            prio_file_base: Any,
            target_file_base: List[Any],
            frame_num: int,

        """
        targ_files = [f"tests/testing_folder/frame/cam{c:d}." for c in range(1, 5)]
        frm = Frame(num_cams=4)
        frm.from_file(
            corres_file_base="tests/testing_folder/frame/rt_is",
            linkage_file_base="tests/testing_folder/frame/ptv_is",
            prio_file_base="tests/testing_folder/frame/added",
            target_file_base=targ_files,
            frame_num=333,
        )

        pos = frm.positions()
        self.assertEqual(pos.shape, (10, 3))

        targs = frm.target_positions_for_camera(3)
        self.assertEqual(targs.shape, (10, 2))

        targs_correct = np.array(
            [
                [426.0, 199.0],
                [429.0, 60.0],
                [431.0, 327.0],
                [509.0, 315.0],
                [345.0, 222.0],
                [465.0, 139.0],
                [487.0, 403.0],
                [241.0, 178.0],
                [607.0, 209.0],
                [563.0, 238.0],
            ]
        )
        np.testing.assert_array_equal(targs, targs_correct)


if __name__ == "__main__":
    unittest.main()
