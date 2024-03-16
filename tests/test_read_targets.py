import os
import unittest

import numpy as np

from openptv_python.tracking_frame_buf import (
    Target,
    read_targets,
    write_targets,
)


class TestTargets(unittest.TestCase):
    """Test the Target class."""

    def test_fill_target(self):
        """Test filling a target."""
        t = Target.copy()
        t['pnr'] = 1
        t['tnr'] = 2
        t['x'] = 1.5
        t['y'] = 2.5
        t['n'] = 20
        t['nx'] = 4
        t['ny'] = 5
        t['sumg'] = 30



        self.assertEqual(t['pnr'], 1)
        self.assertEqual(t['tnr'], 2)
        self.assertEqual((t['x'], t['y']), (1.5, 2.5))
        self.assertEqual((t['n'],t['nx'],t['ny']), (20, 4, 5))
        self.assertEqual(t['sumg'], 30)

    def test_fill_target_array(self):
        """Test filling a target array."""
        # tarr = TargetArray(num_targets=2)

        tarr = np.tile(Target, 2)
        tarr['x'] = [1.5, 3.5]
        tarr['y'] = [2.5, 4.5]

        self.assertEqual((tarr[0]['x'],tarr[0]['y']), (1.5, 2.5))
        self.assertEqual((tarr[1]['x'],tarr[1]['y']), (3.5, 4.5))

    def test_read_targets(self):
        """Reading a targets file from Python."""
        targs = read_targets("tests/testing_folder/sample_%04d", 42)

        self.assertEqual(len(targs), 2)
        self.assertEqual([targ['tnr'] for targ in targs], [1, 0])
        self.assertEqual([targ['x'] for targ in targs], [1127.0, 796.0])
        self.assertEqual([targ['y'] for targ in targs], [796.0, 809.0])

    def test_sort_y(self):
        """Sorting on the Y coordinate in place."""
        targs = read_targets("tests/testing_folder/frame/cam1.%04d", 333)
        revs = read_targets("tests/testing_folder/frame/cam1_reversed.%04d", 333)
        revs.sort(order = 'x')

        for targ, rev in zip(targs, revs):
            self.assertTrue(targ['x'], rev['x'])
            self.assertTrue(targ['y'], rev['y'])

    def test_write_targets(self):
        """Round-trip test of writing targets."""
        targs = read_targets("tests/testing_folder/sample_%04d", 42)
        write_targets(targs, len(targs), "tests/testing_folder/round_trip.%04d", 1)
        tback = read_targets("tests/testing_folder/round_trip.%04d", 1)

        self.assertEqual(len(targs), len(tback))
        self.assertEqual([targ['tnr'] for targ in targs], [targ['tnr'] for targ in tback])
        self.assertEqual(
            [targ['x'] for targ in targs], [targ['x'] for targ in tback]
        )
        self.assertEqual(
            [targ['y'] for targ in targs], [targ['y'] for targ in tback]
        )

    def tearDown(self):
        filename = "tests/testing_folder/round_trip.0001_targets"
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    unittest.main()
