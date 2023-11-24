import os
import unittest

from openptv_python.constants import POSI
from openptv_python.tracking_frame_buf import (
    Corres,
    Pathinfo,
    Target,
    compare_corres,
    compare_path_info,
    compare_targets,
    read_path_frame,
    read_targets,
    write_targets,
)


class TestReadTargets(unittest.TestCase):
    """Test the read_targets function."""

    def test_read_targets(self):
        """Test the read_targets function."""
        tbuf = [None, None]  # Two targets in the sample target file
        t1 = Target(0, 1127.0000, 796.0000, 13320, 111, 120, 828903, 1)
        t2 = Target(1, 796.0000, 809.0000, 13108, 113, 116, 658928, 0)

        file_base = "tests/testing_fodder/sample_%04d"
        frame_num = 42

        tbuf = read_targets(file_base, frame_num)
        targets_read = len(tbuf)

        # print(targets_read)
        # print(tbuf)

        self.assertEqual(targets_read, 2)
        self.assertTrue(compare_targets(tbuf[0], t1))
        self.assertTrue(compare_targets(tbuf[1], t2))


class TestZeroTargets(unittest.TestCase):
    def test_zero_targets(self):
        """
        Zero targets should not generate an error value but just return 0.

        Reads targets from a sample file and checks that the number of targets
        read is 0.

        Arguments:
        ---------
        None

        Returns
        -------
        None
        """
        file_base = "tests/testing_fodder/sample_%04d"
        frame_num = 1
        tbuf = read_targets(file_base, frame_num)
        self.assertEqual(len(tbuf), 0)


class TestWriteTargets(unittest.TestCase):
    def test_write_targets(self):
        """Write and read back two targets, make sure they're the same.

        Assumes that read_targets is ok, which is tested separately.

        Writes two targets to a file, reads them back, and checks if they match the expected values.

        Arguments:
        ---------
        None

        Returns
        -------
        None
        """
        t1 = Target(0, 1127.0000, 796.0000, 13320, 111, 120, 828903, 1)
        t2 = Target(1, 796.0000, 809.0000, 13108, 113, 116, 658928, 0)

        file_base = "tests/testing_fodder/test_%04d"
        frame_num = 42
        num_targets = 2

        tbuf = []
        tbuf.append(t1)
        tbuf.append(t2)

        # Write targets to a file
        self.assertTrue(write_targets(tbuf, num_targets, file_base, frame_num))

        # Read back targets and check if they match
        tbuf = []
        tbuf = read_targets(file_base, frame_num)
        self.assertEqual(len(tbuf), 2)
        self.assertTrue(compare_targets(tbuf[0], t1))
        self.assertTrue(compare_targets(tbuf[1], t2))

        # Clean up the test directory
        os.remove(file_base % frame_num +"_targets")


class TestReadPathFrame(unittest.TestCase):
    def test_read_path_frame(self):
        """Test reading path frame with and without links.

        Reads a path frame without links and checks the correctness of corres and path information.
        Then, reads a path frame with links and checks again.
        """
        cor_buf = [Corres()] * 80
        path_buf = [Pathinfo()] * 80

        # Correct values for particle 3
        tmp = {
            "x": [45.219, -20.269, 25.946],
            "prev_frame": -1,
            "next_frame": -2,
            "prio": 4,
            "finaldecis": 1000000.0,
            "inlist": 0.0,
            "decis": [0.0] * POSI,
            "linkdecis": [-999] * POSI,
        }
        path_correct = Pathinfo(**tmp)

        c_correct = Corres(nr=3, p=[96, 66, 26, 26])

        file_base = "tests/testing_fodder/rt_is"
        linkage_base = "tests/testing_fodder/ptv_is"
        prio_base = "tests/testing_fodder/added"
        frame_num = 818

        # Test unlinked frame
        cor_buf, path_buf = read_path_frame(file_base, "", "", frame_num)
        targets_read = len(cor_buf)

        # for row in cor_buf:
        #     print(row.nr, row.p)

        self.assertEqual(targets_read, 80)

        self.assertTrue(
            compare_corres(cor_buf[2], c_correct),
            f"Got corres: {cor_buf[2].nr}, {cor_buf[2].p}",
        )
        self.assertTrue(compare_path_info(path_buf[2], path_correct))

        # Test frame with links
        path_correct.prev_frame = 0
        path_correct.next_frame = 0
        path_correct.prio = 0

        cor_buf, path_buf = read_path_frame(
            file_base, linkage_base, prio_base, frame_num
        )
        targets_read = len(cor_buf)
        self.assertEqual(targets_read, 80)
        self.assertTrue(
            compare_corres(cor_buf[2], c_correct),
            f"Got corres: {cor_buf[2].nr}, {cor_buf[2].p}",
        )
        self.assertTrue(compare_path_info(path_buf[2], path_correct))


if __name__ == "__main__":
    unittest.main()
