import unittest

from openptv_python.tracking_frame_buf import Target, compare_targets, read_targets


class TestReadTargets(unittest.TestCase):
    """Test the read_targets function."""

    def test_read_targets(self):
        """Test the read_targets function."""
        tbuf = [None, None]  # Two targets in the sample target file
        t1 = Target(0, 1127.0000, 796.0000, 13320, 111, 120, 828903, 1)
        t2 = Target(1, 796.0000, 809.0000, 13108, 113, 116, 658928, 0)

        file_base = "tests/testing_fodder/sample_"
        frame_num = 42

        tbuf = read_targets(file_base, frame_num)
        targets_read = len(tbuf)

        # print(targets_read)
        # print(tbuf)

        self.assertEqual(targets_read, 2)
        self.assertTrue(compare_targets(tbuf[0], t1))
        self.assertTrue(compare_targets(tbuf[1], t2))


if __name__ == "__main__":
    unittest.main()
