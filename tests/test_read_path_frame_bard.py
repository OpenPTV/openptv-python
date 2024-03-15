"""Tests for the read_path_frame() function in tracking_frame_buf.py."""
import unittest

import numpy as np

from openptv_python.constants import POSI
from openptv_python.tracking_frame_buf import (
    Corres_dtype,
    compare_corres,
    compare_path_info,
    read_path_frame,
)
from openptv_python.tracking_frame_buf import Pathinfo as P


class TestReadPathFrame(unittest.TestCase):
    """Test the read_path_frame() function."""

    def test_read_path_frame_bard(self):
        """Tests the read_path_frame() function."""
        # Create a buffer for the corres structures.
        # cor_buf = [Corres() for _ in range(POSI)]
        cor_buf = np.ndarray(POSI, dtype=Corres_dtype)

        # Create a buffer for the path info structures.
        path_buf = [P() for _ in range(POSI)]

        # Create a variable for the alt_link.
        alt_link = 0

        # Correct values for particle 3.
        path_correct = P(
            x=np.array([45.219, -20.269, 25.946]),
            prev_frame=-1,
            next_frame=-2,
            prio=4,
            finaldecis=1000000.0,
            inlist=0,
        )
        for alt_link in range(POSI):
            path_correct['decis'][alt_link] = 0.0
            path_correct.linkdecis[alt_link] = -999

        corres_correct = np.array([(3, [96, 66, 26, 26])], dtype = Corres_dtype)


        # The base name of the correspondence file.
        file_base = "tests/testing_fodder/rt_is"

        # The frame number.
        frame_num = 818

        # The number of targets read.

        # Test unlinked frame:

        # Read the path frame.
        cor_buf, path_buf = read_path_frame(file_base, "", "", frame_num)

        # Check that the correct number of targets were read.
        self.assertEqual(len(cor_buf), POSI)

        # Check that the corres structure at index 2 is correct.
        self.assertEqual(cor_buf[2], corres_correct)

        # Check that the path info structure at index 2 is correct.
        # .assertEqual(path_buf[2], path_correct)

        # Test frame with links:
        path_correct['prev_frame'] = 0
        path_correct['next_frame'] = 0
        path_correct['prio'] = 0

        # Create a buffer for the path info structures.
        cor_buf = np.ndarray(POSI, dtype=Corres_dtype)
        path_buf = [P() for _ in range(POSI)]

        # The base name of the linkage file.
        linkage_base = "tests/testing_fodder/ptv_is"

        # The base name of the prio file.
        prio_base = "tests/testing_fodder/added"

        # Read the path frame.
        cor_buf, path_buf = read_path_frame(
            file_base, linkage_base, prio_base, frame_num
        )

        # Check that the correct number of targets were read.
        self.assertEqual(len(cor_buf), POSI)

        # Check that the corres structure at index 2 is correct.
        self.assertEqual(cor_buf[2], corres_correct)

        # Check that the path info structure at index 2 is correct.
        self.assertEqual(path_buf[2], path_correct)

    def test_read_path_frame_chatgpt(self):
        """Tests the read_path_frame() function."""
        cor_buf = np.ndarray(POSI, dtype=Corres_dtype)
        path_buf = [P() for _ in range(POSI)]

        # Correct values for particle 3
        path_correct = P(
            x=np.array([45.219, -20.269, 25.946]),
            prev_frame=-1,
            next_frame=-2,
            prio=4,
            finaldecis=1000000.0,
            inlist=0,
        )
        path_correct['decis'] = [0.0] * POSI
        path_correct.linkdecis = [-999] * POSI
        c_correct = np.array([(3, [96, 66, 26, 26])],dtype=Corres_dtype)



        file_base = "tests/testing_fodder/rt_is"
        frame_num = 818

        # Test unlinked frame
        cor_buf, path_buf = read_path_frame(file_base, "", "", frame_num)
        self.assertEqual(len(cor_buf), POSI)

        self.assertTrue(
            compare_corres(cor_buf[2], c_correct),
            "Got corres: %d, [%d %d %d %d]"
            % (
                cor_buf[2]['nr'],
                cor_buf[2]['p'][0],
                cor_buf[2]['p'][1],
                cor_buf[2]['p'][2],
                cor_buf[2]['p'][3],
            ),
        )
        self.assertTrue(compare_path_info(path_buf[2], path_correct))

        # Test frame with links
        path_correct['prev_frame'] = 0
        path_correct['next_frame'] = 0
        path_correct['prio'] = 0
        linkage_base = "tests/testing_fodder/ptv_is"
        prio_base = "tests/testing_fodder/added"

        cor_buf, path_buf = read_path_frame(
            file_base, linkage_base, prio_base, frame_num
        )

        self.assertEqual(len(cor_buf), POSI)
        self.assertTrue(
            compare_corres(cor_buf[2], c_correct),
            "Got corres: %d, [%d %d %d %d]"
            % (
                cor_buf[2]['nr'],
                cor_buf[2]['p'][0],
                cor_buf[2]['p'][1],
                cor_buf[2]['p'][2],
                cor_buf[2]['p'][3],
            ),
        )
        self.assertTrue(compare_path_info(path_buf[2], path_correct))


if __name__ == "__main__":
    unittest.main()
