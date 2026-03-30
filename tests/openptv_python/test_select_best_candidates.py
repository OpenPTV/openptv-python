import unittest

import numpy as np

from openptv_python.correspondences import take_best_candidates
from openptv_python.tracking_frame_buf import n_tupel_dtype


class TestTakeBestCandidates(unittest.TestCase):
    def test_candidates_selection(self):
        src = [
            ([1, 2, 3, 4], 0.8),
            ([2, 1, -1, 5], 0.9),
            ([1, -1, -1, 6], 0.7),
            ([1, 2, 3, 7], 0.95),
        ]

        src = np.array(src, dtype=n_tupel_dtype).view(np.recarray)

        num_cams = 4
        max_targets = 8  # Maximum number of targets in tusage
        tusage = np.zeros((num_cams, max_targets), dtype=np.int32)

        dst = np.recarray(len(src), dtype=n_tupel_dtype)
        dst.p = np.zeros(4)
        dst.corr = 0.0

        taken = take_best_candidates(src, dst, num_cams, tusage)

        print(f" src = {src}")
        print(f" dst = {dst}")

        self.assertEqual(taken, 2)
        self.assertTrue(np.all(dst[0].p == [1, 2, 3, 7]))
        self.assertTrue(np.all(dst[1].p == [2, 1, -1, 5]))

        # Note that now the list of dst is same length as src with untaken values as zeros

        self.assertTrue(np.all(dst[2].p == [0, 0, 0, 0]))
        self.assertTrue(np.all(dst[3].p == [0, 0, 0, 0]))
        # self.assertIsNone(dst[2].p)
        # self.assertIsNone(dst[3].p)

    def test_no_candidates_selected(self):
        src = [
            ([-1, 2, 3, 4], 0.8),
            ([1, 2, 3, 5], 0.9),
            ([-1, -1, -1, 6], 0.7),
        ]
        src = np.array(src, dtype=n_tupel_dtype).view(np.recarray)

        num_cams = 4
        max_targets = 8  # Maximum number of targets in tusage
        tusage = np.zeros((num_cams, max_targets), dtype=np.int32)
        tusage[0][1] = 1

        dst = np.recarray(len(src), dtype=n_tupel_dtype)
        taken = take_best_candidates(src, dst, num_cams, tusage)

        self.assertEqual(taken, 2)
        self.assertTrue(np.all(dst[0].p == np.array([-1, 2, 3, 4])))
        self.assertTrue(np.all(dst[1].p == np.array([-1, -1, -1, 6])))


if __name__ == "__main__":
    unittest.main()
