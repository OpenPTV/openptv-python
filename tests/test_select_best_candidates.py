import unittest

from openptv_python.correspondences import take_best_candidates
from openptv_python.tracking_frame_buf import n_tupel


class TestTakeBestCandidates(unittest.TestCase):
    def test_candidates_selection(self):
        src = [
            n_tupel([1, 2, 3, 4], 0.8),
            n_tupel([2, 1, -1, 5], 0.9),
            n_tupel([1, -1, -1, 6], 0.7),
            n_tupel([1, 2, 3, 7], 0.95),
        ]

        num_cams = 4
        max_targets = 8  # Maximum number of targets in tusage
        tusage = [[0] * max_targets for _ in range(num_cams)]

        dst = [n_tupel()] * len(src)
        taken = take_best_candidates(src, dst, num_cams, tusage)

        print(f" src = {src}")
        print(f" dst = {dst}")

        self.assertEqual(taken, 2)
        self.assertEqual(dst[0].p, [1, 2, 3, 7])
        self.assertEqual(dst[1].p, [2, 1, -1, 5])

        # Note that now the list of dst is same length as src with untaken values as zeros

        self.assertEqual(dst[2].p, [0, 0, 0, 0])
        self.assertEqual(dst[3].p, [0, 0, 0, 0])
        # self.assertIsNone(dst[2].p)
        # self.assertIsNone(dst[3].p)

    def test_no_candidates_selected(self):
        src = [
            n_tupel([-1, 2, 3, 4], 0.8),
            n_tupel([1, 2, 3, 5], 0.9),
            n_tupel([-1, -1, -1, 6], 0.7),
        ]
        num_cams = 4
        max_targets = 8  # Maximum number of targets in tusage
        tusage = [[0] * max_targets for _ in range(num_cams)]
        tusage[0][1] = 1

        dst = [n_tupel()] * len(src)
        taken = take_best_candidates(src, dst, num_cams, tusage)

        self.assertEqual(taken, 2)
        self.assertEqual(dst[0].p, [-1, 2, 3, 4])
        self.assertEqual(dst[1].p, [-1, -1, -1, 6])


if __name__ == "__main__":
    unittest.main()
