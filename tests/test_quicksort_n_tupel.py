import unittest

import numpy as np

from openptv_python.tracking_frame_buf import n_tupel_dtype, quicksort_n_tupel


class TestQuicksortNTuple(unittest.TestCase):
    def test_quicksort_n_tupel(self):
        """Test the quicksort_n_tupel function."""
        n_tupel_list = np.recarray(3, dtype=n_tupel_dtype)
        n_tupel_list[0].p = np.array([1, 2, 3, 0])
        n_tupel_list[1].p = np.array([4, 5, 6, 0])
        n_tupel_list[2].p = np.array([7, 8, 9, 0])

        n_tupel_list.corr = np.array([0.5, 1.0, 0.2])

        # expected_list = [
        #     n_tupel(p=[7, 8, 9], corr=0.2),
        #     n_tupel(p=[1, 2, 3], corr=0.5),
        #     n_tupel(p=[4, 5, 6], corr=1.0),
        # ]

        expected_list = np.recarray(3, dtype=n_tupel_dtype)
        expected_list.p = np.array([[7, 8, 9, 0], [1, 2, 3, 0], [4, 5, 6, 0]])
        expected_list.corr = np.array([0.2, 0.5, 1.0])


        actual_list = quicksort_n_tupel(n_tupel_list)

        self.assertTrue(np.all(actual_list.corr ==  expected_list.corr))
        self.assertTrue(np.all(actual_list.p == expected_list.p))

        # equivalent to :
        n_tupel_list.sort(order="corr")
        for i, j in zip(n_tupel_list, expected_list):
            self.assertEqual(i.corr, j.corr)
            self.assertTrue(np.all(i.p == j.p))


if __name__ == "__main__":
    unittest.main()
