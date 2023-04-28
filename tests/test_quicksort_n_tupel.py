import unittest

from openptv_python.tracking_frame_buf import n_tupel, quicksort_n_tupel


class TestQuicksortNTuple(unittest.TestCase):
    def test_quicksort_n_tupel(self):
        """Test the quicksort_n_tupel function."""
        n_tupel_list = [
            n_tupel(p=[1, 2, 3], corr=0.5),
            n_tupel(p=[4, 5, 6], corr=1.0),
            n_tupel(p=[7, 8, 9], corr=0.2),
        ]

        expected_list = [
            n_tupel(p=[7, 8, 9], corr=0.2),
            n_tupel(p=[1, 2, 3], corr=0.5),
            n_tupel(p=[4, 5, 6], corr=1.0),
        ]

        actual_list = quicksort_n_tupel(n_tupel_list)

        self.assertEqual(actual_list, expected_list)

        # equivalent to :
        n_tupel_list.sort(key=lambda x: x.corr)
        self.assertEqual(n_tupel_list, expected_list)


if __name__ == "__main__":
    unittest.main()
