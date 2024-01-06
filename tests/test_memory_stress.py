import unittest

from openptv_python.correspondences import (
    safely_allocate_adjacency_lists,
)


class TestSafelyAllocateAdjacencyLists(unittest.TestCase):
    """Test the safely_allocate_adjacency_lists function."""

    def test_correct_list_size(self):
        """Test that the adjacency lists are correctly sized."""
        num_cams = 5
        target_counts = [3, 5, 2, 4, 1]
        lists = safely_allocate_adjacency_lists(num_cams, target_counts)
        self.assertEqual(len(lists), num_cams)
        for i in range(num_cams):
            self.assertEqual(len(lists[i]), num_cams)
            for j in range(num_cams):
                if i < j:
                    self.assertTrue(len(lists[i][j]) >= target_counts[i]) # recarray is one length

    def test_memory_error(self):
        """Memory stress test."""
        # available_memory = 8GB = 8 * 1024 * 1024 * 1024 bytes
        # overhead = 200MB = 200 * 1024 * 1024 bytes
        # item_size = 4 bytes (for integers)

        # max_items = (8 * 1024 * 1024 * 1024 - 200 * 1024 * 1024) // 4 = 1,995,116,800

        num_cams = 4
        target_counts = [1000, 1000, 1000, 1000]
        # with self.assertRaises(MemoryError):
        _ = safely_allocate_adjacency_lists(num_cams, target_counts)

        # target_counts = [int(1e3), int(1e3), int(1e3), int(1e10)]
        # with self.assertRaises(MemoryError):
        #     safely_allocate_adjacency_lists(num_cams, target_counts)


if __name__ == "__main__":
    unittest.main()
