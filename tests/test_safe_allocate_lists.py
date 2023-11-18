import unittest

from openptv_python.correspondences import safely_allocate_adjacency_lists

# class Correspond:
#     def __init__(self, n=0, p1=0):
#         self.n = n
#         self.p1 = p1


# def safely_allocate_adjacency_lists(lists, num_cams, target_counts):
#     error = 0

#     for c1 in range(num_cams - 1):
#         for c2 in range(c1 + 1, num_cams):
#             if error == 0:
#                 lists[c1][c2] = [Correspond() for _ in range(target_counts[c1])]
#                 if lists[c1][c2] is None:
#                     error = 1
#                     continue

#                 for edge in range(target_counts[c1]):
#                     lists[c1][c2][edge].n = 0
#                     lists[c1][c2][edge].p1 = 0
#             else:
#                 lists[c1][c2] = None

#     if error == 0:
#         return 1

#     return 0


class TestSafelyAllocateAdjacencyLists(unittest.TestCase):
    """Test the safely_allocate_adjacency_lists function."""

    def test_allocation_success(self):
        """Test that the adjacency lists are allocated correctly."""
        num_cams = 4
        # lists = [[None for _ in range(num_cams)] for _ in range(num_cams)]
        target_counts = [1, 2, 3, 1]
        lists = safely_allocate_adjacency_lists(num_cams, target_counts)

        if lists is None:
            self.fail("lists is None")

        for i, row in enumerate(lists):
            for col in row:
                if col:
                    for item in range(target_counts[i]):
                        self.assertEqual(col[item].n, 0)
                        self.assertEqual(col[item].p1, 0)
                    # print(lists[row][col])

        # self.assertEqual(lists, expected_lists)

        lists = safely_allocate_adjacency_lists(0, [0, 0])
        if len(lists) > 0:
            self.fail("lists is not empty")


if __name__ == "__main__":
    unittest.main()
