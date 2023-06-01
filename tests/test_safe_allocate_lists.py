import unittest

class Correspond:
    def __init__(self, n=0, p1=0):
        self.n = n
        self.p1 = p1

def safely_allocate_adjacency_lists(lists, num_cams, target_counts):
    error = 0

    for c1 in range(num_cams - 1):
        for c2 in range(c1 + 1, num_cams):
            if error == 0:
                lists[c1][c2] = [Correspond() for _ in range(target_counts[c1])]
                if lists[c1][c2] is None:
                    error = 1
                    continue

                for edge in range(target_counts[c1]):
                    lists[c1][c2][edge].n = 0
                    lists[c1][c2][edge].p1 = 0
            else:
                lists[c1][c2] = None

    if error == 0:
        return 1

    deallocate_adjacency_lists(lists, num_cams)
    return 0

def deallocate_adjacency_lists(lists, num_cams):
    # Implementation for deallocate_adjacency_lists() goes here
    # This function is not provided in the original code
    pass

class TestSafelyAllocateAdjacencyLists(unittest.TestCase):
    def test_safely_allocate_adjacency_lists(self):
        lists = [[[None] * 4 for _ in range(4)] for _ in range(4)]
        num_cams = 4
        target_counts = [2, 3, 1, 4]
        result = safely_allocate_adjacency_lists(lists, num_cams, target_counts)
        self.assertEqual(result, 1)

        expected_lists = [
            [
                [Correspond(), Correspond(), None, None],
                [Correspond(), Correspond(), Correspond(), None],
                [Correspond(), Correspond(), Correspond(), None],
                [Correspond(), Correspond(), Correspond(), Correspond()]
            ],
            [
                [None, Correspond(), None, None],
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None]
            ],
            [
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None]
            ],
            [
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None]
            ]
        ]

        self.assertEqual(lists, expected_lists)

if __name__ == '__main__':
    unittest.main()
