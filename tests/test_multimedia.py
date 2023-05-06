import unittest
import numpy as np

from openptv_python.multimed import move_along_ray

class Test_Multimedia(unittest.TestCase):
    """ Test the multimedia module."""
    
    def test_move_along_ray_simple(self):
        """ Test the move_along_ray function."""
        glob_Z = 2
        vertex = np.array([1, 1, 1], dtype=np.float64)
        direct = np.array([1, 1, 1], dtype=np.float64)
        result = move_along_ray(glob_Z, vertex, direct)
        expected = np.array([2, 2, 2], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_move_along_ray_parallel(self):
        """ Test the move_along_ray function.""" 
        glob_Z = 0
        vertex = np.array([1, 1, 0], dtype=np.float64)
        direct = np.array([1, 0, 0], dtype=np.float64)
        result = move_along_ray(glob_Z, vertex, direct)
        # we move only in respect to Z
        expected = np.array([1, 1, 0], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_move_along_ray_vertical(self):
        """ Test the move_along_ray function."""
        glob_Z = 5
        vertex = np.array([0, 0, 0], dtype=np.float64)
        direct = np.array([0, 0, 1], dtype=np.float64)
        result = move_along_ray(glob_Z, vertex, direct)
        expected = np.array([0, 0, 5], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_move_along_ray_negative_Z(self):
        """ Test the move_along_ray function."""
        glob_Z = -3
        vertex = np.array([1, 1, 0], dtype=np.float64)
        direct = np.array([1, 0, 1], dtype=np.float64)
        result = move_along_ray(glob_Z, vertex, direct)
        expected = np.array([-2, 1, -3], dtype=np.float64)
        np.testing.assert_allclose(result, expected)        


if __name__ == "__main__":
    unittest.main()
