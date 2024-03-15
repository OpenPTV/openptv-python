import math
import unittest

import numpy as np

from openptv_python.calibration import Exterior
from openptv_python.lsqadj import ata, atl, matmul

EPS = 1e-6


class TestMatmul(unittest.TestCase):
    """Test the multiplication of a matrix with a vector."""

    def test_matmul(self):
        """Test the multiplication of a matrix with a vector."""
        b = np.array([0.0, 0.0, 0.0])
        d = np.array([[1, 2, 3, 99], [4, 5, 6, 99], [7, 8, 9, 99], [99, 99, 99, 99]])
        e = np.array([10, 11, 12, 99])
        f = np.array([0, 0, 0])
        expected = np.array([68, 167, 266])

        test_Ex = Exterior.copy()
        test_Ex['z0'] = 100.0
        test_Ex['dm'] = np.array([[1.0, 0.2, -0.3], [0.2, 1.0, 0.0], [-0.3, 0.0, 1.0]])

        b = np.array([[1.0, 0.2, -0.3], [0.2, 1.0, 0.0], [-0.3, 0.0, 1.0]])
        c = np.array([1, 1, 1])
        a = np.empty(
            3,
        )

        matmul(a, b, c, 3, 3, 1, 3, 3)

        assert np.allclose(a, [0.9, 1.2, 0.7])

        a = np.array([1.0, 1.0, 1.0])
        matmul(b, test_Ex['dm'], a, 3, 3, 1, 3, 3)

        self.assertTrue(
            abs(b[0, 0] - 0.9) < EPS
            and abs(b[0, 1] - 1.20) < EPS
            and abs(b[0, 2] - 0.700) < EPS
        )

        matmul(f, d, e, 3, 3, 1, 4, 4)

        for i in range(3):
            self.assertTrue(abs(f[i] - expected[i]) < EPS)


class TestAta(unittest.TestCase):
    """test the multiplication of a transposed matrix with a matrix."""

    def test_ata(self):
        """Test the multiplication of a transposed matrix with a matrix."""
        a = np.array([[1, 0, 1], [2, 2, 4], [1, 2, 3], [2, 4, 3]])  # input
        expected = np.array(
            [[10, 14, 18], [14, 24, 26], [18, 26, 35]]
        )  # expected output

        # test for n = n_large
        m = 4  # rows
        n = 3  # columns
        n_large = 3  # columns

        b = np.zeros((n, n))
        assert np.equal(b, np.zeros((n, n))).all()

        ata(a, b, m, n, n_large)
        assert np.equal(b, expected).all()

        # test for n < n_large
        n = 2
        b = np.zeros((n, n))
        ata(a, b, m, n, n_large)
        assert np.equal(b, expected[:n, :n]).all()

        # test numpy simle submatrix multiplication
        assert np.equal(b, a[:, :n].T.dot(a[:, :n])).all()


class TestATL(unittest.TestCase):
    """test the multiplication of a transposed matrix with a vector."""

    def test_atl(self):
        """Test the multiplication of a transposed matrix with a vector."""
        a = np.array([[1, 0, 1], [2, 2, 4], [1, 2, 3], [2, 4, 3]])
        b = np.array([1, 2, 3, 4])
        u = np.zeros(
            3,
        )
        expected = np.array([16, 26, 30])

        atl(u, a, b, 4, 3, 3)

        for i in range(3):
            msg = f"wrong item [{i}] {u[i]} instead of {expected[i]}"
            self.assertTrue(math.isclose(u[i], expected[i], rel_tol=EPS), msg)


if __name__ == "__main__":
    unittest.main()
