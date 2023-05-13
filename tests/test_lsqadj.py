import unittest
import numpy as np
from openptv_python.lsqadj import matmul

EPS = 1e-6

from openptv_python.calibration import Exterior


class TestMatmul(unittest.TestCase):

    def test_matmul(self):
        
        b = np.array([0.0, 0.0, 0.0])
        d = np.array([[1, 2, 3, 99], [4, 5, 6, 99], [7, 8, 9, 99], [99, 99, 99, 99]])
        e = np.array([10, 11, 12, 99])
        f = np.array([0, 0, 0])
        expected = np.array([68, 167, 266])

        test_Ex = Exterior(
            0.0, 0.0, 100.0,
            0.0, 0.0, 0.0,
            np.array([[1.0, 0.2, -0.3], [0.2, 1.0, 0.0], [-0.3, 0.0, 1.0]])
        )
        
        
        b = np.array([[1.0, 0.2, -0.3], [0.2, 1.0, 0.0], [-0.3, 0.0, 1.0]])
        c = np.array([1, 1, 1])
        a = matmul(b, c, 3, 3, 1, 3, 3)
        
        assert np.allclose(a, [0.9,1.2,0.7])
        
        a = np.array([1.0, 1.0, 1.0])
        b = matmul(test_Ex.dm, a, 3, 3, 1, 3, 3)

        self.assertTrue(
            abs(b[0,0] - 0.9) < EPS and 
            abs(b[0,1] - 1.20) < EPS and 
            abs(b[0,2] - 0.700)  < EPS)

        f = matmul(d, e, 3, 3, 1, 4, 4)

        for i in range(3):
            self.assertTrue(abs(f[0,i] - expected[i]) < EPS)

if __name__ == "__main__":
    unittest.main()
