import unittest

from openptv_python.tracking_frame_buf import Target, sort_crd_y


class TestQsTargetY(unittest.TestCase):
    def test_qs_target_y(self):
        test_pix = [
            [0, 0.0, -0.2, 5, 1, 2, 10, -999],
            [6, 0.2, 0.0, 10, 8, 1, 20, -999],
            [3, 0.2, 0.8, 10, 3, 3, 30, -999],
            [4, 0.4, -1.1, 10, 3, 3, 40, -999],
            [1, 0.7, -0.1, 10, 3, 3, 50, -999],
            [7, 1.2, 0.3, 10, 3, 3, 60, -999],
            [5, 10.4, 0.1, 10, 3, 3, 70, -999],
        ]

        targs = []
        for t in test_pix:
            targs.append(Target(*t))

        # sorting test_pix vertically by 'y'
        targs = sort_crd_y(targs)

        # first point should be -1.1 and the last 0.8
        self.assertAlmostEqual(targs[0].y, -1.1, places=6)
        self.assertAlmostEqual(targs[1].y, -0.2, places=6)
        self.assertAlmostEqual(targs[6].y, 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
