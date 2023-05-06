import unittest
from openptv_python.imgcoord import flat_image_coord
from openptv_python.vec_utils import vec3d, vec_set
from openptv_python.calibration import Calibration, Exterior, Interior, Glass, ap_52
from openptv_python.parameters import MultimediaPar as mm_np

class TestFlatCenteredCam(unittest.TestCase):

    def test_flat_centered_cam(self):
        # When the image plane is centered on the axis. and the camera looks to
        # a straight angle (e.g. along an axis), the image position can be 
        # gleaned from simple geometry.
        pos = vec_set(10, 5, -20)
        cal = Calibration(
            ext_par=Exterior(0, 0, 40, 0, 0, 0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            int_par=Interior(0, 0, 10),
            glass_par=Glass(0, 0, 20),
            added_par=ap_52(0, 0, 0, 0, 0, 1, 0)
        )
        mm = mm_np(  # All in air, simplest case.
            nlay=1,
            n1=1,
            n2=[1, 0, 0],
            n3=1,
            d=[1, 0, 0]
        )
        
        # Output variables
        x, y = flat_image_coord(pos, cal, mm)
        self.assertAlmostEqual(x, 10/6.)
        self.assertAlmostEqual(x, 2*y)

if __name__ == '__main__':
    unittest.main()
