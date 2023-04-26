"""Test the image processing functions."""
import unittest

import numpy as np

from openptv_python.image_processing import prepare_image
from openptv_python.parameters import ControlPar


class Test_image_processing(unittest.TestCase):
    """Test the image processing functions."""

    def setUp(self):
        """Set up the test."""
        self.input_img = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 255, 255, 255, 0],
                [0, 255, 255, 255, 0],
                [0, 255, 255, 255, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        self.filter_hp = 0
        self.control = ControlPar(4)
        self.control.set_image_size((5, 5))

    def test_arguments(self):
        """Test that the function raises errors when it should."""
        with self.assertRaises(ValueError):
            output_img = prepare_image(
                self.input_img,
                filter_hp=self.filter_hp,
                cpar=self.control,
                dim_lp=1,
            )
            assert output_img.shape == (5, 5)

        with self.assertRaises(ValueError):
            # 3d output
            output_image = prepare_image(
                self.input_img,
                filter_hp=self.filter_hp,
                cpar=self.control,
                dim_lp=1,
            )
            assert output_image.shape == (5, 5, 1)

        with self.assertRaises(ValueError):
            # filter_hp=2 but filter_file=None
            output_img = prepare_image(
                self.input_img,
                filter_hp=2,
                cpar=self.control,
            )

    def test_preprocess_image(self):
        """Test that the function returns the correct result."""
        correct_res = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 142, 85, 142, 0],
                [0, 85, 0, 85, 0],
                [0, 142, 85, 142, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        res = prepare_image(
            self.input_img,
            filter_hp=self.filter_hp,
            cpar=self.control,
            dim_lp=1,
            filter_file=None,
        )

        np.testing.assert_array_equal(res, correct_res)


if __name__ == "__main__":
    unittest.main()
