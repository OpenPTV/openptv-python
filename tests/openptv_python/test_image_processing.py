"""Test the image processing functions."""

import unittest

import numpy as np

import openptv_python.image_processing as image_processing
from openptv_python._native_compat import (
    HAS_NATIVE_PREPROCESS,
    native_preprocess_image,
)
from openptv_python._native_convert import to_native_control_par
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
        self.filter_hp = False
        self.control = ControlPar(4)
        self.control.set_image_size((5, 5))

    def test_arguments(self):
        """Test that the function raises errors when it should."""
        output_img = prepare_image(
            self.input_img,
            # filter_hp=self.filter_hp,
            # dim_lp=True,
        )
        assert output_img.shape == (5, 5)

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
            dim_lp=1,
            # filter_hp=self.filter_hp,
            # filter_file='',
        )

        np.testing.assert_array_equal(res, correct_res)

    @unittest.skipUnless(
        HAS_NATIVE_PREPROCESS,
        "optv native preprocess_image is not available",
    )
    def test_preprocess_image_matches_optv_output(self):
        """The Python and optv preprocessing implementations should agree."""
        input_img = np.array(
            [
                [0, 15, 30, 45, 60, 45, 30],
                [10, 80, 130, 180, 130, 80, 10],
                [20, 120, 255, 255, 255, 120, 20],
                [30, 150, 255, 255, 255, 150, 30],
                [20, 120, 255, 255, 255, 120, 20],
                [10, 80, 130, 180, 130, 80, 10],
                [0, 15, 30, 45, 60, 45, 30],
            ],
            dtype=np.uint8,
        )

        control = ControlPar(1)
        control.set_image_size((7, 7))

        python_result = image_processing.prepare_image(
            input_img,
            dim_lp=1,
            filter_hp=0,
            filter_file="",
        )
        native_result = native_preprocess_image(
            input_img,
            0,
            to_native_control_par(control),
            lowpass_dim=1,
            filter_file=None,
            output_img=None,
        )

        np.testing.assert_array_equal(native_result, python_result)


if __name__ == "__main__":
    unittest.main()
