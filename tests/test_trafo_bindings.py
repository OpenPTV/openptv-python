import unittest

import numpy as np

from .calibration import Calibration, read_calibration
from .parameters import control_par, read_control_par
from .trafo import (
    correct_brown_affine,
    dist_to_flat,
    distort_brown_affine,
    metric_to_pixel,
    pixel_to_metric,
)


class Test_transforms(unittest.TestCase):
    def setUp(self):
        self.input_control_par_file_name = (
            b"testing_folder/control_parameters/control.par"
        )
        self.control = control_par(4)
        self.control = read_control_par(self.input_control_par_file_name)

        self.input_ori_file_name = b"testing_folder/calibration/cam1.tif.ori"
        self.input_add_file_name = b"testing_folder/calibration/cam2.tif.addpar"

        self.calibration = Calibration()
        self.calibration = read_calibration(
            self.input_ori_file_name, self.input_add_file_name, "addpar.dat"
        )

    def test_transforms_typecheck(self):
        """Transform bindings check types."""
        np.zeros((12, 2))
        # Assert TypeError is raised when passing a non (n,2) shaped numpy ndarray
        with self.assertRaises(TypeError):
            [
                [0 for x in range(2)] for x in range(10)
            ]  # initialize a 10x2 list (but not numpy matrix)
            pixel_to_metric(0, 0, self.control)
        with self.assertRaises(TypeError):
            pixel_to_metric(-1, -1, self.control)
        with self.assertRaises(TypeError):
            metric_to_pixel(0, 0, self.control)
        with self.assertRaises(TypeError):
            metric_to_pixel(-1, -1, self.control)

    def test_transforms_regress(self):
        """Transformed values are as before."""
        x_pixel = y_pixel = 1
        x_metric = y_metric = 1

        output = np.zeros((3, 2))
        correct_output_pixel_to_metric = [
            [-8181.0, 6657.92],
            [-8181.0, 6657.92],
            [-8181.0, 6657.92],
        ]
        correct_output_metric_to_pixel = [
            [646.60066007, 505.81188119],
            [646.60066007, 505.81188119],
            [646.60066007, 505.81188119],
        ]

        # Test when passing an array for output
        output = pixel_to_metric(x_pixel, y_pixel, self.control)
        np.testing.assert_array_almost_equal(
            output, correct_output_pixel_to_metric, decimal=7
        )
        output = np.zeros((3, 2))
        output = metric_to_pixel(x_metric, y_metric, self.control)
        np.testing.assert_array_almost_equal(
            output, correct_output_metric_to_pixel, decimal=7
        )

        # Test when NOT passing an array for output
        output = pixel_to_metric(x_pixel, y_pixel, self.control)
        np.testing.assert_array_almost_equal(
            output, correct_output_pixel_to_metric, decimal=7
        )
        output = np.zeros((3, 2))
        output = metric_to_pixel(x_metric, y_metric, self.control)
        np.testing.assert_array_almost_equal(
            output, correct_output_metric_to_pixel, decimal=7
        )

    def test_transforms(self):
        """Transform in well-known setup gives precalculates results."""
        cpar = control_par(1)
        cpar.set_image_size((1280, 1000))
        cpar.set_pixel_size((0.1, 0.1))

        metric_pos = np.array([[1.0, 1.0], [-10.0, 15.0], [20.0, -30.0]])
        pixel_pos = np.array([[650.0, 490.0], [540.0, 350.0], [840.0, 800.0]])

        np.testing.assert_array_almost_equal(
            pixel_pos, metric_to_pixel(metric_pos, cpar)
        )
        np.testing.assert_array_almost_equal(
            metric_pos, pixel_to_metric(pixel_pos, cpar)
        )

    def test_brown_affine_types(self):
        # Assert TypeError is raised when passing a non (n,2) shaped numpy ndarray
        with self.assertRaises(TypeError):
            list = [
                [0 for x in range(2)] for x in range(10)
            ]  # initialize a 10x2 list (but not numpy matrix)
            correct_brown_affine(list, self.calibration, out=None)
        with self.assertRaises(TypeError):
            correct_brown_affine(np.empty((10, 3)), self.calibration, out=None)
        with self.assertRaises(TypeError):
            distort_brown_affine(np.empty((2, 1)), self.calibration, out=None)
        with self.assertRaises(TypeError):
            distort_brown_affine(
                np.zeros((11, 2)), self.calibration, out=np.zeros((12, 2))
            )

    def test_brown_affine_regress(self):
        input = np.full((3, 2), 100.0)
        output = np.zeros((3, 2))
        correct_output_corr = [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]
        correct_output_dist = [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]

        # Test when passing an array for output
        correct_brown_affine(input, self.calibration, out=output)
        np.testing.assert_array_almost_equal(output, correct_output_corr, decimal=7)
        output = np.zeros((3, 2))
        distort_brown_affine(input, self.calibration, out=output)
        np.testing.assert_array_almost_equal(output, correct_output_dist, decimal=7)

        # Test when NOT passing an array for output
        output = correct_brown_affine(input, self.calibration, out=None)
        np.testing.assert_array_almost_equal(output, correct_output_corr, decimal=7)
        output = np.zeros((3, 2))
        output = distort_brown_affine(input, self.calibration, out=None)
        np.testing.assert_array_almost_equal(output, correct_output_dist, decimal=7)

    def test_brown_affine(self):
        """Distortion and correction of pixel coordinates."""
        # This is all based on values from liboptv/tests/check_imgcoord.c
        cal = Calibration()
        cal.set_pos(np.r_[0.0, 0.0, 40.0])
        cal.set_angles(np.r_[0.0, 0.0, 0.0])
        cal.set_primary_point(np.r_[0.0, 0.0, 10.0])
        cal.set_glass_vec(np.r_[0.0, 0.0, 20.0])
        cal.set_radial_distortion(np.zeros(3))
        cal.set_decentering(np.zeros(2))
        cal.set_affine_trans(np.r_[1, 0])

        # reference metric positions:
        ref_pos = np.array([[0.1, 0.1], [1.0, -1.0], [-10.0, 10.0]])

        # Perfect camera: distortion = identity.
        distorted = distort_brown_affine(ref_pos, cal)
        np.testing.assert_array_almost_equal(distorted, ref_pos)

        # Some small radial distortion:
        cal.set_radial_distortion(np.r_[0.001, 0.0, 0.0])
        distorted = distort_brown_affine(ref_pos, cal)
        self.assertTrue(np.all(abs(distorted) > abs(ref_pos)))

    def test_full_correction(self):
        """Round trip distortion/correction."""
        # This is all based on values from liboptv/tests/check_imgcoord.c
        cal = Calibration()
        cal.set_pos(np.r_[0.0, 0.0, 40.0])
        cal.set_angles(np.r_[0.0, 0.0, 0.0])
        cal.set_primary_point(np.r_[0.0, 0.0, 10.0])
        cal.set_glass_vec(np.r_[0.0, 0.0, 20.0])
        cal.set_radial_distortion(np.zeros(3))
        cal.set_decentering(np.zeros(2))
        cal.set_affine_trans(np.r_[1, 0])

        # reference metric positions:
        # Note the last value is different than in test_brown_affine() because
        # the iteration does not converge for a point too far out.
        ref_pos = np.array([[0.1, 0.1], [1.0, -1.0], [-5.0, 5.0]])

        cal.set_radial_distortion(np.r_[0.001, 0.0, 0.0])
        distorted = distort_brown_affine(ref_pos, cal)
        corrected = dist_to_flat(distorted, cal)  # default tight tolerance
        np.testing.assert_array_almost_equal(ref_pos, corrected, decimal=6)


if __name__ == "__main__":
    unittest.main()
