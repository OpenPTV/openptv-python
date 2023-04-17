import unittest

import numpy as np

from openptv_python.calibration import Calibration, read_calibration
from openptv_python.parameters import ControlPar
from openptv_python.trafo import (
    arr_metric_to_pixel,
    arr_pixel_to_metric,
    correct_brown_affine,
    dist_to_flat,
    distort_brown_affine,
    flat_to_dist,
    metric_to_pixel,
)


class testMetricPixel(unittest.TestCase):
    def test_metric_to_pixel():
        """Test the metric_to_pixel function."""
        cpar = ControlPar()
        cpar.imx = 1024
        cpar.imy = 1008
        cpar.pix_x = 0.01
        cpar.pix_y = 0.01

        xp, yp = metric_to_pixel(0.0, 0.0, cpar)
        assert xp == 512
        assert yp == 504

        xp, yp = metric_to_pixel(1.0, 0.0, cpar)
        assert xp == 612
        assert yp == 504

        xp, yp = metric_to_pixel(0.0, -1.0, cpar)
        assert xp == 512
        assert yp == 604

        output = arr_metric_to_pixel([[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]], cpar)
        assert np.allclose(output, [[512, 504], [612, 504], [512, 604]])


class AffineParams:
    def __init__(self, she, scx, k1, k2, k3, p1, p2):
        self.she = she
        self.scx = scx
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2


class TestDistFlatRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        """Check that the order of operations in converting metric flat image to.

        distorted image coordinates and vice-versa is correct.

        Distortion value in this one is kept to a level suitable (marginally)
        to an Rmax = 10 mm sensor. The allowed round-trip error is adjusted
        accordingly. Note that the higher the distortion parameter, the worse
        will the round-trip error be, at least unless we introduce more iteration.
        """
        x, y = 10.0, 10.0
        iter_eps = 1e-5

        cal = Calibration()
        cal.int_par.xh, cal.int_par.yh, cal.int_par.cc = 1.5, 1.5, 60.0
        (
            cal.added_par.k1,
            cal.added_par.k2,
            cal.added_par.k3,
            cal.added_par.p1,
            cal.added_par.p2,
            cal.added_par.csx,
            cal.added_par.she,
        ) = (1e-5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        xdist, ydist = flat_to_dist(x, y, cal)
        xflat, yflat = dist_to_flat(xdist, ydist, cal, iter_eps)

        self.assertAlmostEqual(xflat, x, delta=iter_eps)
        self.assertAlmostEqual(yflat, y, delta=iter_eps)

    def test_correct_brown_affine_exact(self):
        # Define test input parameters
        x = 10.0
        y = 20.0
        ap = AffineParams(0.1, 1.1, 0.01, 0.001, 0.0001, -0.0001, 0.0002)
        tol = 1e-6

        # Define expected output values
        x1_expected = -9.213596346852953
        y1_expected = 18.743734859657906

        # Call the function
        x1, y1 = correct_brown_affine(x, y, ap, tol)

        # Check the output values against the expected values
        assert np.isclose(x1, x1_expected, rtol=1e-6, atol=1e-6)
        assert np.isclose(y1, y1_expected, rtol=1e-6, atol=1e-6)

    def test_round_trip1(self):
        """Less basic distortion test: with radial distortion, a point.

        distorted/corrected would come back as the same point, up to floating
        point errors and an error from the short iteration.
        """
        x, y = 1.0, 1.0
        iter_eps = 1e-2
        cal = Calibration()
        ap = cal.added_par

        xres, yres = distort_brown_affine(
            x,
            y,
            ap,
        )
        xres, yres = correct_brown_affine(
            xres,
            yres,
            ap,
        )

        self.assertAlmostEqual(xres, x, delta=iter_eps)
        self.assertAlmostEqual(yres, y, delta=iter_eps)

    def test_round_trip2(self):
        """Do most basic distortion test: if there is no distortion.

        a point distorted/corrected would come back as the same point, up to
        floating point errors.
        """
        x, y = 1.0, 1.0
        xres, yres = None, None
        eps = 1e-5

        class AP:
            def __init__(self, k1, k2, p1, p2, k3, fx, fy):
                self.k1 = k1
                self.k2 = k2
                self.p1 = p1
                self.p2 = p2
                self.k3 = k3
                self.fx = fx
                self.fy = fy

        ap = AP(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)

        xres, yres = distort_brown_affine(x, y, ap)
        xres, yres = correct_brown_affine(xres, yres, ap)

        self.assertAlmostEqual(xres, x, delta=eps)
        self.assertAlmostEqual(yres, y, delta=eps)

    def test_metric_to_pixel(self):
        # input
        xc = 0.0  # [mm]
        yc = 0.0  # [mm]
        cpar = ControlPar(1)
        cpar.imx = 1024
        cpar.imy = 1008
        cpar.pix_x = 0.01
        cpar.pix_y = 0.01
        cpar.chfield = 0

        # output
        xp, yp = metric_to_pixel(xc, yc, cpar)

        # expected output
        expected_xp = 512.0
        expected_yp = 504.0

        # check that the output matches the expected output
        self.assertAlmostEqual(xp, expected_xp, delta=1e-7)
        self.assertAlmostEqual(yp, expected_yp, delta=1e-7)

        xc = 1.0
        yc = 0.0

        # output
        xp, yp = metric_to_pixel(xc, yc, cpar)

        # expected output
        expected_xp = 612.0
        expected_yp = 504.0

        # check that the output matches the expected output
        self.assertAlmostEqual(xp, expected_xp, delta=1e-6)
        self.assertAlmostEqual(yp, expected_yp, delta=1e-6)

        xc = 0.0
        yc = -1.0

        # output
        xp, yp = metric_to_pixel(xc, yc, cpar)

        # expected output
        expected_xp = 512.0
        expected_yp = 604.0

        # check that the output matches the expected output
        self.assertAlmostEqual(xp, expected_xp, delta=1e-6)
        self.assertAlmostEqual(yp, expected_yp, delta=1e-6)

    # class TestPixelToMetric(unittest.TestCase):
    def test_pixel_to_metric(self):
        """Test pixel_to_metric function."""
        # define input_vec pixel coordinates and parameters
        x_pixel = 320
        y_pixel = 240
        parameters = ControlPar(imx=640, imy=480, pix_x=0.01, pix_y=0.01)

        # expected output metric coordinates
        x_metric_expected = (x_pixel - float(parameters.imx) / 2.0) * parameters.pix_x
        y_metric_expected = (float(parameters.imy) / 2.0 - y_pixel) * parameters.pix_y

        # call the function to get actual output metric coordinates
        metric = arr_pixel_to_metric([x_pixel, y_pixel], parameters)

        # check if the actual output matches the expected output
        self.assertAlmostEqual(metric[:, 0], x_metric_expected)
        self.assertAlmostEqual(metric[:, 1], y_metric_expected)


class Test_transforms(unittest.TestCase):
    """Test the transforms module."""

    def setUp(self):
        """Set up the test fixtures."""
        self.input_control_par_file_name = (
            "tests/testing_folder/control_parameters/control.par"
        )
        self.control = ControlPar(4)
        self.control.from_file(self.input_control_par_file_name)

        self.input_ori_file_name = "tests/testing_folder/calibration/cam1.tif.ori"
        self.input_add_file_name = "tests/testing_folder/calibration/cam2.tif.addpar"

        self.calibration = Calibration()
        self.calibration = read_calibration(
            self.input_ori_file_name, self.input_add_file_name, "addpar.dat"
        )

    # def test_transforms_typecheck(self):
    #     """Transform bindings check types."""

    # irrelevant when we work with Python only
    # # Assert TypeError is raised when passing a non (n,2) shaped numpy ndarray
    # with self.assertRaises(TypeError):
    #     arr_pixel_to_metric([0, 0], self.control)
    # with self.assertRaises(TypeError):
    #     arr_pixel_to_metric([-1, -1], self.control)
    # with self.assertRaises(TypeError):
    #     arr_metric_to_pixel([0, 0], self.control)
    # with self.assertRaises(TypeError):
    #     arr_metric_to_pixel([-1, -1], self.control)

    def test_transforms_regress(self):
        """Transformed values are as before."""
        x_pixel = y_pixel = 1
        x_metric = y_metric = 1

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

        # Test when passing a list
        output = arr_pixel_to_metric([x_pixel, y_pixel], self.control)
        np.testing.assert_array_almost_equal(
            output, correct_output_pixel_to_metric[0], decimal=7
        )

        output = arr_metric_to_pixel([x_metric, y_metric], self.control)
        np.testing.assert_array_almost_equal(
            output, correct_output_metric_to_pixel[0], decimal=7
        )

        # Test when passing an array
        output = arr_pixel_to_metric(np.array([x_pixel, y_pixel]), self.control)
        np.testing.assert_array_almost_equal(
            output, correct_output_pixel_to_metric[0], decimal=7
        )
        output = np.zeros((3, 2))
        output = arr_metric_to_pixel(np.array([x_metric, y_metric]), self.control)
        np.testing.assert_array_almost_equal(
            output, correct_output_metric_to_pixel, decimal=7
        )

    def test_transforms(self):
        """Transform in well-known setup gives precalculates results."""
        cpar = ControlPar(1)
        cpar.set_image_size(1280, 1000)
        cpar.set_pixel_size(0.1, 0.1)

        metric_pos = np.array([[1.0, 1.0], [-10.0, 15.0], [20.0, -30.0]])
        pixel_pos = np.array([[650.0, 490.0], [540.0, 350.0], [840.0, 800.0]])

        np.testing.assert_array_almost_equal(
            pixel_pos, arr_metric_to_pixel(metric_pos, cpar)
        )
        np.testing.assert_array_almost_equal(
            metric_pos, arr_pixel_to_metric(pixel_pos, cpar)
        )

    def test_brown_affine_types(self):
        """Assert TypeError is raised when passing a non (n,2) shaped numpy ndarray."""
        with self.assertRaises(TypeError):
            tmp = [
                [0 for x in range(2)] for x in range(10)
            ]  # initialize a 10x2 list (but not numpy matrix)
            for item in tmp:
                correct_brown_affine(item[0], item[1], self.calibration.added_par)
        with self.assertRaises(TypeError):
            distort_brown_affine(np.empty(1), np.empty(1), self.calibration.added_par)
        with self.assertRaises(TypeError):
            for item in np.zeros((11, 2)):
                distort_brown_affine(item[0], item[1], self.calibration.added_par)

    def test_brown_affine_regress(self):
        """Test that the brown affine transform gives the same results as before."""
        input_vec = np.full((3, 2), 100.0)
        output = np.zeros((3, 2))
        correct_output_corr = [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]
        correct_output_dist = [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]

        # Test when passing an array for output
        for i in range(3):
            output[i, 0], output[i, 1] = correct_brown_affine(
                input_vec[i, 0], input_vec[i, 1], self.calibration.added_par
            )

        np.testing.assert_array_almost_equal(output, correct_output_corr, decimal=7)

        output = np.zeros((3, 2))
        for i in range(3):
            output[i, 0], output[i, 1] = distort_brown_affine(
                input_vec[i, 0], input_vec[i, 1], self.calibration.added_par
            )
        np.testing.assert_array_almost_equal(output, correct_output_dist, decimal=7)

        # # Test when NOT passing an array for output
        # output = correct_brown_affine(input_vec, self.calibration)
        # np.testing.assert_array_almost_equal(output, correct_output_corr, decimal=7)
        # output = np.zeros((3, 2))
        # output = distort_brown_affine(input_vec, self.calibration, out=None)
        # np.testing.assert_array_almost_equal(output, correct_output_dist, decimal=7)

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
        distorted = np.empty_like(ref_pos)
        for i in range(3):
            distorted[i, :] = distort_brown_affine(
                ref_pos[i, 0], ref_pos[i, 1], cal.added_par
            )

        np.testing.assert_array_almost_equal(distorted, ref_pos)

        # Some small radial distortion:
        cal.set_radial_distortion(np.r_[0.001, 0.0, 0.0])
        for i in range(3):
            distorted[i, :] = distort_brown_affine(
                ref_pos[i, 0], ref_pos[i, 1], cal.added_par
            )

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

        distorted = np.empty_like(ref_pos)
        corrected = np.empty_like(distorted)

        for i in range(3):
            distorted[i, 0], distorted[i, 1] = distort_brown_affine(
                ref_pos[i, 0], ref_pos[i, 1], cal.added_par
            )
            corrected[i, 0], corrected[i, 1] = dist_to_flat(
                distorted[i, 0], distorted[i, 1], cal, tol=1e-6
            )  # default tight tolerance
        np.testing.assert_array_almost_equal(ref_pos, corrected, decimal=6)


if __name__ == "__main__":
    unittest.main()
