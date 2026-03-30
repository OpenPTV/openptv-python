import unittest
from pathlib import Path

import numpy as np

from openptv_python._native_compat import HAS_OPTV
from openptv_python._native_convert import to_native_calibration, to_native_control_par
from openptv_python.calibration import read_calibration
from openptv_python.epi import epipolar_curve
from openptv_python.imgcoord import image_coordinates
from openptv_python.multimed import init_mmlut, multimed_nlay, multimed_r_nlay
from openptv_python.orientation import point_position, point_positions
from openptv_python.parameters import (
    ControlPar,
    MultimediaPar,
    VolumePar,
    read_control_par,
    read_volume_par,
)
from openptv_python.trafo import metric_to_pixel

try:
    from optv.epipolar import epipolar_curve as native_epipolar_curve
    from optv.orientation import (
        multi_cam_point_positions as native_multi_cam_point_positions,
    )
    from optv.parameters import VolumeParams as NativeVolumeParams

    HAS_NATIVE_MULTIMEDIA_PARITY = True
except ImportError:
    native_epipolar_curve = None
    native_multi_cam_point_positions = None
    NativeVolumeParams = None
    HAS_NATIVE_MULTIMEDIA_PARITY = False


def _multimed_r_nlay_reference(cal, mm, pos):
    """Match the native C multilayer loop in a Python reference implementation."""
    if mm.n1 == 1 and mm.nlay == 1 and mm.n2[0] == 1 and mm.n3 == 1:
        return 1.0

    x, y, z = pos
    zout = z + sum(mm.d[1 : mm.nlay])
    r = float(np.linalg.norm(np.array([x - cal.ext_par.x0, y - cal.ext_par.y0])))
    rq = r
    rdiff = 0.1
    it = 0

    while abs(rdiff) > 0.001 and it < 40:
        beta1 = np.arctan(rq / (cal.ext_par.z0 - z))
        beta2 = [np.arcsin(np.sin(beta1) * mm.n1 / mm.n2[i]) for i in range(mm.nlay)]
        beta3 = np.arcsin(np.sin(beta1) * mm.n1 / mm.n3)

        rbeta = (cal.ext_par.z0 - mm.d[0]) * np.tan(beta1) - zout * np.tan(beta3)
        for layer in range(mm.nlay):
            rbeta += mm.d[layer] * np.tan(beta2[layer])

        rdiff = r - rbeta
        rq += rdiff
        it += 1

    return 1.0 if r == 0 else float(rq / r)


def _python_multi_cam_point_positions_reference(targets, mm_par, cals):
    """Reconstruct points with the original Python per-point loop."""
    num_targets = targets.shape[0]
    points = np.empty((num_targets, 3), dtype=np.float64)
    rcm = np.empty(num_targets, dtype=np.float64)

    for pt in range(num_targets):
        rcm[pt], points[pt] = point_position(targets[pt], len(cals), mm_par, cals)

    return points, rcm


def _to_native_volume_par(vpar: VolumePar):
    """Convert Python volume parameters into native VolumeParams."""
    if NativeVolumeParams is None:
        raise RuntimeError("optv VolumeParams is not available")

    native_vpar = NativeVolumeParams()
    native_vpar.set_X_lay(list(vpar.x_lay))
    native_vpar.set_Zmin_lay(list(vpar.z_min_lay))
    native_vpar.set_Zmax_lay(list(vpar.z_max_lay))
    native_vpar.set_cn(vpar.cn)
    native_vpar.set_cnx(vpar.cnx)
    native_vpar.set_cny(vpar.cny)
    native_vpar.set_csumg(vpar.csumg)
    native_vpar.set_eps0(vpar.eps0)
    native_vpar.set_corrmin(vpar.corrmin)
    return native_vpar


def _build_multilayer_case():
    """Build a deterministic multi-camera multilayer reconstruction fixture."""
    cpar = ControlPar(4).from_file(
        Path("tests/testing_folder/control_parameters/control.par")
    )
    cpar.mm.set_n1(1.0)
    cpar.mm.set_layers([1.49, 1.10], [5.0, 10.0])
    cpar.mm.set_n3(1.33)

    vpar = read_volume_par(Path("tests/testing_folder/corresp/criteria.par"))
    add_file = Path("tests/testing_folder/calibration/cam1.tif.addpar")
    calibs = [
        read_calibration(
            Path(f"tests/testing_folder/calibration/sym_cam{cam_num}.tif.ori"),
            add_file,
        )
        for cam_num in range(1, 5)
    ]

    points = np.array(
        [
            [17.0, 42.0, 0.0],
            [8.0, 36.0, -4.0],
            [-6.0, 55.0, 7.0],
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [image_coordinates(points, cal, cpar.mm) for cal in calibs],
        dtype=np.float64,
    ).transpose(1, 0, 2)

    return cpar, vpar, calibs, points, targets


class TestMultimedRnlay(unittest.TestCase):
    def setUp(self):
        filepath = Path("tests") / "testing_folder"
        ori_file = filepath / "cal" / "cam1.tif.ori"
        add_file = filepath / "cal" / "cam1.tif.addpar"
        self.cal = read_calibration(ori_file, add_file)
        self.assertIsNotNone(self.cal, "ORI or ADDPAR file reading failed")

        vol_file = filepath / "parameters" / "criteria.par"
        self.vpar = read_volume_par(vol_file)
        self.assertIsNotNone(self.vpar, "volume parameter file reading failed")

        filename = filepath / "parameters" / "ptv.par"
        self.cpar = read_control_par(filename)
        self.assertIsNotNone(self.cpar, "control parameter file reading failed")

        self.cpar.num_cams = 1

    def test_multimed_r_nlay(self):
        """Test the non-recursive version of multimed_r_nlay."""
        pos = np.array([self.cal.ext_par.x0, self.cal.ext_par.y0, 0.0])
        tmp = multimed_r_nlay(self.cal, self.cpar.mm, pos)
        self.assertAlmostEqual(tmp, 1.0)

        self.cal = init_mmlut(self.vpar, self.cpar, self.cal)

        # print("finished with init_mmlut \n")
        # print(self.cal.mmlut.nr, self.cal.mmlut.nz, self.cal.mmlut.rw)

        # Set up input position and expected output values
        pos = np.array([1.23, 1.23, 1.23])

        correct_Xq = 0.74811917
        correct_Yq = 0.75977975

        # radial_shift = multimed_r_nlay (self.cal, self.cpar.mm, pos)
        # print(f"radial shift is {radial_shift}")

        # /* if radial_shift == 1.0, this degenerates to Xq = X, Yq = Y  */
        # Xq = self.cal.ext_par.x0 + (pos[0] - self.cal.ext_par.x0) * radial_shift
        # Yq = self.cal.ext_par.y0 + (pos[1] - self.cal.ext_par.y0) * radial_shift

        # print("\n Xq = %f, Yq = %f \n" % (Xq, Yq));

        # Call function and check output values
        Xq, Yq = multimed_nlay(self.cal, self.cpar.mm, pos)
        self.assertAlmostEqual(Xq, correct_Xq, delta=1e-8)
        self.assertAlmostEqual(Yq, correct_Yq, delta=1e-8)

    def test_multimed_r_nlay_2(self):
        """Test the non-recursive version of multimed_r_nlay."""
        # Set up input position and expected output values
        pos = np.array([1.23, 1.23, 1.23])

        radial_shift = multimed_r_nlay(self.cal, self.cpar.mm, pos)
        # print(f"radial_shift = {radial_shift}")

        self.assertAlmostEqual(radial_shift, 1.0035607, delta=1e-6)
        correct_Xq = 0.8595652692
        correct_Yq = 0.8685290653

        # Call function and check output values
        Xq, Yq = multimed_nlay(self.cal, self.cpar.mm, pos)
        self.assertAlmostEqual(Xq, correct_Xq, delta=1e-6)
        self.assertAlmostEqual(Yq, correct_Yq, delta=1e-6)

    def test_multimed_r_nlay_matches_native_formula_for_multiple_layers(self):
        """Multi-layer radial shift matches the native per-layer formulation."""
        mm = MultimediaPar(
            nlay=2,
            n1=1.0,
            n2=[1.49, 1.10],
            d=[5.0, 10.0],
            n3=1.33,
        )
        pos = np.array([10.0, 15.0, 1.23], dtype=np.float64)

        observed = multimed_r_nlay(self.cal, mm, pos)
        expected = _multimed_r_nlay_reference(self.cal, mm, pos)

        self.assertAlmostEqual(observed, expected, delta=1e-10)

    def test_multimed_r_nlay_matches_reference_across_multilayer_cases(self):
        """Multi-layer radial shift stays aligned with the native loop across cases."""
        cases = [
            (
                MultimediaPar(nlay=2, n1=1.0, n2=[1.49, 1.10], d=[5.0, 10.0], n3=1.33),
                np.array([10.0, 15.0, 1.23], dtype=np.float64),
            ),
            (
                MultimediaPar(nlay=2, n1=1.0, n2=[1.49, 1.20], d=[5.0, 3.0], n3=1.33),
                np.array([1.23, 1.23, 1.23], dtype=np.float64),
            ),
            (
                MultimediaPar(
                    nlay=3,
                    n1=1.0,
                    n2=[1.49, 1.20, 1.05],
                    d=[5.0, 3.0, 1.5],
                    n3=1.33,
                ),
                np.array([12.0, -6.0, 2.5], dtype=np.float64),
            ),
        ]

        for mm, pos in cases:
            with self.subTest(mm=mm, pos=pos):
                observed = multimed_r_nlay(self.cal, mm, pos)
                expected = _multimed_r_nlay_reference(self.cal, mm, pos)
                self.assertAlmostEqual(observed, expected, delta=1e-10)

    @unittest.skipUnless(
        HAS_OPTV and HAS_NATIVE_MULTIMEDIA_PARITY,
        "optv native multimedia comparison hooks are not available",
    )
    def test_multilayer_reconstruction_matches_python_numba_and_native(self):
        """Multi-layer reconstruction agrees across Python, Numba, and optv."""
        cpar, vpar, calibs, points, targets = _build_multilayer_case()

        python_points, python_rcm = _python_multi_cam_point_positions_reference(
            targets,
            cpar.mm,
            calibs,
        )
        compiled_points, compiled_rcm = point_positions(targets, cpar.mm, calibs, vpar)

        assert native_multi_cam_point_positions is not None
        native_points, native_rcm = native_multi_cam_point_positions(
            targets,
            to_native_control_par(cpar),
            [to_native_calibration(cal) for cal in calibs],
        )

        np.testing.assert_allclose(compiled_points, python_points, atol=1e-9)
        np.testing.assert_allclose(compiled_rcm, python_rcm, atol=1e-9)
        np.testing.assert_allclose(native_points, compiled_points, atol=1e-9)
        np.testing.assert_allclose(native_rcm, compiled_rcm, atol=1e-9)

        reprojected = np.asarray(
            [image_coordinates(compiled_points, cal, cpar.mm) for cal in calibs],
            dtype=np.float64,
        ).transpose(1, 0, 2)
        self.assertLess(np.max(np.abs(reprojected - targets)), 0.25)
        self.assertTrue(np.all(np.isfinite(compiled_rcm)))

    @unittest.skipUnless(
        HAS_OPTV and HAS_NATIVE_MULTIMEDIA_PARITY,
        "optv native multimedia comparison hooks are not available",
    )
    def test_multilayer_epipolar_curve_matches_native(self):
        """Multi-layer epipolar geometry matches the native binding output."""
        cpar, vpar, calibs, points, _targets = _build_multilayer_case()
        origin_projection = image_coordinates(points[:1], calibs[0], cpar.mm)[0]
        image_point = np.array(
            metric_to_pixel(origin_projection[0], origin_projection[1], cpar)
        )

        python_line = epipolar_curve(
            image_point,
            calibs[0],
            calibs[2],
            9,
            cpar,
            vpar,
        )

        assert native_epipolar_curve is not None
        native_line = native_epipolar_curve(
            image_point,
            to_native_calibration(calibs[0]),
            to_native_calibration(calibs[2]),
            9,
            to_native_control_par(cpar),
            _to_native_volume_par(vpar),
        )

        np.testing.assert_allclose(native_line, python_line, atol=1e-6)

    def test_set_layers_updates_multimedia_layer_count(self):
        """set_layers keeps nlay in sync with the provided layer arrays."""
        mm = MultimediaPar()
        mm.set_layers([1.49, 1.10], [5.0, 10.0])

        self.assertEqual(mm.nlay, 2)
        self.assertEqual(mm.n2, [1.49, 1.10])
        self.assertEqual(mm.d, [5.0, 10.0])


if __name__ == "__main__":
    unittest.main()
