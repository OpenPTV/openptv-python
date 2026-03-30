import tempfile
import unittest
from pathlib import Path

import numpy as np

from openptv_python.calibration import read_calibration, write_calibration
from openptv_python.calibration_compare import (
    compare_calibration_folders,
    format_calibration_comparison,
)


class TestCalibrationCompare(unittest.TestCase):
    def test_compare_same_folder_is_zero(self):
        cavity_cal_dir = Path("tests/testing_folder/test_cavity/cal")
        deltas = compare_calibration_folders(cavity_cal_dir, cavity_cal_dir)

        self.assertEqual(sorted(deltas.keys()), [f"cam{i}.tif" for i in range(1, 5)])
        for delta in deltas.values():
            np.testing.assert_allclose(delta.position_delta, 0.0)
            np.testing.assert_allclose(delta.angle_delta, 0.0)
            np.testing.assert_allclose(delta.primary_point_delta, 0.0)
            np.testing.assert_allclose(delta.glass_delta, 0.0)
            np.testing.assert_allclose(delta.added_par_delta, 0.0)

    def test_compare_modified_folder_reports_numeric_deltas(self):
        cavity_cal_dir = Path("tests/testing_folder/test_cavity/cal")

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            for cam_num in range(1, 5):
                cal = read_calibration(
                    cavity_cal_dir / f"cam{cam_num}.tif.ori",
                    cavity_cal_dir / f"cam{cam_num}.tif.addpar",
                )
                if cam_num == 2:
                    cal.set_pos(cal.get_pos() + np.array([1.0, -2.0, 3.0]))
                    cal.set_angles(cal.get_angles() + np.array([0.01, -0.02, 0.03]))
                    cal.added_par[0] += 1.5e-4
                    cal.added_par[3] -= 2.5e-4
                write_calibration(
                    cal,
                    tmp_dir / f"cam{cam_num}.tif.ori",
                    tmp_dir / f"cam{cam_num}.tif.addpar",
                )

            deltas = compare_calibration_folders(cavity_cal_dir, tmp_dir)
            cam2 = deltas["cam2.tif"]
            np.testing.assert_allclose(cam2.position_delta, [1.0, -2.0, 3.0])
            np.testing.assert_allclose(cam2.angle_delta, [0.01, -0.02, 0.03])
            np.testing.assert_allclose(
                cam2.added_par_delta,
                [1.5e-4, 0.0, 0.0, -2.5e-4, 0.0, 0.0, 0.0],
            )

            rendered = format_calibration_comparison(deltas)
            self.assertIn("cam2.tif:", rendered)
            self.assertIn("+1.000000000 -2.000000000 +3.000000000", rendered)
            self.assertIn("+0.000150000", rendered)


if __name__ == "__main__":
    unittest.main()
