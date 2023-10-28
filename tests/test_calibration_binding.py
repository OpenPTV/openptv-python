import filecmp
import os
import shutil
import unittest

import numpy as np

from openptv_python.calibration import (
    Calibration,
    Exterior,
    Glass,
    Interior,
    ap_52,
    compare_addpar,
    compare_calibration,
    read_calibration,
    rotation_matrix,
    write_calibration,
)


class Test_Calibration(unittest.TestCase):
    def setUp(self):
        self.input_ori_file_name = "tests/testing_folder/calibration/cam1.tif.ori"
        self.input_add_file_name = "tests/testing_folder/calibration/cam2.tif.addpar"
        self.output_directory = "tests/testing_folder/calibration/testing_output/"

        # create a temporary output directory (will be deleted by the end of test)
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        # create an instance of Calibration wrapper class
        self.cal = Calibration()

    def test_full_instantiate(self):
        pos = np.r_[1.0, 3.0, 5.0]
        angs = np.r_[2.0, 4.0, 6.0]
        prim_point = pos * 3
        rad_dist = pos * 4
        decent = pos[:2] * 5
        affine = decent * 1.5
        glass = pos * 7

        Ext = Exterior(x0=1.0, y0=3.0, z0=5.0, omega=2.0, phi=4.0, kappa=6.0)
        In = Interior(xh=3.0, yh=9.0, cc=15.0)
        G = Glass(vec_x=glass[0], vec_y=glass[1], vec_z=glass[2])
        addpar = ap_52(
            k1=rad_dist[0],
            k2=rad_dist[1],
            k3=rad_dist[2],
            p1=decent[0],
            p2=decent[1],
            scx=affine[0],
            she=affine[1],
        )

        cal = Calibration()
        cal.ext_par = Ext
        cal.int_par = In
        cal.glass_par = G
        cal.added_par = addpar
        # cal.mmlut = mmlut

        np.testing.assert_array_equal(pos, cal.get_pos())
        np.testing.assert_array_equal(angs, cal.get_angles())
        np.testing.assert_array_equal(prim_point, cal.get_primary_point())
        np.testing.assert_array_equal(rad_dist, cal.get_radial_distortion())
        np.testing.assert_array_equal(decent, cal.get_decentering())
        np.testing.assert_array_equal(affine, cal.get_affine())
        np.testing.assert_array_equal(glass, cal.get_glass_vec())

    def test_calibration_instantiation(self):
        """Filling a calibration object by reading ori files."""
        output_ori_file_name = self.output_directory + "output_ori"
        output_add_file_name = self.output_directory + "output_add"

        # Using a round-trip test.
        cal = read_calibration(self.input_ori_file_name, self.input_add_file_name)
        # print(cal)

        write_calibration(cal, output_ori_file_name, output_add_file_name)

        tmp_cal = read_calibration(output_ori_file_name, output_add_file_name)
        # print(tmp_cal)

        self.assertTrue(compare_calibration(cal, tmp_cal))

        # self.assertTrue(filecmp.cmp(self.input_ori_file_name, output_ori_file_name, 0))
        self.assertTrue(filecmp.cmp(self.input_add_file_name, output_add_file_name, 0))

        # with open(self.input_add_file_name,'r') as file1:
        #     file1_info = file1.readlines()
        # with open(output_add_file_name,'r') as file2:
        #     file2_info = file2.readlines()

        # print(file1_info)

        # diff = difflib.unified_diff(
        # file1_info, file2_info, fromfile=self.input_add_file_name,tofile=output_add_file_name, lineterm=''
        # )

        # for lines in diff:
        #     print(lines)

    def test_set_pos(self):
        """Set exterior position, only for admissible values."""
        # test set_pos() by passing a np array of 3 elements
        new_np = np.array([111.1111, 222.2222, 333.3333])
        self.cal.set_pos(new_np)

        # test getting position and assert that position is equal to set position
        np.testing.assert_array_equal(new_np, self.cal.get_pos())

        # assert set_pos() raises ValueError exception when given more or less than 3 elements
        self.assertRaises(ValueError, self.cal.set_pos, np.array([1, 2, 3, 4]))
        self.assertRaises(ValueError, self.cal.set_pos, np.array([1, 2]))

    def test_set_angles(self):
        """Set angles correctly."""
        dmatrix_before = self.cal.get_rotation_matrix()  # dmatrix before setting angles
        angles_np = np.array([0.1111, 0.2222, 0.3333])
        self.cal.set_angles(angles_np)
        # rotate the dmatrix by the angles_np
        rotation_matrix(self.cal.ext_par)

        dmatrix_after = self.cal.get_rotation_matrix()  # dmatrix after setting angles
        np.testing.assert_array_equal(self.cal.get_angles(), angles_np)

        # assert dmatrix was recalculated (before vs after)
        self.assertFalse(np.array_equal(dmatrix_before, dmatrix_after))

        self.assertRaises(ValueError, self.cal.set_angles, np.array([1, 2, 3, 4]))
        self.assertRaises(ValueError, self.cal.set_angles, np.array([1, 2]))

    def tearDown(self):
        # remove the testing output directory and its files
        shutil.rmtree(self.output_directory)

    def test_set_primary(self):
        """Set primary point (interior) position, only for admissible values."""
        new_pp = np.array([111.1111, 222.2222, 333.3333])
        self.cal.set_primary_point(new_pp)

        np.testing.assert_array_equal(new_pp, self.cal.get_primary_point())
        self.assertRaises(ValueError, self.cal.set_primary_point, np.ones(4))
        self.assertRaises(ValueError, self.cal.set_primary_point, np.ones(2))

    def test_set_radial(self):
        """Set radial distortion, only for admissible values."""
        new_rd = [111.1111, 222.2222, 333.3333]
        self.cal.set_radial_distortion(new_rd)

        np.testing.assert_array_equal(new_rd, self.cal.get_radial_distortion())
        self.assertRaises(ValueError, self.cal.set_radial_distortion, np.ones(4))
        self.assertRaises(ValueError, self.cal.set_radial_distortion, np.ones(2))

    def test_set_decentering(self):
        """Set radial distortion, only for admissible values."""
        new_de = np.array([111.1111, 222.2222])
        self.cal.set_decentering(new_de)

        np.testing.assert_array_equal(new_de, self.cal.get_decentering())
        self.assertRaises(ValueError, self.cal.set_decentering, np.ones(3))
        self.assertRaises(ValueError, self.cal.set_decentering, np.ones(1))

    def test_set_glass(self):
        """Set glass vector, only for admissible values."""
        new_gv = np.array([1.0, 2.0, 3.0])
        self.cal.set_glass_vec(new_gv)

        np.testing.assert_array_equal(new_gv, self.cal.get_glass_vec())
        self.assertRaises(ValueError, self.cal.set_glass_vec, np.ones(2))
        self.assertRaises(ValueError, self.cal.set_glass_vec, np.ones(1))


class TestCompareAddpar(unittest.TestCase):
    def test_compare_addpar(self):
        """Test compare_addpar() function."""
        a1 = ap_52(1, 2, 3, 4, 5, 6, 7)
        a2 = ap_52(1, 2, 3, 4, 5, 6, 7)
        self.assertTrue(compare_addpar(a1, a2))

        a3 = ap_52(1, 2, 3, 4, 6, 6, 7)
        self.assertFalse(compare_addpar(a1, a3))


if __name__ == "__main__":
    unittest.main()
