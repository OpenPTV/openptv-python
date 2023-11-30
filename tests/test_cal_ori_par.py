import unittest

from openptv_python.parameters import read_cal_ori_parameters


class TestCalibrationParameters(unittest.TestCase):
    """Tests for the read_cal_ori_parameters function."""

    def setUp(self):
        self.temp_file = 'tests/testing_fodder/parameters/cal_ori.par'


    # def tearDown(self):
    #     # Close and remove the temporary file
    #     self.temp_file.close()
    #     os.remove(self.temp_file.name)

    def test_read_parameters(self):
        """Test that the function returns the correct parameters."""
        parameters = read_cal_ori_parameters(self.temp_file, 4)

        self.assertEqual(parameters.fixp_name, "cal/calblock_20.txt")
        self.assertEqual(parameters.img_name, ["cal/cam1.tif",
                                               "cal/cam2.tif",
                                               "cal/cam3.tif",
                                               "cal/cam4.tif"
                                               ])
        self.assertEqual(parameters.img_ori0, ["cal/cam1.tif.ori",
                                               "cal/cam2.tif.ori",
                                               "cal/cam3.tif.ori",
                                               "cal/cam4.tif.ori"])
        self.assertEqual(parameters.tiff_flag, 1)
        self.assertEqual(parameters.pair_flag, 0)
        self.assertEqual(parameters.chfield, 0)

    def test_read_parameters_nonexistent_file(self):
        """Test that the function raises FileNotFoundError when the file does not exist."""
        with self.assertRaises(FileNotFoundError):
            read_cal_ori_parameters("nonexistent_file.par", 4)

    # def test_read_parameters_invalid_file(self):
    #     with open(self.temp_file, 'w') as invalid_file:
    #         invalid_file.write("Invalid content")

    #     with self.assertRaises(ValueError):
    #         read_cal_ori_parameters(self.temp_file, 4)

if __name__ == '__main__':
    unittest.main()
