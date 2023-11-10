import os
import shutil
import unittest

import numpy
from numpy import r_

from openptv_python.parameters import (
    ControlPar,
    MultimediaPar,
    SequencePar,
    TargetPar,
    TrackPar,
    VolumePar,
    compare_sequence_par,
    read_sequence_par,
    read_target_par,
)


class Test_SequenceParams(unittest.TestCase):
    """Test SequencePar class."""

    def test_read_sequence_par(self):
        """Test read_sequence_par function."""
        """Create a test SequencePar object and write it to a temporary file."""
        # Create the SequencePar object
        expected_sp = SequencePar(
            num_cams=2,
            img_base_name=["img/cam1.", "img/cam2."],
            first=10000,
            last=10004,
        )
        # Write it to a temporary file
        filename = "tests/testing_folder/sequence_parameters/test_sequence_par.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("img/cam1.\nimg/cam2.\n10000\n10004\n")
        # Return the filename and the SequencePar object
        # return (filename, sp)

        # Get the filename and the SequencePar object from the fixture
        # filename, expected_sp = test_sequence_par
        # Call the function and check the result
        assert read_sequence_par(filename, expected_sp.num_cams) == expected_sp

    def test_compare_sequence_par(self):
        """Test compare_sequence_par function."""
        # Create two SequencePar objects with the same values
        sp1 = SequencePar(
            num_cams=2,
            img_base_name=["img/cam1.", "img/cam2."],
            first=10000,
            last=10004,
        )
        sp2 = SequencePar(
            num_cams=2,
            img_base_name=["img/cam1.", "img/cam2."],
            first=10000,
            last=10004,
        )
        # Call the function and check the result
        assert compare_sequence_par(sp1, sp2) is True

        # Create two SequencePar objects with different values
        sp3 = SequencePar(
            num_cams=2,
            img_base_name=["img/cam1.", "img/cam2."],
            first=10000,
            last=10003,
        )
        sp4 = SequencePar(
            num_cams=2,
            img_base_name=["img/cam1.", "img/cam2."],
            first=10000,
            last=10004,
        )
        # Call the function and check the result
        assert compare_sequence_par(sp3, sp4) is False


class Test_MultimediaParams(unittest.TestCase):
    """Test MultimediaPar class."""

    def test_mm_np_instantiation(self):
        """Test that MultimediaPar can be instantiated with numpy arrays."""
        n2_np = numpy.array([11, 22, 33], dtype=float)
        d_np = numpy.array([55, 66, 77])

        # Initialize MultimediaPar object (uses all setters of MultimediaPar)
        m = MultimediaPar(n1=2, n2=n2_np, d=d_np, n3=4)
        self.assertEqual(m.get_nlay(), 1)
        self.assertEqual(m.get_n1(), 2)
        self.assertEqual(m.get_n3(), 4)
        # self.assertEqual(m.get_nlay(), len(d_np))

        numpy.testing.assert_array_equal(m.get_d(), d_np)
        numpy.testing.assert_array_equal(m.get_n2(), n2_np)

        # pass two arrays with different number of elements
        new_arr = numpy.array([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            m.set_layers(new_arr, d_np)
        new_arr = numpy.array([1, 2, 3])

        arr = m.n2  # don't copy the values: link directly to memory
        arr[0] = 77.77
        arr[1] = 88.88
        arr[2] = 99.99
        # assert that the arr affected the contents of m object
        numpy.testing.assert_array_equal(m.get_n2(), [77.77, 88.88, 99.99])


class Test_TrackingParams(unittest.TestCase):
    """Test TrackingParams class.

        typedef struct
    {
        double  dacc, dangle, dvxmax, dvxmin;
        double dvymax, dvymin, dvzmax, dvzmin;
        int dsumg, dn, dnx, dny, add;
    } track


    """

    def setUp(self):
        self.input_tracking_par_file_name = (
            "tests/testing_folder/tracking_parameters/track.par"
        )

        # create an instance of TrackPar class
        # testing setters that are used in constructor
        self.track_obj1 = TrackPar(
            dacc=1.1,
            dangle=2.2,
            add=1,
            dvxmin=3.3,
            dvxmax=4.4,
            dvymin=5.5,
            dvymax=6.6,
            dvzmin=7.7,
            dvzmax=8.8,
        )

    # Testing getters according to the values passed in setUp

    def test_TrackingParams_getters(self):
        """Test getters."""
        self.assertTrue(self.track_obj1.dacc == 1.1)
        self.assertTrue(self.track_obj1.dangle == 2.2)
        self.assertTrue(self.track_obj1.dvxmin == 3.3)
        self.assertTrue(self.track_obj1.dvxmax == 4.4)
        self.assertTrue(self.track_obj1.dvymin == 5.5)
        self.assertTrue(self.track_obj1.dvymax == 6.6)
        self.assertTrue(self.track_obj1.dvzmin == 7.7)
        self.assertTrue(self.track_obj1.dvzmax == 8.8)
        self.assertTrue(self.track_obj1.add == 1)

    def test_TrackingParams_read_from_file(self):
        """Filling a TrackPar object by reading file."""
        # read tracking parameters from file
        self.track_obj1.from_file(self.input_tracking_par_file_name)

        # check that the values of track_obj1 are equal to values in tracking parameters file
        # the check is performed according to the order the parameters were read from same file
        with open(
            self.input_tracking_par_file_name, "r", encoding="utf-8"
        ) as track_file:
            self.assertTrue(self.track_obj1.get_dacc() == float(track_file.readline()))
            self.assertTrue(
                self.track_obj1.get_dangle() == float(track_file.readline())
            )
            self.assertTrue(
                self.track_obj1.get_dvxmin() == float(track_file.readline())
            )
            self.assertTrue(
                self.track_obj1.get_dvxmax() == float(track_file.readline())
            )

            self.assertTrue(
                self.track_obj1.get_dvymin() == float(track_file.readline())
            )
            self.assertTrue(
                self.track_obj1.get_dvymax() == float(track_file.readline())
            )
            self.assertTrue(
                self.track_obj1.get_dvz_min() == float(track_file.readline())
            )

            self.assertTrue(
                self.track_obj1.get_dvz_max() == float(track_file.readline())
            )
            self.assertTrue(self.track_obj1.get_add() == int(track_file.readline()))

            self.assertTrue(self.track_obj1.get_dsumg() == 0)
            self.assertTrue(self.track_obj1.get_dn() == 0)
            self.assertTrue(self.track_obj1.get_dnx() == 0)
            self.assertTrue(self.track_obj1.get_dny() == 0)


class Test_SequenceParamsC(unittest.TestCase):
    """Test SequenceParams class."""

    def setUp(self):
        self.input_sequence_par_file_name = (
            "tests/testing_folder/sequence_parameters/sequence.par"
        )

        # create an instance of SequencParams class
        self.seq_obj = SequencePar(num_cams=4)

    def test_read_sequence(self):
        """Test read_sequence_par function."""
        # Fill the SequencePar object with parameters from test file
        self.seq_obj = read_sequence_par(self.input_sequence_par_file_name, 4)

        # check that all parameters are equal to the contents of test file
        self.assertTrue(self.seq_obj.img_base_name[0] == "dumbbell/cam1_Scene77_")
        self.assertTrue(self.seq_obj.img_base_name[1] == "dumbbell/cam2_Scene77_")
        self.assertTrue(self.seq_obj.img_base_name[2] == "dumbbell/cam3_Scene77_")
        self.assertTrue(self.seq_obj.img_base_name[3] == "dumbbell/cam4_Scene77_")
        self.assertTrue(self.seq_obj.first == 497)
        self.assertTrue(self.seq_obj.last == 597)

    def test_getters_setters(self):
        """Test getters and setters."""
        cams_num = 4
        self.seq_obj = SequencePar(num_cams=cams_num)
        for cam in range(cams_num):
            newStr = str(cam) + "some string" + str(cam)
            self.seq_obj.set_img_base_name(cam, newStr)
            self.assertTrue(self.seq_obj.get_img_base_name(cam) == newStr)

        self.seq_obj.set_first(1234)
        self.assertTrue(self.seq_obj.get_first() == 1234)
        self.seq_obj.set_last(5678)
        self.assertTrue(self.seq_obj.get_last() == 5678)

    # testing __richcmp__ comparison method of SequencePar class
    def test_rich_compare(self):
        self.seq_obj2 = read_sequence_par(self.input_sequence_par_file_name, 4)

        self.seq_obj3 = read_sequence_par(self.input_sequence_par_file_name, 4)

        self.assertTrue(self.seq_obj2 == self.seq_obj3)
        self.assertFalse(self.seq_obj2 != self.seq_obj3)

        self.seq_obj2.set_first(-999)
        self.assertTrue(self.seq_obj2 != self.seq_obj3)
        self.assertFalse(self.seq_obj2 == self.seq_obj3)

    def test_full_instantiate(self):
        """Instantiate a SequencePar object from keywords."""
        spar = SequencePar(
            num_cams=2, img_base_name=["test1", "test2"], first=1, last=100
        )

        self.assertTrue(spar.img_base_name[0] == "test1")
        self.assertTrue(spar.img_base_name[1] == "test2")
        self.assertTrue(spar.first == 1)
        self.assertTrue(spar.last == 100)


class Test_VolumeParams(unittest.TestCase):
    """Test VolumePar class."""

    def setUp(self):
        """Set up for testing VolumePar class."""
        self.input_volume_par_file_name = (
            "tests/testing_folder/volume_parameters/volume.par"
        )
        self.temp_output_directory = (
            "tests/testing_folder/volume_parameters/testing_output"
        )

        # create a temporary output directory (will be deleted by the end of test)
        if not os.path.exists(self.temp_output_directory):
            os.makedirs(self.temp_output_directory)

        # create an instance of VolumePar class
        self.vol_obj = VolumePar()

    def test_read_volume(self):
        """Test reading volume parameters from file."""
        # Fill the VolumePar object with parameters from test file
        self.vol_obj.from_file(self.input_volume_par_file_name)

        # check that all parameters are equal to the contents of test file
        numpy.testing.assert_array_equal(
            numpy.array([111.111, 222.222]), self.vol_obj.X_lay
        )
        numpy.testing.assert_array_equal(
            numpy.array([333.333, 444.444]), self.vol_obj.z_min_lay
        )
        numpy.testing.assert_array_equal(
            numpy.array([555.555, 666.666]), self.vol_obj.z_max_lay
        )

        self.assertTrue(self.vol_obj.cnx == 777.777)
        self.assertTrue(self.vol_obj.cny == 888.888)
        self.assertTrue(self.vol_obj.cn == 999.999)
        self.assertTrue(self.vol_obj.csumg == 1010.1010)
        self.assertTrue(self.vol_obj.corrmin == 1111.1111)
        self.assertTrue(self.vol_obj.eps0 == 1212.1212)

    def test_setters(self):
        """Test setting volume parameters."""
        xlay = numpy.array([111.1, 222.2])
        self.vol_obj.X_lay = xlay
        numpy.testing.assert_array_equal(xlay, self.vol_obj.X_lay)

        z_min = numpy.array([333.3, 444.4])
        self.vol_obj.set_z_min_lay(z_min)
        numpy.testing.assert_array_equal(z_min, self.vol_obj.z_min_lay)

        z_max = numpy.array([555.5, 666.6])
        self.vol_obj.set_z_max_lay(z_max)
        numpy.testing.assert_array_equal(z_max, self.vol_obj.z_max_lay)

        self.vol_obj.cn = 1
        self.assertTrue(self.vol_obj.cn == 1)

        self.vol_obj.cnx = 2
        self.assertTrue(self.vol_obj.cnx == 2)

        self.vol_obj.cny = 3
        self.assertTrue(self.vol_obj.cny == 3)

        self.vol_obj.csumg = 4
        self.assertTrue(self.vol_obj.csumg == 4)

        self.vol_obj.eps0 = 5
        self.assertTrue(self.vol_obj.eps0 == 5)

        self.vol_obj.corrmin = 6
        self.assertTrue(self.vol_obj.corrmin == 6)

    def test_init_kwargs(self):
        """Initialize volume parameters with keyword arguments."""
        xlay = numpy.array([111.1, 222.2])
        zlay = [r_[333.3, 555.5], r_[444.4, 666.6]]
        z_min, z_max = list(zip(*zlay))

        vol_obj = VolumePar(
            X_lay=xlay,
            z_min_lay=z_min,
            z_max_lay=z_max,
            cn=1,
            cnx=2,
            cny=3,
            csumg=4,
            eps0=5,
            corrmin=6,
        )

        numpy.testing.assert_array_equal(xlay, vol_obj.X_lay)
        numpy.testing.assert_array_equal(z_min, vol_obj.z_min_lay)
        numpy.testing.assert_array_equal(z_max, vol_obj.z_max_lay)

        self.assertTrue(vol_obj.cn == 1)
        self.assertTrue(vol_obj.cnx == 2)
        self.assertTrue(vol_obj.cny == 3)
        self.assertTrue(vol_obj.csumg == 4)
        self.assertTrue(vol_obj.eps0 == 5)
        self.assertTrue(vol_obj.corrmin == 6)

    # testing __richcmp__ comparison method of VolumePar class
    def test_rich_compare(self):
        """Test comparison of VolumePar objects."""
        self.vol_obj2 = VolumePar()
        self.vol_obj2.from_file(self.input_volume_par_file_name)
        self.vol_obj3 = VolumePar()
        self.vol_obj3.from_file(self.input_volume_par_file_name)
        self.assertTrue(self.vol_obj2 == self.vol_obj3)
        self.assertFalse(self.vol_obj2 != self.vol_obj3)

        self.vol_obj2.set_cn(-999)
        self.assertTrue(self.vol_obj2 != self.vol_obj3)
        self.assertFalse(self.vol_obj2 == self.vol_obj3)

    def tearDown(self):
        # remove the testing output directory and its files
        shutil.rmtree(self.temp_output_directory)


class Test_ControlPar(unittest.TestCase):
    def setUp(self):
        self.input_control_par_file_name = (
            "tests/testing_folder/control_parameters/control.par"
        )
        self.temp_output_directory = (
            "tests/testing_folder/control_parameters/testing_output"
        )

        # create a temporary output directory (will be deleted by the end of test)
        if not os.path.exists(self.temp_output_directory):
            os.makedirs(self.temp_output_directory)
        # create an instance of ControlPar class
        self.cp_obj = ControlPar(4)

    def test_read_control(self):
        # Fill the ControlPar object with parameters from test file
        self.cp_obj.from_file(self.input_control_par_file_name)
        # check if all parameters are equal to the contents of test file
        self.assertTrue(self.cp_obj.img_base_name[0] == "dumbbell/cam1_Scene77_4085")
        self.assertTrue(self.cp_obj.img_base_name[1] == "dumbbell/cam2_Scene77_4085")
        self.assertTrue(self.cp_obj.img_base_name[2] == "dumbbell/cam3_Scene77_4085")
        self.assertTrue(self.cp_obj.img_base_name[3] == "dumbbell/cam4_Scene77_4085")

        self.assertTrue(self.cp_obj.cal_img_base_name[0] == "cal/cam1.tif")
        self.assertTrue(self.cp_obj.cal_img_base_name[1] == "cal/cam2.tif")
        self.assertTrue(self.cp_obj.cal_img_base_name[2] == "cal/cam3.tif")
        self.assertTrue(self.cp_obj.cal_img_base_name[3] == "cal/cam4.tif")

        self.assertTrue(self.cp_obj.get_num_cams() == 4)
        self.assertTrue(self.cp_obj.get_hp_flag())
        self.assertTrue(self.cp_obj.get_allCam_flag())
        self.assertTrue(self.cp_obj.get_tiff_flag())
        self.assertTrue(self.cp_obj.get_image_size(), (1280, 1024))
        self.assertTrue(self.cp_obj.get_pixel_size() == (15.15, 16.16))
        self.assertTrue(self.cp_obj.get_chfield() == 17)

        self.assertTrue(self.cp_obj.get_multimedia_params().get_n1() == 18)
        self.assertTrue(self.cp_obj.get_multimedia_params().get_n2()[0] == 19.19)
        self.assertTrue(self.cp_obj.get_multimedia_params().get_n3() == 20.20)
        self.assertTrue(self.cp_obj.get_multimedia_params().get_d()[0] == 21.21)

    def test_getters_setters(self):
        """Test getters and setters of ControlPar class."""
        cams_num = 4
        for cam in range(cams_num):
            new_str = str(cam) + "some string" + str(cam)

            self.cp_obj.img_base_name.append(new_str)
            self.assertTrue(self.cp_obj.img_base_name[cam] == new_str)

            self.cp_obj.cal_img_base_name.append(new_str)
            self.assertTrue(self.cp_obj.cal_img_base_name[cam] == new_str)

        self.cp_obj.hp_flag = True
        self.assertTrue(self.cp_obj.hp_flag)
        self.cp_obj.hp_flag = False
        self.assertTrue(not self.cp_obj.hp_flag)

        self.cp_obj.allCam_flag = True
        self.assertTrue(self.cp_obj.allCam_flag)
        self.cp_obj.allCam_flag = False
        self.assertTrue(not self.cp_obj.allCam_flag)

        self.cp_obj.tiff_flag = True
        self.assertTrue(self.cp_obj.tiff_flag)
        self.cp_obj.tiff_flag = False
        self.assertTrue(not self.cp_obj.tiff_flag)

        self.cp_obj.set_image_size((4, 5))
        self.assertTrue(self.cp_obj.get_image_size() == (4, 5))
        print(self.cp_obj.pix_x, self.cp_obj.pix_y)
        self.cp_obj.set_pixel_size((6.1, 7.0))
        numpy.testing.assert_array_equal(self.cp_obj.get_pixel_size(), (6.1, 7))

        self.cp_obj.chfield = 8
        self.assertTrue(self.cp_obj.chfield == 8)

    # testing __richcmp__ comparison method of ControlPar class
    def test_rich_compare(self):
        """Test __richcmp__ method of ControlPar class."""
        self.cp_obj2 = ControlPar(num_cams=4)
        self.cp_obj2.from_file(self.input_control_par_file_name)

        self.cp_obj3 = ControlPar(num_cams=4)
        self.cp_obj3.from_file(self.input_control_par_file_name)

        self.assertTrue(self.cp_obj2 == self.cp_obj3)
        self.assertFalse(self.cp_obj2 != self.cp_obj3)

        self.cp_obj2.set_hp_flag(False)
        self.assertTrue(self.cp_obj2 != self.cp_obj3)
        self.assertFalse(self.cp_obj2 == self.cp_obj3)

    def tearDown(self):
        # remove the testing output directory and its files
        shutil.rmtree(self.temp_output_directory)


class TestTargetPar(unittest.TestCase):
    def test_read(self):
        """Test reading of TargetPar class."""
        inp_filename = "tests/testing_folder/target_parameters/targ_rec.par"
        tp = read_target_par(inp_filename)
        if tp is None:
            print(inp_filename)
            print(os.getcwd())
            print(os.path.exists(inp_filename))

        self.assertEqual(tp.discont, 5)
        self.assertEqual(tp.nnmin, 3)
        self.assertEqual(tp.nnmax, 100)
        self.assertEqual((tp.nxmin, tp.nxmax), (1, 20))
        self.assertEqual((tp.nymin, tp.nymax), (1, 20))
        self.assertEqual(tp.sumg_min, 3)

        numpy.testing.assert_array_equal(tp.gvthresh, [3, 2, 2, 3])

    def test_instantiate_fast(self):
        tp = TargetPar(
            discont=1,
            gvthresh=[2, 3, 4, 5],
            nnmin=10,
            nnmax=100,
            nxmin=20,
            nxmax=200,
            nymin=30,
            nymax=300,
            sumg_min=60,
            cr_sz=3,
        )

        self.assertEqual(tp.discont, 1)
        self.assertEqual((tp.nnmin, tp.nnmax), (10, 100))
        self.assertEqual((tp.nxmin, tp.nxmax), (20, 200))
        self.assertEqual((tp.nymin, tp.nymax), (30, 300))
        self.assertEqual(tp.sumg_min, 60)

        numpy.testing.assert_array_equal(tp.gvthresh, [2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
