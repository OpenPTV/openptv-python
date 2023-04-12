import numpy as np

from openptv_python.calibration import Calibration
from openptv_python.correspondences import MatchedCoords, correspondences
from openptv_python.imgcoord import img_coord
from openptv_python.parameters import (
    ControlPar,
    VolumePar,
    read_control_par,
    read_volume_par,
)
from openptv_python.tracking_frame_buf import read_targets
from openptv_python.trafo import metric_to_pixel


def test_instantiate():
    """Creating a MatchedCoords object."""
    cal = Calibration()
    cal.from_file(
        "tests/testing_folder/calibration/cam1.tif.ori",
        "tests/testing_folder/calibration/cam2.tif.addpar",
    )
    cpar = read_control_par("tests/testing_folder/corresp/control.par")
    targs = read_targets("tests/testing_folder/frame/cam1.", 333)

    mc = MatchedCoords(targs, cpar, cal)
    pos, pnr = mc.as_arrays()

    # x sorted?
    assert np.all(pos[1:, 0] > pos[:-1, 0])

    # Manually verified order for the loaded data:
    np.testing.assert_array_equal(pnr, np.r_[6, 11, 10, 8, 1, 4, 7, 0, 2, 9, 5, 3, 12])


def test_full_corresp(self):
    """Full scene correspondences."""
    from openptv_python.correspondences import TargetArray

    cpar = read_control_par("tests/testing_folder/corresp/control.par")
    vpar = read_volume_par("tests/testing_folder/corresp/criteria.par")

    # Cameras are at so high angles that opposing cameras don't see each
    # other in the normal air-glass-water setting.
    cpar.mm.set_layers([1.0001], [1.0])
    cpar.mm.n3 = 1.0001

    cals = []
    img_pts = []
    corrected = []
    for c in range(4):
        cal = Calibration()
        cal.from_file(
            "tests/testing_folder/calibration/sym_cam%d.tif.ori" % (c + 1),
            "tests/testing_folder/calibration/cam1.tif.addpar",
        )
        cals.append(cal)

        # Generate test targets.
        targs = TargetArray(16)
        for row, col in np.ndindex(4, 4):
            targ_ix = row * 4 + col
            # Avoid symmetric case:
            if c % 2:
                targ_ix = 15 - targ_ix
            targ = targs[targ_ix]

            pos3d = 10 * np.array([[col, row, 0]], dtype=np.float64)
            pos2d = img_coord(pos3d, cal, cpar.mm)
            targ.set_pos(metric_to_pixel(pos2d, cpar))

            targ.set_pnr(targ_ix)
            targ.set_pixel_counts(25, 5, 5)
            targ.set_sum_grey_value(10)

        img_pts.append(targs)
        mc = MatchedCoords(targs, cpar, cal)
        corrected.append(mc)

    _, _, num_targs = correspondences(img_pts, corrected, cals, vpar, cpar)
    assert num_targs == 16

    def test_single_cam_corresp(self):
        """Single camera correspondence."""
        cpar = ControlPar
        cpar.read_control_par("tests/testing_folder/single_cam/parameters/ptv.par")
        vpar = VolumePar()
        vpar.read_volume_par("tests/testing_folder/single_cam/parameters/criteria.par")

        # Cameras are at so high angles that opposing cameras don't see each
        # other in the normal air-glass-water setting.
        cpar.get_multimedia_params().set_layers([1.0], [1.0])
        cpar.get_multimedia_params().set_n3(1.0)

        cals = []
        img_pts = []
        corrected = []
        cal = Calibration()
        cal.from_file(
            "tests/testing_folder/single_cam/calibration/cam_1.tif.ori",
            "tests/testing_folder/single_cam/calibration/cam_1.tif.addpar",
        )
        cals.append(cal)

        # Generate test targets.
        targs = TargetArray(9)
        for row, col in np.ndindex(3, 3):
            targ_ix = row * 3 + col
            targ = targs[targ_ix]

            pos3d = 10 * np.array([[col, row, 0]], dtype=np.float64)
            pos2d = img_coord(pos3d, cal, cpar.mm)
            targ.set_pos(metric_to_pixel(pos2d, cpar))

            targ.set_pnr(targ_ix)
            targ.set_pixel_counts(25, 5, 5)
            targ.set_sum_grey_value(10)

            img_pts.append(targs)
            mc = MatchedCoords(targs, cpar, cal)
            corrected.append(mc)

        _, _, num_targs = correspondences(img_pts, corrected, cals, vpar, cpar)

        self.assertEqual(num_targs, 9)
