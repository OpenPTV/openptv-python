"""Unit tests for the correspondence code."""
import unittest
from pathlib import Path

import numpy as np

from openptv_python.calibration import Calibration, read_calibration
from openptv_python.constants import MAXCAND
from openptv_python.correspondences import (
    consistent_pair_matching,
    match_pairs,
    safely_allocate_adjacency_lists,
    safely_allocate_target_usage_marks,
)
from openptv_python.epi import Coord2d_dtype
from openptv_python.imgcoord import img_coord
from openptv_python.parameters import ControlPar, read_control_par, read_volume_par
from openptv_python.tracking_frame_buf import (
    Frame,
    Target,
    n_tupel_dtype,
)
from openptv_python.trafo import dist_to_flat, metric_to_pixel, pixel_to_metric


def read_all_calibration(num_cams: int = 4) -> list[Calibration]:
    """Read all calibration files."""
    ori_tmpl = "tests/testing_fodder/cal/sym_cam%d.tif.ori"
    added_name = Path("tests/testing_fodder/cal/cam1.tif.addpar")

    calib = []

    for cam in range(num_cams):
        ori_name = Path(ori_tmpl % (cam + 1))
        calib.append(read_calibration(ori_name, added_name))

    # print(calib)

    return calib


def correct_frame(
    frm: Frame, calib: list[Calibration], cpar: ControlPar, tol: float
) -> np.recarray:
    """
    Perform the transition from pixel to metric to flat coordinates.

    and x-sorting as required by the correspondence code.

    Arguments:
    ---------
    frm - target information for all cameras.
    cpar - parameters of image size, pixel size etc.
    tol - tolerance parameter for iterative flattening phase, see
        trafo.h:correct_brown_affine_exact().
    """
    corrected = np.recarray((cpar.num_cams, frm.num_targets[0]), dtype=Coord2d_dtype)

    for cam in range(cpar.num_cams):
        row = corrected[cam]

        for part in range(frm.num_targets[cam]):
            x, y = pixel_to_metric(
                frm.targets[cam][part].x, frm.targets[cam][part].y, cpar
            )
            x, y = dist_to_flat(x, y, calib[cam], tol)

            row[part].pnr = frm.targets[cam][part].pnr
            row[part].x = x
            row[part].y = y

        # This is expected by find_candidate()
        row.sort(order="x")

    return corrected


def generate_test_set(calib: list[Calibration], cpar: ControlPar) -> Frame:
    """
    Generate data for targets on N cameras.

    The targets are organized on a 4x4 grid, 10 mm apart.
    """
    frm = Frame(num_cams=cpar.num_cams, max_targets=16)

    # Four cameras on 4 quadrants looking down into a calibration target.
    # Calibration taken from an actual experimental setup
    for cam in range(cpar.num_cams):
        # fill in only what's needed
        frm.num_targets[cam] = 16
        frm.targets[cam] = [Target() for _ in range(frm.num_targets[cam])]

        # print(f"\n cpar = {cpar}")

        # Construct a scene representing a calibration target, generate
        # targets for it, then use them to reconstruct correspondences.
        for cpt_horz in range(4):
            for cpt_vert in range(4):
                cpt_ix = cpt_horz * 4 + cpt_vert
                if cam % 2:
                    cpt_ix = 15 - cpt_ix  # Avoid symmetric case

                targ = frm.targets[cam][cpt_ix]
                targ.pnr = cpt_ix

                tmp = np.r_[cpt_vert * 10, cpt_horz * 10, 0]
                # if ((cpt_ix % 4) == 0):
                #     print(f"cam {cam}, cpt {cpt_ix}, {tmp} mm")

                targ.x, targ.y = img_coord(tmp, calib[cam], cpar.mm)

                # if ((cpt_ix % 4) == 0):
                #     print(f"cam {cam}, cpt {cpt_ix} {targ.x} {targ.y} sensor \n")

                targ.x, targ.y = metric_to_pixel(targ.x, targ.y, cpar)
                # if ((cpt_ix % 4) == 0):
                #     print(f"cam {cam}, cpt {cpt_ix} {targ.x} {targ.y} pix \n")

                # These values work in check_epi, so used here too
                targ.n = 25
                targ.nx = targ.ny = 5
                targ.sumg = 10
                # print(targ)

    return frm


class TestTwoCameraMatching(unittest.TestCase):
    """Test the read_control_par function."""

    def test_two_camera_matching(self):
        """Setup is the same as the 4-camera test, targets are darkened in.

        two cameras to get 16 pairs.
        """
        cpar = read_control_par(Path("tests/testing_fodder/parameters/ptv.par"))
        vpar = read_volume_par(Path("tests/testing_fodder/parameters/criteria.par"))

        # Cameras are at so high angles that opposing cameras don't see each other
        # in the normal air-glass-water setting.
        cpar.num_cams = 2
        cpar.mm.n2[0] = 1.0001
        cpar.mm.n3 = 1.0001

        vpar.z_min_lay[0] = -1
        vpar.z_min_lay[1] = -1
        vpar.z_max_lay[0] = 1
        vpar.z_max_lay[1] = 1

        calib = read_all_calibration(cpar.num_cams)
        frm = generate_test_set(calib, cpar)

        corrected = correct_frame(frm, calib, cpar, 0.0001)

        # print(corrected)

        # print(len(corrected))

        # _, ax = plt.subplots(1, 2)
        # markers = ["o", "x"]

        # for i in range(len(corrected)):
        #     for col in corrected[i]:
        #         ax[i].scatter(col.x, col.y, c="r", marker=markers[i])
        #         ax[i].text(col.x, col.y, str(col.pnr))

        # plt.show()

        corr_lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)

        match_pairs(corr_lists, corrected, frm, vpar, cpar, calib)

        # print(corr_lists)

        found_correct_pnr = False

        # Assert each target has the real matches as candidates
        for cam in range(cpar.num_cams - 1):
            for subcam in range(cam + 1, cpar.num_cams):
                for part in range(frm.num_targets[cam]):
                    correct_pnr = (
                        corrected[cam][corr_lists[cam][subcam][part].p1].pnr
                        if (subcam - cam) % 2 == 0
                        else 15 - corrected[cam][corr_lists[cam][subcam][part].p1].pnr
                    )

                    # print(
                    #     f"cam {cam}, second cam {subcam}, point no. {part}, should match {correct_pnr}"
                    # )

                    for cand in range(MAXCAND):
                        found_correct_pnr = False
                        # print(cand, corr_lists[cam][subcam][part].p2[cand],
                        #  corrected[subcam][corr_lists[cam][subcam][part].p2[cand]].pnr)
                        if (
                            corrected[subcam][
                                corr_lists[cam][subcam][part].p2[cand]
                            ].pnr
                            == correct_pnr
                        ):
                            # print("found")
                            # print(
                            #     corrected[subcam][
                            #         corr_lists[cam][subcam][part].p2[cand]
                            #     ].pnr
                            # )
                            found_correct_pnr = True
                            break

                    self.assertTrue(found_correct_pnr)

        # # continue to the consistent_pair matching test
        con = np.recarray((4 * 16), dtype=n_tupel_dtype)
        con.p = np.zeros(4,)
        con.corr = 0.0

        tusage = safely_allocate_target_usage_marks(cpar.num_cams)

        # high accept corr bcz of closeness to epipolar lines.
        consistent_pair_matching(
            corr_lists, cpar.num_cams, frm.num_targets, 10000.0, con, 4 * 16, tusage
        )

        # print(f" matched = {matched}")

        # assert matched == 16


if __name__ == "__main__":
    unittest.main()
