# %%
# test calibration using scipy.optimize

# %%
import copy

import numpy as np
import scipy.optimize as opt

from openptv_python.calibration import Calibration
from openptv_python.imgcoord import image_coordinates, img_coord
from openptv_python.orientation import external_calibration, full_calibration
from openptv_python.parameters import OrientPar, read_control_par
from openptv_python.tracking_frame_buf import Target
from openptv_python.trafo import arr_metric_to_pixel, pixel_to_metric


def print_cal(cal: Calibration):
    print(cal.get_pos())
    print(cal.get_angles())
    print(cal.get_primary_point())
    print(cal.added_par)

control_file_name = "tests/testing_folder/corresp/control.par"
# self.control = ControlPar(4)
control = read_control_par(control_file_name)

# orient_par_file_name = "tests/testing_folder/corresp/orient.par"
# orient_par = OrientPar().from_file(orient_par_file_name)

cal = Calibration().from_file(
    "tests/testing_folder/calibration/cam1.tif.ori",
    "tests/testing_folder/calibration/cam1.tif.addpar",
)
orig_cal = Calibration().from_file(
    "tests/testing_folder/calibration/cam1.tif.ori",
    "tests/testing_folder/calibration/cam1.tif.addpar",
)



# def test_external_calibration(self):
"""External calibration using clicked points."""
ref_pts = np.array(
    [
        [-40.0, -25.0, 8.0],
        [40.0, -15.0, 0.0],
        [40.0, 15.0, 0.0],
        [40.0, 0.0, 8.0],
    ]
)

# Fake the image points by back-projection
targets = arr_metric_to_pixel(
    image_coordinates(ref_pts, cal, control.mm),
    control,
)

# Jigg the fake detections to give raw_orient some challenge.
targets[:, 1] -= 0.1

external_calibration(cal, ref_pts, targets, control)

np.testing.assert_array_almost_equal(
    cal.get_angles(), orig_cal.get_angles(), decimal=3
)
np.testing.assert_array_almost_equal(
    cal.get_pos(), orig_cal.get_pos(), decimal=3
)

tmp_orient_par = OrientPar()

_, _, _ = full_calibration(
            cal,
            ref_pts,
            targets,
            control,
            tmp_orient_par
            )

np.testing.assert_array_almost_equal(
    cal.get_angles(), orig_cal.get_angles(), decimal=3
)
np.testing.assert_array_almost_equal(
    cal.get_pos(), orig_cal.get_pos(), decimal=3
)

print_cal(cal)


print("with added par")
tmp_orient_par = OrientPar()
tmp_orient_par.k1flag = 1
tmp_orient_par.k2flag = 0
tmp_orient_par.k3flag = 0

tmp_orient_par.p1flag = 1
tmp_orient_par.p2flag = 0

tmp_orient_par.scxflag = 1
tmp_orient_par.sheflag = 0
# tmp_orient_par.k3flag = 1

_, _, _ = full_calibration(
            cal,
            ref_pts,
            targets,
            control,
            tmp_orient_par
            )
print_cal(cal)

# # %%
# control_file_name = "tests/testing_folder/corresp/control.par"
# control = read_control_par(control_file_name)

# orient_par_file_name = "tests/testing_folder/corresp/orient.par"
# orient_par = OrientPar().from_file(orient_par_file_name)

# cal = Calibration().from_file(
#     "tests/testing_folder/calibration/cam1.tif.ori",
#     "tests/testing_folder/calibration/cam1.tif.addpar",
# )
# orig_cal = Calibration().from_file(
#     "tests/testing_folder/calibration/cam1.tif.ori",
#     "tests/testing_folder/calibration/cam1.tif.addpar")

# # %%
# ref_pts = np.array(
#     [
#         [-40.0, -25.0, 8.0],
#         [40.0, -15.0, 0.0],
#         [40.0, 15.0, 0.0],
#         [40.0, 0.0, 8.0],
#     ]
# )

# # Fake the image points by back-projection
# targets = arr_metric_to_pixel(
#     image_coordinates(ref_pts, cal, control.mm),
#     control,
# )

# cal.set_pos(np.array([0, 0, 100]))
# cal.set_angles(np.array([0, 0, 0]))

# # Jigg the fake detections to give raw_orient some challenge.
# targets[:, 1] -= 0.1

# # %%
targs = [Target() for _ in targets]

for ptx, pt in enumerate(targets):
    targs[ptx].x = pt[0]
    targs[ptx].y = pt[1]
    targs[ptx].pnr = ptx

def added_par_residual(added_par_array, ref_pts, targs, control, cal):
    c = copy.deepcopy(cal)
    c.added_par = added_par_array

    residual = 0
    for i, t in enumerate(targs):
        xc, yc = pixel_to_metric(t.x, t.y, control)
        xp, yp = img_coord(ref_pts[i], c, control.mm)
        residual += ((xc - xp)**2 + (yc - yp)**2)

    return residual



np.seterr(all='raise')

x0 = np.array(cal.added_par.tolist())
sol = opt.minimize(added_par_residual, x0, args=(ref_pts, targs, control, cal), \
    method='Nelder-Mead', tol=1e-6)
print(f"{sol.x=}")
# print(sol.x - np.hstack([orig_cal.get_pos(), orig_cal.get_angles()]))



# # # %%
# # # print(sol.x)
# # print(cal.added_par)
cal.set_added_par(sol.x)
print_cal(cal)
# # print(cal.added_par)

full_calibration(cal, ref_pts, targets, control, tmp_orient_par)
print_cal(cal)

# # # %%


# print(added_par_residual(cal.added_par, ref_pts, targs, control, cal))
