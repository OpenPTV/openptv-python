import numpy as np

from openptv_python.calibration import Calibration
from openptv_python.imgcoord import img_coord
from openptv_python.parameters import ControlPar
from openptv_python.trafo import metric_to_pixel

"""
Generate a 5-frame trajectory that is pretty degenerates so is good for
testing. It starts from (0,0,0) and moves in a straight line on the x axis,
at a slow velocity.
"""


num_cams = 3
num_frames = 5
velocity = 0.01

part_traject = np.zeros((num_frames, 3))
part_traject[:, 0] = np.r_[:num_frames] * velocity

# Find targets on each camera.
cpar = ControlPar(num_cams=3)
cpar.from_file("testing_fodder/track/parameters/control_newpart.par")

targs = []
for cam in range(num_cams):
    cal = Calibration()
    cal.from_file(
        f"testing_fodder/cal/sym_cam{cam+1}.tif.ori",
        "testing_fodder/cal/cam1.tif.addpar",
    )
    # check this out
    x, y = img_coord(part_traject, cal, cpar.mm)
    x, y = metric_to_pixel(x, y, cpar)
    targs.append([x, y])

for frame in range(num_frames):
    # write 3D positions:
    with open(
        "testing_fodder/track/res_orig/particles.%d" % (frame + 1), "w"
    ) as outfile:
        # Note correspondence to the single target in each frame.
        outfile.writelines(
            [
                str(1) + "\n",
                "{:5d}{:10.3f}{:10.3f}{:10.3f}{:5d}{:5d}{:5d}{:5d}\n".format(
                    1,
                    part_traject[frame, 0],
                    part_traject[frame, 1],
                    part_traject[frame, 1],
                    0,
                    0,
                    0,
                    0,
                ),
            ]
        )

    # write associated targets from all cameras:
    for cam in range(num_cams):
        with open(
            "testing_fodder/track/newpart/cam%d.%04d_targets" % (cam + 1, frame + 1),
            "w",
        ) as outfile:
            outfile.writelines(
                [
                    str(1) + "\n",
                    "{:5d}{:10.3f}{:10.3f}{:5d}{:5d}{:5d}{:10d}{:5d}\n".format(
                        0,
                        targs[cam][frame, 0],
                        targs[cam][frame, 1],
                        100,
                        10,
                        10,
                        10000,
                        0,
                    ),
                ]
            )

# That's all, folks!
