import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    # test calibration using scipy.optimize
    return


@app.cell
def _():
    import numpy as np
    import scipy.optimize as opt

    from openptv_python.calibration import Calibration
    from openptv_python.imgcoord import image_coordinates, img_coord
    from openptv_python.orientation import external_calibration, full_calibration
    from openptv_python.parameters import OrientPar, read_control_par
    from openptv_python.tracking_frame_buf import Target
    from openptv_python.trafo import arr_metric_to_pixel, pixel_to_metric

    return (
        Calibration,
        OrientPar,
        Target,
        arr_metric_to_pixel,
        external_calibration,
        full_calibration,
        image_coordinates,
        img_coord,
        np,
        opt,
        pixel_to_metric,
        read_control_par,
    )


@app.cell
def _(
    Calibration,
    OrientPar,
    arr_metric_to_pixel,
    external_calibration,
    full_calibration,
    image_coordinates,
    np,
    read_control_par,
):
    _control_file_name = 'testing_folder/corresp/control.par'
    control = read_control_par(_control_file_name)
    _orient_par_file_name = 'testing_folder/corresp/orient.par'
    orient_par = OrientPar().from_file(_orient_par_file_name)
    cal = Calibration().from_file('testing_folder/calibration/cam1.tif.ori', 'testing_folder/calibration/cam1.tif.addpar')
    _orig_cal = Calibration().from_file('testing_folder/calibration/cam1.tif.ori', 'testing_folder/calibration/cam1.tif.addpar')
    'External calibration using clicked points.'
    ref_pts = np.array([[-40.0, -25.0, 8.0], [40.0, -15.0, 0.0], [40.0, 15.0, 0.0], [40.0, 0.0, 8.0]])
    targets = arr_metric_to_pixel(image_coordinates(ref_pts, cal, control.mm), control)
    targets[:, 1] = targets[:, 1] - 0.1
    external_calibration(cal, ref_pts, targets, control)
    np.testing.assert_array_almost_equal(cal.get_angles(), _orig_cal.get_angles(), decimal=3)
    np.testing.assert_array_almost_equal(cal.get_pos(), _orig_cal.get_pos(), decimal=3)
    _, _, _ = full_calibration(cal, ref_pts, targets, control, orient_par)
    np.testing.assert_array_almost_equal(cal.get_angles(), _orig_cal.get_angles(), decimal=3)
    np.testing.assert_array_almost_equal(cal.get_pos(), _orig_cal.get_pos(), decimal=3)
    return


@app.cell
def _(Calibration, OrientPar, read_control_par):
    _control_file_name = 'testing_folder/corresp/control.par'
    control_1 = read_control_par(_control_file_name)
    _orient_par_file_name = 'testing_folder/corresp/orient.par'
    orient_par_1 = OrientPar().from_file(_orient_par_file_name)
    cal_1 = Calibration().from_file('testing_folder/calibration/cam1.tif.ori', 'testing_folder/calibration/cam1.tif.addpar')
    _orig_cal = Calibration().from_file('testing_folder/calibration/cam1.tif.ori', 'testing_folder/calibration/cam1.tif.addpar')
    return cal_1, control_1, orient_par_1


@app.cell
def _(arr_metric_to_pixel, cal_1, control_1, image_coordinates, np):
    ref_pts_1 = np.array([[-40.0, -25.0, 8.0], [40.0, -15.0, 0.0], [40.0, 15.0, 0.0], [40.0, 0.0, 8.0]])
    targets_1 = arr_metric_to_pixel(image_coordinates(ref_pts_1, cal_1, control_1.mm), control_1)
    cal_1.set_pos(np.array([0, 0, 100]))
    cal_1.set_angles(np.array([0, 0, 0]))
    # Fake the image points by back-projection
    # Jigg the fake detections to give raw_orient some challenge.
    targets_1[:, 1] = targets_1[:, 1] - 0.1
    return ref_pts_1, targets_1


@app.cell
def _(Target, targets_1):
    targs = [Target() for _ in targets_1]
    for ptx, pt in enumerate(targets_1):
        targs[ptx].x = pt[0]
        targs[ptx].y = pt[1]
    return (targs,)


@app.cell
def _():
    # def residual(calibration_array, ref_pts, targs, control, cc):
    #     # print(calibration_array)
    #     # print(ref_pts)
    #     # print(targs)
    #     # print(control)
    #     # print(calibration_array)

    #     c = Calibration()
    #     c.set_pos(calibration_array[:3])
    #     c.set_angles(calibration_array[3:])
    #     c.int_par.cc = cc
    #     c.update_rotation_matrix()


    #     # print(f"{c.get_pos()=}")

    #     residual = 0
    #     for i in range(len(targs)):
    #         xc, yc = pixel_to_metric(targs[i].x, targs[i].y, control)
    #         # print(f"{xc=}, {yc=} mm")

    #         xp, yp = img_coord(ref_pts[i], c, control.mm)
    #         # print(f"{xp=}, {yp=} mm")
    #         residual += ((xc - xp)**2 + (yc - yp)**2)

    #         # print(f"{residual=}")

    #     return residual
    return


@app.cell
def _():
    # x0 = np.hstack([cal.get_pos(), cal.get_angles()])
    # cc = orig_cal.int_par.cc

    # sol = opt.minimize(residual, x0, args=(ref_pts, targs, control, cc), method='Nelder-Mead', tol=1e-6)
    return


@app.cell
def _():
    # print( residual(np.hstack([orig_cal.get_pos(), orig_cal.get_angles()]),
    # ref_pts, targs, control, orig_cal.int_par.cc))
    return


@app.cell
def _(img_coord, pixel_to_metric):
    import copy

    def added_par_residual(added_par_array, ref_pts, targs, control, cal):
        c = copy.deepcopy(cal)
        c.added_par = added_par_array  # print(calibration_array)
        residual = 0  # print(ref_pts)
        for i in range(len(targs)):  # print(targs)
            xc, yc = pixel_to_metric(targs[i].x, targs[i].y, control)  # print(control)
            xp, yp = img_coord(ref_pts[i], c, control.mm)  # print(calibration_array)
            residual = residual + ((xc - xp) ** 2 + (yc - yp) ** 2)
        return residual  # print(f"{c.get_pos()=}")  # print(f"{xc=}, {yc=} mm")  # print(f"{xp=}, {yp=} mm")  # print(f"{residual=}")

    return (added_par_residual,)


@app.cell
def _(added_par_residual, cal_1, control_1, np, opt, ref_pts_1, targs):
    x0 = np.array(cal_1.added_par.tolist())
    sol = opt.minimize(added_par_residual, x0, args=(ref_pts_1, targs, control_1, cal_1), method='Nelder-Mead', tol=1e-06)
    # print(sol.x - np.hstack([orig_cal.get_pos(), orig_cal.get_angles()]))
    print(f'sol.x={sol.x!r}')
    return (sol,)


@app.cell
def _(
    cal_1,
    control_1,
    full_calibration,
    orient_par_1,
    ref_pts_1,
    sol,
    targets_1,
):
    # print(sol.x)
    print(cal_1.added_par)
    cal_1.set_added_par(sol.x)
    print(cal_1.added_par)
    full_calibration(cal_1, ref_pts_1, targets_1, control_1, orient_par_1)
    print(cal_1.added_par)
    return


@app.cell
def _(cal_1):
    print(cal_1.get_pos())
    print(cal_1.get_angles())
    print(cal_1.get_primary_point())
    print(cal_1.get_decentering())
    print(cal_1.added_par)
    return


if __name__ == "__main__":
    app.run()
