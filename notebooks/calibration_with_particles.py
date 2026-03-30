import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Calibrate with particles
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The idea is to run PyPTV as usual, and check the box "Use only 4 frames". The result will be in the /res folder with only quadruplets as 3D and the respective indices of 2D targets per image

    If we read this dataset into the proper format, we can now reproject every 3D point in rt_is back into the image and then optimize calibration with disparity between the position of the target as detected and the reprojected center.
    """)
    return


@app.cell
def _():
    # -*- coding: utf-8 -*-
    # copy of https://github.com/alexlib/pbi/blob/master/ptv/shake.py
    """
    BOOM shake shake shake the room!!!

    Fine-tune calibration using the "shaking" method of comparing 3D positions 
    obtained with existing calibration to their 2D projections. It's a kind of a 
    feedback step over the normal calibration with known points.

    Created on Sun Jan 31 13:42:18 2016

    @author: Yosef Meller
    """
    import numpy as np
    import os
    from pathlib import Path
    from pyptv.ptv import py_start_proc_c
    from pyptv.parameters import OrientParams
    from optv.orientation import full_calibration
    from optv.tracking_framebuf import TargetArray, Frame
    from pyptv.ptv import full_scipy_calibration

    present_folder = Path.cwd()

    working_folder = Path.cwd() / "tests" / "working_folder" / "test_cavity"
    par_path = working_folder / "parameters"
    working_folder.exists(), par_path.exists()

    # we work inside the working folder, all the other paths are relative to this
    num_cams = 4
    os.chdir(working_folder)
    cpar, spar, vpar, track_par, tpar, calibs, epar = py_start_proc_c(num_cams)
    assert cpar.get_num_cams() == num_cams

    targ_files = [
        spar.get_img_base_name(c).decode().split("%d")[0].encode() for c in range(num_cams)
    ]

    print(targ_files)


    # recognized names for the flags:
    NAMES = ["cc", "xh", "yh", "k1", "k2", "k3", "p1", "p2", "scale", "shear"]
    op = OrientParams()
    op.read()
    flags = [name for name in NAMES if getattr(op, name) == 1]

    print(flags)
    return (
        Frame,
        TargetArray,
        calibs,
        cpar,
        flags,
        full_calibration,
        full_scipy_calibration,
        np,
        num_cams,
        spar,
        targ_files,
    )


@app.cell
def _(cpar):
    def backup_ori_files(cpar):
        """backup ORI/ADDPAR files to the backup_cal directory"""
        import shutil

        for i_cam in range(cpar.get_num_cams()):
            f = cpar.get_cal_img_base_name(i_cam).decode()
            print(f"Backing up {f}.ori")
            shutil.copyfile(f + ".ori", f + ".ori.bck")
            shutil.copyfile(f + ".addpar", f + ".addpar.bck")


    # Backup is the first thing to do
    backup_ori_files(cpar)
    return


@app.cell
def _(calibs, num_cams):
    print('Starting from: calibration')
    for _cam in range(num_cams):
        print(f'cam={_cam!r} {calibs[_cam].get_pos()}, {calibs[_cam].get_angles()}')
    return


@app.cell
def _(Frame, cpar, np, spar, targ_files):
    # Iterate over frames, loading the big lists of 3D positions and
    # respective detections.
    all_known = []
    all_detected = [[] for c in range(cpar.get_num_cams())]
    for frm_num in range(spar.get_first(), spar.get_last() + 1):
        frame = Frame(cpar.get_num_cams(), corres_file_base='res/rt_is'.encode(), linkage_file_base='res/ptv_is'.encode(), target_file_base=targ_files, frame_num=frm_num)
        all_known.append(frame.positions())
        for _cam in range(cpar.get_num_cams()):  # all frames for now, think of skipping some
            all_detected[_cam].append(frame.target_positions_for_camera(_cam))
    # Make into the format needed for full_calibration.
    all_known = np.vstack(all_known)
    return all_detected, all_known


@app.cell
def _(
    TargetArray,
    all_detected,
    all_known,
    calibs,
    cpar,
    full_calibration,
    np,
    num_cams,
):
    # Calibrate each camera accordingly.
    for _cam in range(num_cams):
        _detects = np.vstack(all_detected[_cam])
        assert _detects.shape[0] == all_known.shape[0]
        _have_targets = ~np.isnan(_detects[:, 0])
        _used_detects = _detects[_have_targets, :]
        _used_known = all_known[_have_targets, :]
        _targs = TargetArray(len(_used_detects))
        for _tix in range(len(_used_detects)):
            _targ = _targs[_tix]
            _targ.set_pnr(_tix)
            _targ.set_pos(_used_detects[_tix])
        try:
            _residuals, _targ_ix, _err_est = full_calibration(calibs[_cam], _used_known, _targs, cpar, flags=[])
            print(f'After full calibration, {np.sum(_residuals ** 2)}')
            print('Camera %d' % (_cam + 1))
            print(calibs[_cam].get_pos())  # residuals = full_calibration(calibs[cam], used_known, targs, cpar)
            print(calibs[_cam].get_angles())
        except Exception as e:
            print(f'Error in full_calibration: {e}, run Scipy.optimize')
            continue  # else:  #     if args.output is None:  #         ori = cal_args[cam]['ori_file']  #         distort = cal_args[cam]['addpar_file']  #     else:  #         ori = args.output % (cam + 1) + '.ori'  #         distort = args.output % (cam + 1) + '.addpar'  #     calibs[cam].write(ori.encode(), distort.encode())
    return


@app.cell
def _(
    TargetArray,
    all_detected,
    all_known,
    calibs,
    cpar,
    flags,
    full_calibration,
    np,
    num_cams,
):
    # Calibrate each camera accordingly.
    for _cam in range(num_cams):
        _detects = np.vstack(all_detected[_cam])
        assert _detects.shape[0] == all_known.shape[0]
        _have_targets = ~np.isnan(_detects[:, 0])
        _used_detects = _detects[_have_targets, :]
        _used_known = all_known[_have_targets, :]
        _targs = TargetArray(len(_used_detects))
        for _tix in range(len(_used_detects)):
            _targ = _targs[_tix]
            _targ.set_pnr(_tix)
            _targ.set_pos(_used_detects[_tix])
        try:
            _residuals, _targ_ix, _err_est = full_calibration(calibs[_cam], _used_known, _targs, cpar, flags=flags)
            print(f'After full calibration, {np.sum(_residuals ** 2)}')
            print('Camera %d' % (_cam + 1))
            print(calibs[_cam].get_pos())  # residuals = full_calibration(calibs[cam], used_known, targs, cpar)
            print(calibs[_cam].get_angles())
        except Exception as e:
            print(f'Error in full_calibration: {e}, run Scipy.optimize')
            continue
    return


@app.cell
def _(
    TargetArray,
    all_detected,
    all_known,
    calibs,
    cpar,
    flags,
    full_scipy_calibration,
    np,
    num_cams,
):
    # Calibrate each camera accordingly.
    for _cam in range(num_cams):
        _detects = np.vstack(all_detected[_cam])
        assert _detects.shape[0] == all_known.shape[0]
        _have_targets = ~np.isnan(_detects[:, 0])
        _used_detects = _detects[_have_targets, :]
        _used_known = all_known[_have_targets, :]
        _targs = TargetArray(len(_used_detects))
        for _tix in range(len(_used_detects)):
            _targ = _targs[_tix]
            _targ.set_pnr(_tix)
            _targ.set_pos(_used_detects[_tix])
        _residuals = full_scipy_calibration(calibs[_cam], _used_known, _targs, cpar, flags=flags)
        print(f'After scipy full calibration, {np.sum(_residuals ** 2)}')
        print('Camera %d' % (_cam + 1))
        print(calibs[_cam].get_pos())
    # targ_ix = [t.pnr() for t in targs if t.pnr() != -999]
    # targ_ix = np.arange(len(all_detected))
    # save the results from calibs[cam]
    # _write_ori(i_cam, addpar_flag=True)
        print(calibs[_cam].get_angles())
    return


if __name__ == "__main__":
    app.run()
