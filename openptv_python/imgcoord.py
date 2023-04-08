from openptv_python.calibration import Calibration
from openptv_python.multimed import back_trans_Point, multimed_nlay, trans_Cam_Point
from openptv_python.trafo import flat_to_dist
from openptv_python.vec_utils import vec3d, vec_set


def flat_image_coord(orig_pos, cal, mm):
    """_summary_.

    Args:
    ----
        orig_pos (_type_): _description_
        cal (_type_): _description_
        mm (_type_): _description_

    Returns:
    -------
        _type_: _description_
    """
    deno = 0.0
    cal_t = Calibration()
    X_t = Y_t = 0.0
    cross_p = [0.0] * 3
    cross_c = [0.0] * 3
    pos_t = vec3d
    pos = vec3d

    cal_t.mmlut = cal.mmlut

    # This block calculate 3D position in an imaginary air-filled space,
    # i.e. where the point will have been seen in the absence of refractive
    # layers between it and the camera.
    trans_Cam_Point(
        cal.ext_par, mm, cal.glass_par, orig_pos, cal_t.ext_par, pos_t, cross_p, cross_c
    )
    multimed_nlay(cal_t, mm, pos_t, X_t, Y_t)
    pos_t = vec_set(X_t, Y_t, pos_t[2])
    back_trans_Point(pos_t, mm, cal.glass_par, cross_p, cross_c, pos)

    deno = (
        cal.ext_par.dm[0][2] * (pos[0] - cal.ext_par.x0)
        + cal.ext_par.dm[1][2] * (pos[1] - cal.ext_par.y0)
        + cal.ext_par.dm[2][2] * (pos[2] - cal.ext_par.z0)
    )

    x = (
        -cal.int_par.cc
        * (
            cal.ext_par.dm[0][0] * (pos[0] - cal.ext_par.x0)
            + cal.ext_par.dm[1][0] * (pos[1] - cal.ext_par.y0)
            + cal.ext_par.dm[2][0] * (pos[2] - cal.ext_par.z0)
        )
        / deno
    )

    y = (
        -cal.int_par.cc
        * (
            cal.ext_par.dm[0][1] * (pos[0] - cal.ext_par.x0)
            + cal.ext_par.dm[1][1] * (pos[1] - cal.ext_par.y0)
            + cal.ext_par.dm[2][1] * (pos[2] - cal.ext_par.z0)
        )
        / deno
    )

    return x, y


def img_coord(pos, cal, mm):
    x = y = 0.0

    # Estimate metric coordinates in image space using flat_image_coord()
    x, y = flat_image_coord(pos, cal, mm)

    # Distort the metric coordinates using the Brown distortion model
    x, y = flat_to_dist(x, y, cal)

    return x, y
