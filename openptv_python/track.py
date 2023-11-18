"""Tracking algorithm."""
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .calibration import Calibration
from .constants import (
    ADD_PART,
    COORD_UNUSED,
    CORRES_NONE,
    MAX_CANDS,
    NEXT_NONE,
    POS_INF,
    PREV_NONE,
    PT_UNUSED,
    TR_BUFSPACE,
    TR_MAX_CAMS,
    TR_UNUSED,
)
from .imgcoord import img_coord
from .orientation import point_position
from .parameters import ControlPar, TrackPar
from .tracking_frame_buf import Frame, Target
from .tracking_run import TrackingRun
from .trafo import dist_to_flat, metric_to_pixel, pixel_to_metric
from .vec_utils import vec_copy, vec_diff_norm, vec_subt

default_naming = {
    "corres": b"res/rt_is",
    "linkage": b"res/ptv_is",
    "prio": b"res/added",
}


@dataclass
class Foundpix:
    """A Foundpix object holds the parameters for a found pixel."""

    ftnr: int = TR_UNUSED
    freq: int = 0
    whichcam: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.whichcam = [0] * TR_MAX_CAMS


def reset_foundpix_array(arr: List[Foundpix], arr_len: int, num_cams: int) -> None:
    """Set default values for foundpix objects in an array.

    Arguments:
    ---------
    arr -- the array to reset
    arr_len -- array length
    num_cams -- number of places in the whichcam member of foundpix.
    """
    for i in range(arr_len):
        # Set default values for each foundpix object in the array
        arr[i].ftnr = TR_UNUSED
        arr[i].freq = 0

        # Set default values for each whichcam member of the foundpix object
        for cam in range(num_cams):
            if len(arr[i].whichcam) < num_cams:
                arr[i].whichcam.append(0)
            else:
                arr[i].whichcam[cam] = 0

    return None


def copy_foundpix_array(
    dest: List[Foundpix], src: List[Foundpix], arr_len: int, num_cams: int
) -> None:
    """copy_foundpix_array() copies foundpix objects from one array to another.

    Arguments:
    ---------
    dest -- dest receives the copied array
    src -- src is the array to copy
    arr_len -- array length
    num_cams -- number of places in the whichcam member of foundpix.
    """
    for i in range(arr_len):
        # Copy values from source foundpix object to destination foundpix object
        dest[i].ftnr = src[i].ftnr
        dest[i].freq = src[i].freq

        # Copy values from source whichcam member to destination whichcam member
        for cam in range(num_cams):
            dest[i].whichcam[cam] = src[i].whichcam[cam]


def register_closest_neighbs(
    targets: List[Target],
    num_targets: int,
    cam: int,
    cent_x: float,
    cent_y: float,
    dl: float,
    dr: float,
    du: float,
    dd: float,
    reg: List[Foundpix],
    cpar: ControlPar,
) -> List[int]:
    """Register_closest_neighbs() finds candidates for continuing a particle's.

    path in the search volume, and registers their data in a foundpix array
    that is later used by the tracking algorithm.

    Arguments:
    ---------
    targets -- the targets list to search.
    num_targets -- target array length.
    cam -- the index of the camera we're working on.
    cent_x -- image coordinate of search area center along x-axis, [pixel]
    cent_y -- image coordinate of search area center along y-axis, [pixel]
    dl -- left distance to the search area border from its center, [pixel]
    dr -- right distance to the search area border from its center, [pixel]
    du -- up distance to the search area border from its center, [pixel]
    dd -- down distance to the search area border from its center, [pixel]
    reg -- an array of foundpix objects, one for each possible neighbour. Output array.
    cpar -- control parameter object
    """
    # all_cands = [-999] * MAX_CANDS  # Initialize all candidate indexes to -999

    all_cands = candsearch_in_pix(
        targets, num_targets, cent_x, cent_y, dl, dr, du, dd, cpar
    )

    for cand_idx in range(MAX_CANDS):
        # Set default value for unused foundpix objects
        if all_cands[cand_idx] == PT_UNUSED:
            reg[cand_idx].ftnr = TR_UNUSED
        else:
            # Register candidate data in the foundpix object
            reg[cand_idx].whichcam[cam] = 1
            reg[cand_idx].ftnr = targets[all_cands[cand_idx]].tnr

    return all_cands


def search_volume_center_moving(
    prev_pos: np.ndarray, curr_pos: np.ndarray
) -> np.ndarray:
    """Find the position of the center of the search volume for a moving.

    particle using the velocity of last step.

    Args:
    ----
        prev_pos (vec3d): Previous position of the particle.
        curr_pos (vec3d): Current position of the particle.
        output (vec3d): Output variable for the calculated position.

    Returns
    -------
        None
    """
    # Multiply current position by 2 and subtract previous position
    # output[0] = 2 * curr_pos[0] - prev_pos[0]
    # output[1] = 2 * curr_pos[1] - prev_pos[1]
    # output[2] = 2 * curr_pos[2] - prev_pos[2]

    return 2 * curr_pos - prev_pos


def predict(prev_pos, curr_pos, output):
    """Predicts the position of a particle in the next_frame frame, using the.

    previous and current positions.

    Args:
    ----
        prev_pos (vec2d): 2D position at previous frame.
        curr_pos (vec2d): 2D position at current frame.
        output (vec2d): Output of the 2D positions of the particle in the next_frame frame.

    Returns
    -------
        None
    """
    # Calculate the position of the particle in the next_frame frame using the current and previous positions
    output[0] = 2 * curr_pos[0] - prev_pos[0]
    output[1] = 2 * curr_pos[1] - prev_pos[1]


def pos3d_in_bounds(pos, bounds):
    """Check that all components of a pos3d are in their respective bounds.

    taken from a track_par object.

    Args:
    ----
        pos (vec3d): The 3-component array to check.
        bounds (track_par): The struct containing the bounds specification.

    Returns
    -------
        True if all components are in bounds, False otherwise.
    """
    # Check if all three components of pos are within their respective bounds in bounds.
    return (
        bounds.dvxmin < pos[0] < bounds.dvxmax
        and bounds.dvymin < pos[1] < bounds.dvymax
        and bounds.dvzmin < pos[2] < bounds.dvzmax
    )


# def angle_acc(
#     start: np.ndarray, pred: np.ndarray, cand: np.ndarray
# ) -> Tuple[float, float]:
#     """Calculate the angle between the (1st order) numerical velocity vectors.

#     to the predicted next_frame position and to the candidate actual position. The
#     angle is calculated in [gon], see [1]. The predicted position is the
#     position if the particle continued at current velocity.

#     Arguments:
#     ---------
#     start -- vec3d, the particle start position
#     pred -- vec3d, predicted position
#     cand -- vec3d, possible actual position

#     Returns:
#     -------
#     angle -- float, the angle between the two velocity vectors, [gon]
#     acc -- float, the 1st-order numerical acceleration embodied in the deviation from prediction.
#     """
#     v0 = pred - start
#     v1 = cand - start

#     acc = math.dist(v0, v1)
#     # acc = np.linalg.norm(v0 - v1)

#     if np.all(v0 == -v1):
#         angle = 200
#     elif np.all(v0 == v1):
#         angle = 0
#     else:
#         angle = float((200.0 / math.pi) * math.acos(
#             math.fsum([v0[i] * v1[i] for i in range(3)])
#             / (math.dist(start, pred) * math.dist(start, cand)))
#         )

#     return angle, acc


def angle_acc(
    start: np.ndarray, pred: np.ndarray, cand: np.ndarray
) -> Tuple[float, float]:
    """Calculate the angle between the (1st order) numerical velocity vectors.

    to the predicted next_frame position and to the candidate actual position. The
    angle is calculated in [gon], see [1]. The predicted position is the
    position if the particle continued at the current velocity.

    Arguments:
    ---------
    start -- vec3d, the particle start position
    pred -- vec3d, predicted position
    cand -- vec3d, possible actual position

    Returns
    -------
    angle -- float, the angle between the two velocity vectors, [gon]
    acc -- float, the 1st-order numerical acceleration embodied in the deviation from prediction.
    """
    v0 = pred - start
    v1 = cand - start

    acc = np.linalg.norm(v0 - v1)

    if np.all(v0 == -v1):
        angle = 200
    elif np.all(v0 == v1):
        angle = 0
    else:
        dot_product = np.sum(v0 * v1)
        norm_start_pred = np.linalg.norm(start - pred)
        norm_start_cand = np.linalg.norm(start - cand)

        angle = (200.0 / np.pi) * np.arccos(
            dot_product / (norm_start_pred * norm_start_cand)
        )

    return float(angle), float(acc)


def candsearch_in_pix(
    next_frame: List[Target],
    num_targets: int,
    cent_x: float,
    cent_y: float,
    dl: float,
    dr: float,
    du: float,
    dd: float,
    cpar: ControlPar,
) -> List[int]:
    """Search for a nearest candidate in unmatched target list."""
    # counter = 0
    dmin = 1e20
    p1 = p2 = p3 = p4 = TR_UNUSED
    p = [-1] * MAX_CANDS
    d1, d2, d3, d4 = dmin, dmin, dmin, dmin

    xmin, xmax, ymin, ymax = cent_x - dl, cent_x + dr, cent_y - du, cent_y + dd

    if xmin < 0:
        xmin = 0
    if xmax > cpar.imx:
        xmax = cpar.imx
    if ymin < 0:
        ymin = 0
    if ymax > cpar.imy:
        ymax = cpar.imy

    if cent_x >= 0 and cent_x <= cpar.imx and cent_y >= 0 and cent_y <= cpar.imy:
        j0 = num_targets // 2
        dj = num_targets // 4
        while dj > 1:
            if next_frame[j0].y < ymin:
                j0 += dj
            else:
                j0 -= dj
            dj //= 2

        j0 -= 12
        if j0 < 0:
            j0 = 0

        for j in range(j0, num_targets):
            if next_frame[j].tnr != -1:
                if next_frame[j].y > ymax:
                    break
                if xmin < next_frame[j].x < xmax and ymin < next_frame[j].y < ymax:
                    d = np.sqrt(
                        (cent_x - next_frame[j].x) ** 2
                        + (cent_y - next_frame[j].y) ** 2
                    )

                    if d < dmin:
                        dmin = d

                    if d < d1:
                        p4, p3, p2, p1 = p3, p2, p1, j
                        d4, d3, d2, d1 = d3, d2, d1, d
                    elif d1 < d < d2:
                        p4, p3, p2 = p3, p2, j
                        d4, d3, d2 = d3, d2, d
                    elif d2 < d < d3:
                        p4, p3 = p3, j
                        d4, d3 = d3, d
                    elif d3 < d < d4:
                        p4 = j
                        d4 = d

        p[0] = p1
        p[1] = p2
        p[2] = p3
        p[3] = p4

        # print("from inside p = ", p)

        # TODO: check why we need counter, we can use counter = len(p) - p.count(-1)
        # for j in range(4):
        #     if p[j] != -1:
        #         counter += 1

    return p


def candsearch_in_pix_rest(
    next_frame: List[Target],
    num_targets: int,
    cent_x: float,
    cent_y: float,
    dl: float,
    dr: float,
    du: float,
    dd: float,
    p: List[int],
    cpar: ControlPar,
) -> int:
    """Search for a nearest candidate in unmatched target list.

    Arguments:
    ---------
    next_frame - 2D numpy array of targets (pointer, x,y, n, nx,ny, sumg, track ID),
        assumed to be y sorted.
    num_targets - number of targets in the next_frame
    cent_x, cent_y - image coordinates of the position of a particle [pixel]
    dl, dr, du, dd - respectively the left, right, up, down distance
        to the search area borders from its center, [pixel]
    cpar - control_par object with attributes imx and imy.

    Returns
    -------
    int - the number of candidates found, between 0 - 1
    """
    counter = 0
    dmin = POS_INF
    xmin, xmax, ymin, ymax = cent_x - dl, cent_x + dr, cent_y - du, cent_y + dd

    xmin = max(xmin, 0.0)
    xmax = min(xmax, cpar.imx)
    ymin = max(ymin, 0.0)
    ymax = min(ymax, cpar.imy)

    if 0 <= cent_x <= cpar.imx and 0 <= cent_y <= cpar.imy:
        # binarized search for start point of candidate search
        j0, dj = num_targets // 2, num_targets // 4
        while dj > 1:
            j0 += dj if next_frame[j0].y < ymin else -dj
            dj //= 2

        j0 -= 12 if j0 >= 12 else j0  # due to trunc
        for j in range(j0, num_targets):
            if next_frame[j].tnr == TR_UNUSED:
                if next_frame[j].y > ymax:
                    break  # finish search
                if xmin < next_frame[j].x < xmax and ymin < next_frame[j].y < ymax:
                    d = np.sqrt(
                        (cent_x - next_frame[j].x) ** 2
                        + (cent_y - next_frame[j].y) ** 2
                    )
                    if d < dmin:
                        dmin = d
                        p[0] = j

        if p[0] != -1:
            counter += 1

    return counter


def searchquader(
    point: np.ndarray, tpar: TrackPar, cpar: ControlPar, cal: List[Calibration]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the search volume in image space."""
    mins = np.array([tpar.dvxmin, tpar.dvymin, tpar.dvzmin])
    maxes = np.array([tpar.dvxmax, tpar.dvymax, tpar.dvzmax])

    quader = np.zeros((8, 3))
    xr = np.zeros(cpar.num_cams)
    xl = np.zeros(cpar.num_cams)
    yd = np.zeros(cpar.num_cams)
    yu = np.zeros(cpar.num_cams)

    for pt in range(8):
        quader[pt] = point.copy()
        # print(f" pt {pt} {quader[pt]}")
        for dim in range(3):
            if pt & (1 << dim):
                quader[pt][dim] += maxes[dim]
            else:
                quader[pt][dim] += mins[dim]
        # print(f" pt {pt} {quader[pt]}")

    # calculation of search area in each camera
    for i in range(cpar.num_cams):
        # initially large or small values
        xr[i] = 0
        xl[i] = cpar.imx
        yd[i] = 0
        yu[i] = cpar.imy

        # pixel position of a search center
        center = point_to_pixel(point, cal[i], cpar)
        # print(" center", center[0], center[1])

        # mark 4 corners of the search region in pixels
        for pt in range(8):
            corner = point_to_pixel(quader[pt], cal[i], cpar)
            # print(" corner", corner[0], corner[1])

            if corner[0] < xl[i]:
                xl[i] = corner[0]
            if corner[1] < yu[i]:
                yu[i] = corner[1]
            if corner[0] > xr[i]:
                xr[i] = corner[0]
            if corner[1] > yd[i]:
                yd[i] = corner[1]

        if xl[i] < 0:
            xl[i] = 0
        if yu[i] < 0:
            yu[i] = 0
        if xr[i] > cpar.imx:
            xr[i] = cpar.imx
        if yd[i] > cpar.imy:
            yd[i] = cpar.imy

        # print(" xl", xl[i], " xr", xr[i], " yu", yu[i], " yd", yd[i])

        # eventually xr, xl, yd, yu are pixel distances relative to the point
        xr[i] = xr[i] - center[0]
        xl[i] = center[0] - xl[i]
        yd[i] = yd[i] - center[1]
        yu[i] = center[1] - yu[i]

        # print(" xl", xl[i], " xr", xr[i], " yu", yu[i], " yd", yd[i])

    return xr, xl, yd, yu


def sort_candidates_by_freq(foundpix: List[Foundpix], num_cams: int) -> int:
    """Sort candidates by frequency."""
    different = 0

    # where what was found
    for i in range(num_cams * MAX_CANDS):
        for j in range(num_cams):
            for m in range(MAX_CANDS):
                if foundpix[i].ftnr == foundpix[4 * j + m].ftnr:
                    foundpix[i].whichcam[j] = 1

    # how often was ftnr found
    for i in range(num_cams * MAX_CANDS):
        for j in range(num_cams):
            if foundpix[i].whichcam[j] == 1 and foundpix[i].ftnr != TR_UNUSED:
                foundpix[i].freq += 1

    # sort freq
    for i in range(1, num_cams * MAX_CANDS):
        for j in range(num_cams * MAX_CANDS - 1, i - 1, -1):
            if foundpix[j - 1].freq < foundpix[j].freq:
                foundpix[j - 1], foundpix[j] = foundpix[j], foundpix[j - 1]

    # prune the duplicates or those that are found only once
    for i in range(num_cams * MAX_CANDS):
        for j in range(i + 1, num_cams * MAX_CANDS):
            if foundpix[i].ftnr == foundpix[j].ftnr or foundpix[j].freq < 2:
                foundpix[j].freq = 0
                foundpix[j].ftnr = TR_UNUSED

    # sort freq again on the clean dataset
    for i in range(1, num_cams * MAX_CANDS):
        for j in range(num_cams * MAX_CANDS - 1, i - 1, -1):
            if foundpix[j - 1].freq < foundpix[j].freq:
                foundpix[j - 1], foundpix[j] = foundpix[j], foundpix[j - 1]

    for i in range(num_cams * MAX_CANDS):
        if foundpix[i].freq != 0:
            different += 1

    return different


def sort(n: int, a: List[float], b: List[int]) -> Tuple[List[float], List[int]]:
    """In-place sorts a float list 'a' and an integer list 'b' equal lengths, sort up to n.

    Arguments:
    ---------
    a -- float array (returned sorted in ascending order)
    b -- integer array (returned sorted according to float array a)

    Returns
    -------
    Sorted arrays a and b.
    """
    # idx = np.argsort(a)
    # a[...] = a[idx]
    # b[...] = b[idx]

    # return a, b

    sorted_pairs = sorted(zip(a[:n], b[:n]))
    a[:n], b[:n] = zip(*sorted_pairs)
    return a, b


def point_to_pixel(point: np.ndarray, cal: Calibration, cpar: ControlPar) -> np.ndarray:
    """Return vec2d with pixel positions (x,y) in the camera.

    Arguments:
    ---------
    point -- vec3d point in 3D space
    cal -- Calibration parameters
    cpar -- Control parameters (num cams, multimedia parameters, cpar->mm, etc.)

    Returns
    -------
    vec2d with pixel positions (x,y) in the camera.
    """
    # print(f"point {point}")
    # print(f"cal {cal}")
    # print(f"cpar.mm {cpar.mm}")

    x, y = img_coord(point, cal, cpar.mm)
    # print("img coord x, y", x, y)
    x, y = metric_to_pixel(x, y, cpar)
    # print("metric to pixel x, y", x, y)
    return np.array([x, y])


def sorted_candidates_in_volume(
    center: np.ndarray, center_proj: np.ndarray, frm: Frame, run: TrackingRun
) -> List[Foundpix]:
    """Find candidates for continuing a particle's path in the search volume."""
    points = [Foundpix() for _ in range(frm.num_cams * MAX_CANDS)]
    # reset_foundpix_array(points, frm.num_cams * MAX_CANDS, frm.num_cams)

    # Search limits in image space
    right, left, down, up = searchquader(center, run.tpar, run.cpar, run.cal)

    # search in pix for candidates in the next_frame time step
    for cam in range(frm.num_cams):
        register_closest_neighbs(
            frm.targets[cam],
            frm.num_targets[cam],
            cam,
            center_proj[cam][0],
            center_proj[cam][1],
            left[cam],
            right[cam],
            up[cam],
            down[cam],
            points[cam * MAX_CANDS :],
            run.cpar,
        )

    # fill and sort candidate struct
    num_cands = sort_candidates_by_freq(points, frm.num_cams)
    if num_cands > 0:
        points = points[:num_cands] + [Foundpix(ftnr=TR_UNUSED)]
    else:
        points = []

    return points


def assess_new_position(
    pos: np.ndarray, frm: Frame, run: TrackingRun
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Determine the nearest target on each camera around a search position.

    #     and prepares the data structures accordingly with the determined target
    #     info or the unused flag value.

    #     Arguments:
    #     ---------
    #     pos - vec3d, the position around which to search.
    #     targ_pos - vec2d, the determined targets' respective positions.
    #     cand_inds - 2D array of integers, output buffer, the determined targets'
    #         index in the respective camera's target list.
    #     frm - frame object, holdin target data for the search position.
    #     run - TrackingRun object, scene information struct.

    #     Returns:
    #     -------
    #     Integer, the number of cameras where a suitable target was found.

    """
    # Output variables
    targ_pos = np.array(
        [[COORD_UNUSED, COORD_UNUSED] for _ in range(run.cpar.num_cams)]
    )
    cand_inds = np.array([[-1] * MAX_CANDS for _ in range(run.cpar.num_cams)])

    # Search rectangle limits
    left, right, up, down = ADD_PART, ADD_PART, ADD_PART, ADD_PART

    # for cam in range(run.cpar.num_cams):
    #     targ_pos[cam] = [COORD_UNUSED, COORD_UNUSED]

    for cam in range(run.cpar.num_cams):
        # Convert 3D search position to 2D pixel coordinates
        pixel = point_to_pixel(pos, run.cal[cam], run.cpar)
        print(f"pos {pos}")
        print(f"pixel {pixel}")

        # Nearest neighbor search
        num_cands = candsearch_in_pix_rest(
            frm.targets[cam],
            frm.num_targets[cam],
            pixel[0],
            pixel[1],
            left,
            right,
            up,
            down,
            cand_inds[cam],
            run.cpar,
        )

        if num_cands > 0:
            _ix = cand_inds[cam][0]  # first nearest neighbour
            targ_pos[cam][0] = frm.targets[cam][_ix].x
            targ_pos[cam][1] = frm.targets[cam][_ix].y

    valid_cams = 0

    for cam in range(run.cpar.num_cams):
        if (targ_pos[cam][0] != COORD_UNUSED) and (targ_pos[cam][1] != COORD_UNUSED):
            # Convert pixel coordinates to metric coordinates
            x, y = pixel_to_metric(targ_pos[cam][0], targ_pos[cam][1], run.cpar)

            # Apply additional transformations
            targ_pos[cam][0], targ_pos[cam][1] = dist_to_flat(
                x, y, run.cal[cam], run.flatten_tol
            )

            valid_cams += 1

    return valid_cams, targ_pos, cand_inds


def add_particle(frm: Frame, pos: np.ndarray, cand_inds: np.ndarray) -> None:
    """Add a new particle to the frame buffer."""
    num_parts = frm.num_parts
    ref_path_inf = frm.path_info[num_parts]
    ref_path_inf.x = pos
    ref_path_inf.reset_links()

    ref_corres = frm.correspond[num_parts]
    ref_targets = frm.targets

    for cam in range(frm.num_cams):
        ref_corres.p[cam] = CORRES_NONE

        # We always take the 1st candidate, apparently. Why did we fetch 4?
        if cand_inds[cam][0] != PT_UNUSED:
            _ix = cand_inds[cam][0]
            ref_targets[cam][_ix].tnr = num_parts
            ref_corres.p[cam] = _ix
            ref_corres.nr = num_parts

    frm.num_parts += 1


def track_forward_start(tr: TrackingRun):
    """Initialize the tracking frame buffer with the first frames.

    Arguments:
    ---------
    tr - an object holding the per-run tracking parameters, and
         a frame buffer with 4 positions.
    """
    # step = tr.seq_par.first

    # Prime the buffer with first three frames, fourth frame is read when we track
    for step in range(tr.seq_par.first, tr.seq_par.first + TR_BUFSPACE - 1):
        tr.fb.read_frame_at_end(step)
        tr.fb.fb_next()

    tr.fb.fb_prev()


def trackcorr_c_loop(run_info, step):
    """Sequence loop."""
    # Initialize variables
    philf = np.zeros((4, MAX_CANDS))
    # quali = 0
    diff_pos = np.empty((3,))

    # 7 reference points used in the algorithm, TODO: check if can reuse some
    # angle, acc, angle0, acc0, dl = 0.0, 0.0, 0.0, 0.0, 0.0
    # angle1, acc1 = 0.0, 0.0

    rr = 0.0

    _ix = 0  # For use in any of the complex index expressions below
    orig_parts = 0  # avoid infinite loop with particle addition set
    num_added = 0
    count1 = 0
    count2 = 0
    count3 = 0

    fb = run_info.fb
    cal = run_info.cal
    tpar = run_info.tpar
    vpar = run_info.vpar
    cpar = run_info.cpar
    curr_targets = fb.buf[1].targets

    v1 = np.zeros((cpar.num_cams, 2))  # volume center projection on cameras
    v2 = np.zeros((cpar.num_cams, 2))  # volume center projection on cameras

    # try to track correspondences from previous 0 - corp, variable h
    orig_parts = fb.buf[1].num_parts
    for h in range(orig_parts):
        X = np.zeros((6, 3))

        curr_path_inf = fb.buf[1].path_info[h]
        curr_corres = fb.buf[1].correspond[h]

        curr_path_inf.inlist = 0

        # 3D-position
        X[1] = vec_copy(curr_path_inf.x)

        # use information from previous to locate new search position
        # and to calculate values for search area
        if curr_path_inf.prev_frame >= 0:
            ref_path_inf = fb.buf[0].path_info[curr_path_inf.prev_frame]
            X[0] = vec_copy(ref_path_inf.x)
            X[2] = search_volume_center_moving(ref_path_inf.x, curr_path_inf.x)

            for j in range(fb.num_cams):
                v1[j] = point_to_pixel(X[2], cal[j], cpar)
        else:
            X[2] = vec_copy(X[1])
            for j in range(fb.num_cams):
                if curr_corres.p[j] == CORRES_NONE:
                    v1[j] = point_to_pixel(X[2], cal[j], cpar)
                else:
                    _ix = curr_corres.p[j]
                    v1[j] = np.r_[curr_targets[j][_ix].x, curr_targets[j][_ix].y]
                    # print(f"v1[{j}], {v1[j]}")

        # calculate search cuboid and reproject it to the image space
        w = sorted_candidates_in_volume(X[2], v1, fb.buf[2], run_info)
        if not w:  # empty
            continue

        # Continue to find candidates for the candidates.
        count2 += 1
        mm = 0
        while w[mm].ftnr != TR_UNUSED:  # counter1-loop
            # search for found corr of current the corr in next_frame with predicted location

            # found 3D-position
            ref_path_inf = fb.buf[2].path_info[w[mm].ftnr]
            X[3] = vec_copy(ref_path_inf.x)
            # print(f"X[3] {X[3]}")

            if curr_path_inf.prev_frame >= 0:
                # for j in range(3):
                #     X[5][j] = 0.5 * (5.0 * X[3][j] - 4.0 * X[1][j] + X[0][j])
                X[5] = 0.5 * (5 * X[3] - 4 * X[1] + X[0])
            else:
                X[5] = search_volume_center_moving(X[1], X[3])

            # print(f"X[5] {X[5]}")

            for j in range(fb.num_cams):
                v1[j] = point_to_pixel(X[5], cal[j], cpar)
                #  print(f"v1[{j}], {v1[j]}")

            # end of search in pix
            wn = sorted_candidates_in_volume(X[5], v1, fb.buf[3], run_info)
            if len(wn) > 0:  # not empty
                count3 += 1
                kk = 0
                while wn[kk].ftnr != TR_UNUSED:
                    # print(f" inside wn[{kk}].ftnr {wn[kk].ftnr}")
                    ref_path_inf = fb.buf[3].path_info[wn[kk].ftnr]
                    X[4] = vec_copy(ref_path_inf.x)
                    #  print(f"X[4] {X[4]}")

                    diff_pos = vec_subt(X[4], X[3])
                    #  print(f"inside kk loop {kk}")
                    # print(f"diff_pos {diff_pos}")

                    if pos3d_in_bounds(diff_pos, tpar):
                        angle1, acc1 = angle_acc(X[3], X[4], X[5])
                        if curr_path_inf.prev_frame >= 0:
                            angle0, acc0 = angle_acc(X[1], X[2], X[3])
                        else:
                            acc0 = acc1
                            angle0 = angle1

                        acc = (acc0 + acc1) / 2
                        angle = (angle0 + angle1) / 2
                        quali = wn[kk].freq + w[mm].freq

                        if (
                            acc < tpar.dacc
                            and angle < tpar.dangle
                            or acc < tpar.dacc / 10
                        ):
                            dl = (
                                vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[4], X[3])
                            ) / 2
                            rr = (
                                dl / run_info.lmax
                                + acc / tpar.dacc
                                + angle / tpar.dangle
                            ) / quali
                            curr_path_inf.register_link_candidate(rr, w[mm].ftnr)
                            # print(f"kk {kk}, rr {rr}, w[mm].ftnr {w[mm].ftnr}")

                    kk += 1  # End of searching 2nd-frame candidates.
                    # print(f"kk is {kk}")

            # creating new particle position,
            # reset img coord because of num_cams < 4
            # fix distance of 3 pixels to define xl,xr,yu,yd instead of searchquader
            # and search for unused candidates in next_frame time step

            quali, v2, philf = assess_new_position(X[5], fb.buf[3], run_info)
            # print(f"quali {quali}, v2 {v2}, philf {philf}")

            # quali >=2 means at least in two cameras
            # we found a candidate
            if quali >= 2:
                in_volume = 0  # inside volume

                dl, X[4] = point_position(v2, cpar.num_cams, cpar.mm, cal)

                # volume check
                if (
                    vpar.x_lay[0] < X[4][0]
                    and X[4][0] < vpar.x_lay[1]
                    and run_info.ymin < X[4][1]
                    and X[4][1] < run_info.ymax
                    and vpar.z_min_lay[0] < X[4][2]
                    and X[4][2] < vpar.z_max_lay[1]
                ):
                    in_volume = 1

                diff_pos = vec_subt(X[3], X[4])
                # print(f"second diff_pos {diff_pos}")

                if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                    angle, acc = angle_acc(X[3], X[4], X[5])
                    # print(f"angle {angle}, acc {acc}")

                    if acc < tpar.dacc and angle < tpar.dangle or acc < tpar.dacc / 10:
                        dl = (vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[4], X[3])) / 2
                        # print(f" dl {dl} ")
                        rr = (
                            dl / run_info.lmax + acc / tpar.dacc + angle / tpar.dangle
                        ) / (quali + w[mm].freq)

                        # print(f"acc {acc}, angle {angle}, quali {quali}, w[mm].freq {w[mm].freq}")
                        # print(f"rr {rr}, w[mm].ftnr {w[mm].ftnr}")
                        curr_path_inf.register_link_candidate(rr, w[mm].ftnr)

                        if tpar.add:
                            add_particle(fb.buf[3], X[4], philf)
                            num_added += 1

                in_volume = 0

            quali = 0

            # end of creating new particle position
            # ***************************************************************

            # try to link if kk is not found/good enough and prev exist
            if curr_path_inf.inlist == 0 and curr_path_inf.prev_frame >= 0:
                diff_pos = vec_subt(X[3], X[1])
                if pos3d_in_bounds(diff_pos, tpar):
                    angle, acc = angle_acc(X[1], X[2], X[3])
                    if (acc < tpar.dacc and angle < tpar.dangle) or (
                        acc < tpar.dacc / 10
                    ):
                        quali = w[mm].freq
                        dl = (vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[0], X[1])) / 2
                        rr = (
                            dl / run_info.lmax + acc / tpar.dacc + angle / tpar.dangle
                        ) / quali

                        # print(f"prev exists {mm}")
                        # print(f"rr {rr}, w[mm].ftnr {w[mm].ftnr}")
                        curr_path_inf.register_link_candidate(rr, w[mm].ftnr)

            del wn
            mm += 1  # increment mm

        # begin of inlist still zero
        if tpar.add:
            if curr_path_inf.inlist == 0 and curr_path_inf.prev_frame >= 0:
                quali, v2, philf = assess_new_position(X[2], fb.buf[2], run_info)
                if quali >= 2:
                    X[3] = vec_copy(X[2])
                    in_volume = 0
                    dl, X[3] = point_position(v2, fb.num_cams, cpar.mm, cal)

                    # in volume check
                    if (
                        vpar.x_lay[0] < X[3][0] < vpar.x_lay[1]
                        and run_info.ymin < X[3][1] < run_info.ymax
                        and vpar.z_min_lay[0] < X[3][2] < vpar.z_max_lay[1]
                    ):
                        in_volume = 1

                    diff_pos = vec_subt(X[2], X[3])
                    if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                        angle, acc = angle_acc(X[1], X[2], X[3])
                        if (acc < tpar.dacc and angle < tpar.dangle) or (
                            acc < tpar.dacc / 10
                        ):
                            dl = (
                                vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[0], X[1])
                            ) / 2
                            rr = (
                                dl / run_info.lmax
                                + acc / tpar.dacc
                                + angle / tpar.dangle
                            ) / quali
                            curr_path_inf.register_link_candidate(
                                rr, fb.buf[2].num_parts
                            )
                            add_particle(fb.buf[2], X[3], philf)
                            num_added += 1
                    in_volume = 0

        # end of inlist still zero
        # ***********************************

        del w

    # sort decis and give preliminary "finaldecis"
    for h in range(fb.buf[1].num_parts):
        curr_path_inf = fb.buf[1].path_info[h]

        if curr_path_inf.inlist > 0:
            curr_path_inf.decis, curr_path_inf.linkdecis = sort(
                curr_path_inf.inlist, curr_path_inf.decis, curr_path_inf.linkdecis
            )
            curr_path_inf.finaldecis = curr_path_inf.decis[0]
            curr_path_inf.next_frame = curr_path_inf.linkdecis[0]
            # print(f"curr_path_inf.finaldecis {curr_path_inf.finaldecis}")
            # print(f"curr_path_inf.next_frame {curr_path_inf.next_frame}")

    # create links with decision check
    for h in range(fb.buf[1].num_parts):
        curr_path_inf = fb.buf[1].path_info[h]

        if curr_path_inf.inlist > 0:
            ref_path_inf = fb.buf[2].path_info[curr_path_inf.next_frame]

            if ref_path_inf.prev_frame == PREV_NONE:
                # best choice wasn't used yet, so link is created
                ref_path_inf.prev_frame = h
                # print(f"link created {h}, ref_path_inf.next_frame {ref_path_inf.next_frame}")
            else:
                # best choice was already used by mega[2][mega[1][h].next_frame].prev_frame
                # check which is the better choice
                if (
                    fb.buf[1].path_info[ref_path_inf.prev_frame].finaldecis
                    > curr_path_inf.finaldecis
                ):
                    # remove link with prev
                    fb.buf[1].path_info[ref_path_inf.prev_frame].next_frame = NEXT_NONE
                    ref_path_inf.prev_frame = h
                else:
                    curr_path_inf.next_frame = NEXT_NONE

        if curr_path_inf.next_frame != NEXT_NONE:
            count1 += 1

    # end of creation of links with decision check
    print(
        f"step: {step}, curr: {fb.buf[1].num_parts}, next_frame: {fb.buf[2].num_parts}, \
            links: {count1}, lost: {fb.buf[1].num_parts - count1}, add: {num_added}"
    )

    # for the average of particles and links
    run_info.npart = run_info.npart + fb.buf[1].num_parts
    run_info.nlinks = run_info.nlinks + count1

    fb.fb_next()
    fb.write_frame_from_start(step)

    if step < run_info.seq_par.last - 2:
        fb.read_frame_at_end(step + 3, False)
    # end of sequence loop


def trackcorr_c_finish(run_info, step: int):
    """Close the links and write the last frame."""
    track_range = run_info.seq_par.last - run_info.seq_par.first
    npart, nlinks = run_info.npart / track_range, run_info.nlinks / track_range
    print(
        f"Average over sequence, particles: {npart:.1f}, links: {nlinks:.1f}, lost: {npart - nlinks:.1f}"
    )

    run_info.fb.fb_next()
    run_info.fb.write_frame_from_start(step)


def trackback_c(run_info: TrackingRun):
    """Trackback algorithm in C."""
    count1, count2, num_added, quali = 0, 0, 0, 0
    Ymin, Ymax, npart, nlinks = 0, 0, 0.0, 0.0

    philf = np.zeros((4, MAX_CANDS))
    X = np.empty((6, 3))
    n = np.empty((4, 2))
    v2 = np.empty((4, 2))

    fb = run_info.fb
    seq_par = run_info.seq_par
    tpar = run_info.tpar
    vpar = run_info.vpar
    cpar = run_info.cpar
    cal = run_info.cal

    step = 0

    # Prime the buffer with first frames
    for step in range(seq_par.last, seq_par.last - 4, -1):
        fb.read_frame_at_end(step, read_links=True)
        fb.fb_next()

    fb.fb_prev()

    # sequence loop
    for step in range(seq_par.last - 1, seq_par.first, -1):
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]

            # We try to find link only if the forward search failed to.
            if curr_path_inf.next_frame < 0 or curr_path_inf.prev_frame != -1:
                continue

            for j in range(6):
                X[j] = np.zeros(
                    3,
                )  # check it out

            curr_path_inf.inlist = 0

            # 3D-position of current particle
            X[1] = vec_copy(curr_path_inf.x)

            # use information from previous to locate new search position
            # and to calculate values for search area
            ref_path_inf = fb.buf[0].path_info[curr_path_inf.next_frame]
            X[0] = vec_copy(ref_path_inf.x)
            X[2] = search_volume_center_moving(ref_path_inf.x, curr_path_inf.x)

            for j in range(fb.num_cams):
                n[j] = point_to_pixel(X[2], cal[j], cpar)

            # calculate searchquader and reprojection in image space
            w = sorted_candidates_in_volume(X[2], n, fb.buf[2], run_info)

            if w is not None:
                count2 += 1

                i = 0
                while w[i].ftnr != TR_UNUSED:
                    ref_path_inf = fb.buf[2].path_info[w[i].ftnr]
                    X[3] = vec_copy(ref_path_inf.x)

                    diff_pos = vec_subt(X[1], X[3])
                    if pos3d_in_bounds(diff_pos, tpar):
                        angle, acc = angle_acc(X[1], X[2], X[3])

                        # *********************check link *****************************
                        if (
                            acc < tpar.dacc
                            and angle < tpar.dangle
                            or acc < tpar.dacc / 10
                        ):
                            dl = (
                                vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[0], X[1])
                            ) / 2
                            quali = w[i].freq
                            rr = (
                                dl / run_info.lmax
                                + acc / tpar.dacc
                                + angle / tpar.dangle
                            ) / quali
                            curr_path_inf.register_link_candidate(rr, w[i].ftnr)

                    i += 1

            del w

            # if old wasn't found try to create new particle position from rest
            if tpar.add:
                if curr_path_inf.inlist == 0:
                    quali, v2, philf = assess_new_position(X[2], fb.buf[2], run_info)
                    if quali >= 2:
                        # vec_copy(X[3], X[2])
                        in_volume = 0

                        _, X[3] = point_position(v2, fb.num_cams, cpar.mm, cal)

                        # volume check
                        if (
                            vpar.x_lay[0] < X[3][0] < vpar.x_lay[1]
                            and Ymin < X[3][1] < Ymax
                            and vpar.z_min_lay[0] < X[3][2] < vpar.z_max_lay[1]
                        ):
                            in_volume = 1

                        diff_pos = vec_subt(X[1], X[3])
                        if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                            angle, acc = angle_acc(X[1], X[2], X[3])

                            if (
                                acc < tpar.dacc
                                and angle < tpar.dangle
                                or acc < tpar.dacc / 10
                            ):
                                dl = (
                                    vec_diff_norm(X[1], X[3])
                                    + vec_diff_norm(X[0], X[1])
                                ) / 2
                                rr = (
                                    dl / run_info.lmax
                                    + acc / tpar.dacc
                                    + angle / tpar.dangle
                                ) / (quali)
                                curr_path_inf.register_link_candidate(
                                    rr, fb.buf[2].num_parts
                                )

                                add_particle(fb.buf[2], X[3], philf)

                        in_volume = 0

        # end of h-loop
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]

            if curr_path_inf.inlist > 0:
                curr_path_inf.decis, curr_path_inf.linkdecis = sort(
                    curr_path_inf.inlist, curr_path_inf.decis, curr_path_inf.linkdecis
                )

        # create links with decision check
        count1 = 0
        num_added = 0

        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]

            if curr_path_inf.inlist > 0:
                ref_path_inf = fb.buf[2].path_info[curr_path_inf.linkdecis[0]]

                if (
                    ref_path_inf.prev_frame == PREV_NONE
                    and ref_path_inf.next_frame == NEXT_NONE
                ):
                    curr_path_inf.finaldecis = curr_path_inf.decis[0]
                    curr_path_inf.prev_frame = curr_path_inf.linkdecis[0]
                    fb.buf[2].path_info[curr_path_inf.prev_frame].next_frame = h
                    num_added += 1

                if (
                    ref_path_inf.prev_frame != PREV_NONE
                    and ref_path_inf.next_frame == NEXT_NONE
                ):
                    X[0] = vec_copy(
                        fb.buf[0].path_info[curr_path_inf.next_frame].x,
                    )
                    X[1] = vec_copy(curr_path_inf.x)
                    X[3] = vec_copy(ref_path_inf.x)
                    X[4] = vec_copy(fb.buf[3].path_info[ref_path_inf.prev_frame].x)

                    for j in range(3):
                        X[5][j] = 0.5 * (5.0 * X[3][j] - 4.0 * X[1][j] + X[0][j])

                    angle, acc = angle_acc(X[3], X[4], X[5])

                    if (acc < tpar.dacc and angle < tpar.dangle) or (
                        acc < tpar.dacc / 10
                    ):
                        curr_path_inf.finaldecis = curr_path_inf.decis[0]
                        curr_path_inf.prev_frame = curr_path_inf.linkdecis[0]
                        fb.buf[2].path_info[curr_path_inf.prev_frame].next_frame = h
                        num_added += 1

            if curr_path_inf.prev_frame != PREV_NONE:
                count1 += 1

        npart += fb.buf[1].num_parts
        nlinks += count1

        fb.fb_next()
        fb.write_frame_from_start(step)

        if step > seq_par.first + 2:
            fb.read_frame_at_end(step - 3, read_links=True)

        print(
            "step: {}, curr: {}, next_frame: {}, links: {}, lost: {}, add: {}".format(
                step,
                fb.buf[1].num_parts,
                fb.buf[2].num_parts,
                count1,
                fb.buf[1].num_parts - count1,
                num_added,
            )
        )

    npart /= seq_par.last - seq_par.first - 1
    nlinks /= seq_par.last - seq_par.first - 1

    print(
        f"Average over sequence, particles: {npart:.1f}, links: {nlinks:.1f}, lost: {npart-nlinks:.1f}"
    )

    fb.fb_next()
    fb.write_frame_from_start(step)

    return nlinks


default_naming = {
    "corres": b"res/rt_is",
    "linkage": b"res/ptv_is",
    "prio": b"res/added",
}


# class Tracker:
#     """
#     Workflow: instantiate, call restart() to initialize the frame buffer, then.

#     call either ``step_forward()`` while it still return True, then call
#     ``finalize()`` to finish the run. Alternatively, ``full_forward()`` will
#     do all this for you.
#     """

#     def __init__(
#         self,
#         cpar: ControlPar,
#         vpar: VolumePar,
#         tpar: TrackPar,
#         spar: SequencePar,
#         cals: List[Calibration],
#         naming: dict,
#         flatten_tol: float = 0.0001,
#     ):
#         """
#         Initialize the tracker.

#         Arguments:
#         ---------
#         ControlPar cpar, VolumePar vpar, TrackPar tpar,
#         SequencePar spar - the usual parameter objects, as read from
#             anywhere.
#         cals - a list of Calibratiopn objects.
#         dict naming - a dictionary with naming rules for the frame buffer
#             files. See the ``default_naming`` member (which is the default).
#         """
#         # We need to keep a reference to the Python objects so that their
#         # allocations are not freed.
#         self._keepalive = (cpar, vpar, tpar, spar, cals)

#         self.run_info = tr_new(
#             spar,
#             tpar,
#             vpar,
#             cpar,
#             TR_BUFSPACE,
#             MAX_TARGETS,
#             naming["corres"],
#             naming["linkage"],
#             naming["prio"],
#             cals,
#             flatten_tol,
#         )
#         self.step = self.run_info.seq_par.first

#     def restart(self):
#         """
#         Prepare a tracking run. Sets up initial buffers and performs the.

#         one-time calculations used throughout the loop.
#         """
#         self.step = self.run_info.seq_par.first
#         track_forward_start(self.run_info)

#     def step_forward(self):
#         """Perform one tracking step for the current frame of iteration."""
#         if self.step >= self.run_info.seq_par.last:
#             return False

#         trackcorr_c_loop(self.run_info, self.step)
#         self.step += 1
#         return True

#     def finalize(self):
#         """Finish a tracking run."""
#         trackcorr_c_finish(self.run_info, self.step)

#     def full_forward(self):
#         """Do a full tracking run from restart to finalize."""
#         track_forward_start(self.run_info)
#         for step in range(self.run_info.seq_par.first, self.run_info.seq_par.last):
#             trackcorr_c_loop(self.run_info, step)
#         trackcorr_c_finish(self.run_info, self.run_info.seq_par.last)

#     def full_backward(self):
#         """Do a full backward run on existing tracking results. so make sure.

#         results exist or it will explode in your face.
#         """
#         trackback_c(self.run_info)

#     def current_step(self):
#         return self.step
