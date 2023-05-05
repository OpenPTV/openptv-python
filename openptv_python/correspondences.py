"""Correspondences."""
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .calibration import Calibration
from .constants import CORRES_NONE, MAX_TARGETS, MAXCAND, NMAX, PT_UNUSED
from .epi import Coord2d, epi_mm, find_candidate
from .parameters import ControlPar, VolumePar
from .tracking_frame_buf import Frame, Target, n_tupel


@dataclass
class Correspond:
    """Correspondence candidate data structure."""

    p1: int  # point number of master point
    n: int  # number of candidates
    p2: np.ndarray = field(
        default=np.empty(MAXCAND, dtype=int)
    )  # point numbers of candidates
    corr: np.ndarray = field(
        default=np.empty(MAXCAND, dtype=float)
    )  # feature-based correlation coefficient
    dist: np.ndarray = field(
        default=np.empty(MAXCAND, dtype=float)
    )  # distance perpendicular to epipolar line


def deallocate_target_usage_marks(tusage, num_cams):
    """Deallocate target usage marks."""
    for cam in range(num_cams):
        del tusage[cam][:]
    del tusage


# We can use numpy instead of this
# def safely_allocate_target_usage_marks(num_cams, nmax=NMAX):
#     """Safely allocate target usage marks."""
#     tusage = []

#     try:
#         for _ in range(num_cams):
#             tusage.append([0 for _ in range(nmax)])

#     except MemoryError as exc:
#         raise MemoryError("Memory allocation error in safely_allocate_target_usage_marks") from exc

#     return tusage


# Python should take care of memory collection
# def deallocate_adjacency_lists(
#     lists: List[List[List[Correspond]]], num_cams: int
# ) -> None:
#     """ Deallocate adjacency lists."""
#     for c1 in range(num_cams - 1):
#         for c2 in range(c1 + 1, num_cams):
#             for i in range(len(lists[c1][c2])):
#                 lists[c1][c2][i] = None
#             lists[c1][c2] = None


def safely_allocate_adjacency_lists(
    num_cams: int, target_counts: List[int]
) -> List[List[List[Correspond]]]:
    """Safely allocate adjacency lists.

    Allocates and initializes pairwise
    adjacency lists. If an error occurs, cleans up memory and returns 0,
    otherwise returns 1.

    Arguments:
    ---------
    lists - an existing array of arrays of pointers,
        where allocations will be stored.
    num_cams - number of cameras to handle (up to 4).
    target_counts - an array holding the number of targets in each camera.

    """
    return [
        [
            [Correspond(n=0, p1=0) for _ in range(target_counts[c1])]
            for _ in range(c1 + 1, num_cams)
        ]
        for c1 in range(num_cams - 1)
    ]

    # for c1 in range(num_cams - 1):
    #     lists.append([])
    #     for _ in range(c1 + 1, num_cams):
    #         list_of_correspondences =
    #         lists[c1].append(list_of_correspondences)

    # return lists


def four_camera_matching(
    corr_list, base_target_count, accept_corr, scratch, scratch_size
) -> List[List[List[Correspond]]]:
    """Four-camera matching."""
    matched = 0

    for i in range(base_target_count):
        p1 = corr_list[0][1][i].p1
        for j in range(corr_list[0][1][i].n):
            p2 = corr_list[0][1][i].p2[j]
            for k in range(corr_list[0][2][i].n):
                p3 = corr_list[0][2][i].p2[k]
                for ll in range(corr_list[0][3][i].n):
                    p4 = corr_list[0][3][i].p2[ll]

                    for m in range(corr_list[1][2][p2].n):
                        p31 = corr_list[1][2][p2].p2[m]
                        if p3 != p31:
                            continue

                        for n in range(corr_list[1][3][p2].n):
                            p41 = corr_list[1][3][p2].p2[n]
                            if p4 != p41:
                                continue

                            for o in range(corr_list[2][3][p3].n):
                                p42 = corr_list[2][3][p3].p2[o]
                                if p4 != p42:
                                    continue

                                corr = (
                                    corr_list[0][1][i].corr[j]
                                    + corr_list[0][2][i].corr[k]
                                    + corr_list[0][3][i].corr[ll]
                                    + corr_list[1][2][p2].corr[m]
                                    + corr_list[1][3][p2].corr[n]
                                    + corr_list[2][3][p3].corr[o]
                                ) / (
                                    corr_list[0][1][i].dist[j]
                                    + corr_list[0][2][i].dist[k]
                                    + corr_list[0][3][i].dist[ll]
                                    + corr_list[1][2][p2].dist[m]
                                    + corr_list[1][3][p2].dist[n]
                                    + corr_list[2][3][p3].dist[o]
                                )

                                if corr <= accept_corr:
                                    continue

                                # accept as preliminary match
                                scratch[matched].p[0] = p1
                                scratch[matched].p[1] = p2
                                scratch[matched].p[2] = p3
                                scratch[matched].p[3] = p4
                                scratch[matched].corr = corr

                                matched += 1
                                if matched == scratch_size:
                                    print("Overflow in correspondences.")
                                    return matched

    return matched


def three_camera_matching(
    corr_list, num_cams, target_counts, accept_corr, scratch, scratch_size, tusage
):
    """Three-camera matching."""
    matched = 0
    nmax = NMAX

    for i1 in range(num_cams - 2):
        for i in range(target_counts[i1]):
            for i2 in range(i1 + 1, num_cams - 1):
                p1 = corr_list[i1][i2][i].p1
                if p1 > nmax or tusage[i1][p1] > 0:
                    continue

                for j in range(corr_list[i1][i2][i].n):
                    p2 = corr_list[i1][i2][i].p2[j]
                    if p2 > nmax or tusage[i2][p2] > 0:
                        continue

                    for i3 in range(i2 + 1, num_cams):
                        for k in range(corr_list[i1][i3][i].n):
                            p3 = corr_list[i1][i3][i].p2[k]
                            if p3 > nmax or tusage[i3][p3] > 0:
                                continue

                            indices = np.where(corr_list[i2][i3][p2].p2 == p3)[0]
                            if indices.size == 0:
                                continue

                            m = indices[0]
                            corr = (
                                corr_list[i1][i2][i].corr[j]
                                + corr_list[i1][i3][i].corr[k]
                                + corr_list[i2][i3][p2].corr[m]
                            ) / (
                                corr_list[i1][i2][i].dist[j]
                                + corr_list[i1][i3][i].dist[k]
                                + corr_list[i2][i3][p2].dist[m]
                            )

                            if corr <= accept_corr:
                                continue

                            p = np.full(num_cams, -2)
                            p[i1], p[i2], p[i3] = p1, p2, p3
                            scratch[matched].p = p
                            scratch[matched].corr = corr

                            matched += 1
                            if matched == scratch_size:
                                print("Overflow in correspondences.\n")
                                return matched
    return matched


def consistent_pair_matching(
    corr_list, num_cams, target_counts, accept_corr, scratch, scratch_size, tusage
) -> int:
    """Find consistent pairs of correspondences."""
    matched = 0
    # nmax = np.inf
    nmax = NMAX
    for i1 in range(num_cams - 1):
        for i2 in range(i1 + 1, num_cams):
            for i in range(target_counts[i1]):
                p1 = corr_list[i1][i2][i].p1
                if p1 > nmax or tusage[i1][p1] > 0:
                    continue

                if corr_list[i1][i2][i].n != 1:
                    continue

                p2 = corr_list[i1][i2][i].p2[0]
                if p2 > nmax or tusage[i2][p2] > 0:
                    continue

                corr = corr_list[i1][i2][i].corr[0] / corr_list[i1][i2][i].dist[0]
                if corr <= accept_corr:
                    continue

                for n in range(num_cams):
                    scratch[matched].p[n] = -2

                scratch[matched].p[i1] = p1
                scratch[matched].p[i2] = p2
                scratch[matched].corr = corr

                matched += 1
                if matched == scratch_size:
                    print("Overflow in correspondences.\n")
                    return matched

    return matched


def match_pairs(corrected, frm, vpar, cpar, calib) -> List[List[List[Correspond]]]:
    """Match pairs of cameras."""
    # MAXCAND = 100

    corr_list = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)

    for i1 in range(cpar.num_cams - 1):
        for i2 in range(i1 + 1, cpar.num_cams):
            for i in range(frm.num_targets[i1]):
                if corrected[i1][i].x == PT_UNUSED:
                    continue

                xa12, ya12, xb12, yb12 = epi_mm(
                    corrected[i1][i].x,
                    corrected[i1][i].y,
                    calib[i1],
                    calib[i2],
                    cpar.mm,
                    vpar,
                )

                # origin point in the corr_list
                corr_list[i1][i2][i].p1 = i
                pt1 = corrected[i1][i].pnr

                # search for a conjugate point in corrected[i2]
                cand = [Correspond(0, 0) for _ in range(MAXCAND)]
                count = find_candidate(
                    corrected[i2],
                    frm.targets[i2],
                    frm.num_targets[i2],
                    xa12,
                    ya12,
                    xb12,
                    yb12,
                    frm.targets[i1][pt1].n,
                    frm.targets[i1][pt1].nx,
                    frm.targets[i1][pt1].ny,
                    frm.targets[i1][pt1].sumg,
                    cand,
                    vpar,
                    cpar,
                    calib[i2],
                )

                # write all corresponding candidates to the preliminary corr_list of correspondences
                if count > MAXCAND:
                    count = MAXCAND

                for j in range(count):
                    corr_list[i1][i2][i].p2[j] = cand[j].pnr
                    corr_list[i1][i2][i].corr[j] = cand[j].corr
                    corr_list[i1][i2][i].dist[j] = cand[j].tol

                corr_list[i1][i2][i].n = count

    return corr_list


def take_best_candidates(src, dst, num_cams, num_cands, tusage):
    """Take the best candidates from the corr_list of candidates."""
    taken = 0

    # sort candidates by match quality (.corr)
    src.sort(key=lambda x: x.corr, reverse=True)

    # take quadruplets from the top to the bottom of the sorted corr_list
    # only if none of the points has already been used
    for cand in src:
        has_used_target = False
        for cam in range(num_cams):
            tnum = cand.p[cam]

            # if any correspondence in this camera, check that target is free
            if tnum > -1 and tusage[cam][tnum] > 0:
                has_used_target = True
                break

        if has_used_target:
            continue

        # Only now can we commit to marking used targets.
        for cam in range(num_cams):
            tnum = cand.p[cam]
            if tnum > -1:
                tusage[cam][tnum] += 1

        dst[taken] = cand
        taken += 1

    return taken


def py_correspondences(
    img_pts: List[Target],
    flat_coords: List[Coord2d],
    calib: List[Calibration],
    vparam: VolumePar,
    cparam: ControlPar,
) -> Tuple[Tuple, Tuple, int]:
    """
    Get the correspondences for each clique size.

    Arguments:
    ---------
    img_pts - a list of c := len(cals), containing TargetArray objects, each
        with the target coordinates of n detections in the respective image.
        The target arrays are clobbered: returned arrays have the tnr property
        set. the pnr property should be set to the target index in its array.
    flat_coords - a list of MatchedCoordinates objects, one per camera, holding
        the x-sorted flat-coordinates conversion of the respective image
        targets.
    cals - a list of Calibration objects, each for the camera taking one image.
    VolumeParams vparam - an object holding observed volume size parameters.
    ControlParams cparam - an object holding general control parameters.

    Returns:
    -------
    sorted_pos - a tuple of (c,?,2) arrays, each with the positions in each of
        c image planes of points belonging to quadruplets, triplets, pairs
        found.
    sorted_corresp - a tuple of (c,?) arrays, each with the point identifiers
        of targets belonging to a quad/trip/etc per camera.
    num_targs - total number of targets (must be greater than the sum of
        previous 3).
    """
    num_cams = cparam.num_cams
    frm = Frame(num_cams, MAX_TARGETS)
    corrected = [Coord2d() for _ in range(num_cams)]

    # Special case of a single camera, follow the single_cam_correspondence docstring
    if num_cams == 1:
        sorted_pos, sorted_corresp, num_targs = single_cam_correspondences(
            img_pts, flat_coords
        )
        return sorted_pos, sorted_corresp, num_targs

    # cdef:
    #     calibration **calib = <calibration **> malloc(
    #         num_cams * sizeof(calibration *))
    #     coord_2d **corrected = <coord_2d **> malloc(
    #         num_cams * sizeof(coord_2d *))
    #     frame frm

    # np.ndarray[ndim=2, dtype=np.int_t] clique_ids
    # np.ndarray[ndim=3, dtype=np.float64_t] clique_targs

    # Return buffers:
    # int *match_counts = <int *> malloc(num_cams * sizeof(int))
    # n_tupel *corresp_buf

    match_counts = [0] * num_cams
    corresp_buf = []  # of n_tupel

    # Initialize frame partially, without the extra momory used by init_frame.
    # frm.targets = <target**> calloc(num_cams, sizeof(target*))
    # frm.num_targets = <int *> calloc(num_cams, sizeof(int))
    frm.targets = [Target() for _ in range(num_cams)]
    frm.num_targets = [0] * num_cams

    for cam in range(num_cams):
        # calib[cam] = (<Calibration>cals[cam])._calibration
        # frm.targets[cam] = (<TargetArray>img_pts[cam])._tarr
        frm.targets[cam] = [Target() for _ in range(len(img_pts[cam]))]
        frm.num_targets[cam] = len(img_pts[cam])
        # corrected[cam] = (<MatchedCoords>flat_coords[cam]).buf
        # corrected[cam] = [MatchedCoords() for _ in range(len(flat_coords[cam]))]

    # The biz:
    corresp_buf = correspondences(frm, corrected, vparam, cparam, calib, match_counts)

    # Distribute data to return structures:
    sorted_pos = [None] * (num_cams - 1)
    sorted_corresp = [None] * (num_cams - 1)
    last_count = 0

    for clique_type in range(num_cams - 1):
        num_points = match_counts[4 - num_cams + clique_type]  # for 1-4 cameras
        clique_targs = np.full((num_cams, num_points, 2), PT_UNUSED, dtype=np.float64)
        clique_ids = np.full((num_cams, num_points), CORRES_NONE, dtype=np.int_)

        # Trace back the pixel target properties through the flat metric
        # intermediary that's x-sorted.
        for cam in range(num_cams):
            for pt in range(num_points):
                geo_id = corresp_buf[pt + last_count].p[cam]
                if geo_id < 0:
                    continue

                p1 = corrected[cam][geo_id].pnr
                clique_ids[cam, pt] = p1

                if p1 > -1:
                    targ = img_pts[cam][p1]
                    clique_targs[cam, pt, 0] = targ.x
                    clique_targs[cam, pt, 1] = targ.y

        last_count += num_points
        sorted_pos[clique_type] = clique_targs
        sorted_corresp[clique_type] = clique_ids

    # Clean up.
    num_targs = match_counts[num_cams - 1]

    return sorted_pos, sorted_corresp, num_targs


def correspondences(
    frm: Frame,
    corrected: List[List[Coord2d]],
    vpar: VolumePar,
    cpar: ControlPar,
    calib: List[List[Calibration]],
    match_counts: List[int],
) -> List[n_tupel]:
    """Find correspondences between cameras."""
    nmax = NMAX

    # Allocation of scratch buffers for internal tasks and return-value space
    con0 = [
        n_tupel(p=[-1] * cpar.num_cams, corr=0.0) for _ in range(nmax * cpar.num_cams)
    ]
    con = [
        n_tupel(p=[-1] * cpar.num_cams, corr=0.0) for _ in range(nmax * cpar.num_cams)
    ]

    tim = np.zeros((cpar.num_cams, nmax), dtype=np.int32)

    # allocate memory for lists of correspondences
    corr_list = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)

    # if I understand correctly, the number of matches cannot be more than the number of
    # targets (dots) in the first image. In the future we'll replace it by the maximum
    # number of targets in any image (if we will implement the cyclic search) but for
    # a while we always start with the cam1

    # Generate adjacency lists: mark candidates for correspondence.
    # matching 1 -> 2,3,4 + 2 -> 3,4 + 3 -> 4
    corr_list = match_pairs(corrected, frm, vpar, cpar, calib)

    # search consistent quadruplets in the corr_list
    if cpar.num_cams == 4:
        match0 = four_camera_matching(
            corr_list, frm.num_targets[0], vpar.corrmin, con0, 4 * nmax
        )

        match_counts[0] = take_best_candidates(con0, con, cpar.num_cams, match0, tim)
        match_counts[3] += match_counts[0]

    # search consistent triplets: 123, 124, 134, 234
    if (cpar.num_cams == 4 and cpar.allCam_flag == 0) or cpar.num_cams == 3:
        match0 = three_camera_matching(
            corr_list, cpar.num_cams, frm.num_targets, vpar.corrmin, con0, 4 * nmax, tim
        )

        match_counts[1] = take_best_candidates(
            con0, con[match_counts[3] :], cpar.num_cams, match0, tim
        )
        match_counts[3] += match_counts[1]

    # Search consistent pairs: 12, 13, 14, 23, 24, 34
    if cpar.num_cams > 1 and cpar.allCam_flag == 0:
        match0 = consistent_pair_matching(
            corr_list, cpar.num_cams, frm.num_targets, vpar.corrmin, con0, 4 * nmax, tim
        )
        match_counts[2] = take_best_candidates(
            con0, con[match_counts[3] :], cpar.num_cams, match0, tim
        )
        match_counts[3] += match_counts[2]

    # Give each used pix the correspondence number
    for i in range(match_counts[3]):
        for j in range(cpar.num_cams):
            # Skip cameras without a correspondence obviously.
            if con[i].p[j] < 0:
                continue

            p1 = corrected[j][con[i].p[j]].pnr
            if p1 > -1 and p1 < 1202590843:
                frm.targets[j][p1].tnr = i

    # Free all other allocations
    # deallocate_adjacency_lists(corr_list, cpar.num_cams)
    # deallocate_target_usage_marks(tim, cpar.num_cams)
    # del con0

    return con


def single_cam_correspondences(img_pts: List[Target], flat_coords: List[float]):
    """
    Single camera correspondence is not a real correspondence, it will be only a projection.

    of a 2D target from the image space into the 3D position, x,y,z using epi_mm_2d
    function. Here we only update the pointers of the targets and return it in a proper format.

    Arguments:
    ---------
    img_pts - a corr_list of c := len(cals), containing TargetArray objects, each
        with the target coordinates of n detections in the respective image.
        The target arrays are clobbered: returned arrays have the tnr property
        set. the pnr property should be set to the target index in its array.
    flat_coords - a corr_list of MatchedCoordinates objects, one per camera, holding
        the x-sorted flat-coordinates conversion of the respective image
        targets.

    Returns:
    -------
    sorted_pos - a tuple of (c,?,2) arrays, each with the positions in each of
        c image planes of points belonging to quadruplets, triplets, pairs
        found.
    sorted_corresp - a tuple of (c,?) arrays, each with the point identifiers
        of targets belonging to a quad/trip/etc per camera.
    num_targs - total number of targets (must be greater than the sum of
        previous 3).
    """
    # cdef:
    #     int pt, num_points
    #     coord_2d *corrected = <coord_2d *> malloc(sizeof(coord_2d *))

    corrected = flat_coords[0].buf

    num_points = len(img_pts[0])

    clique_targs = np.full((1, num_points, 2), PT_UNUSED, dtype=np.float64)
    clique_ids = np.full((1, num_points), CORRES_NONE, dtype=np.int_)

    # Trace back the pixel target properties through the flat metric
    # intermediary that's x-sorted.
    for pt in range(num_points):
        # From Beat code (issue #118) pix[0][geo[0][i].pnr].tnr=i;

        p1 = corrected[pt].pnr
        clique_ids[0, pt] = p1

        if p1 > -1:
            targ = img_pts[0][p1]
            clique_targs[0, pt, 0] = targ.x
            clique_targs[0, pt, 1] = targ.y
            # we also update the tnr, see docstring of correspondences
            targ.tnr = pt

    sorted_pos = [clique_targs]
    sorted_corresp = [clique_ids]

    return sorted_pos, sorted_corresp, num_points


# def consistent_pair_matching(
#     list, num_cams, target_counts, accept_corr, scratch, scratch_size, tusage
# ):
#     """Search consistent pairs: 12, 13, 14, 23, 24, 34."""
#     matched = 0

#     for i1 in range(num_cams - 1):
#         for i2 in range(i1 + 1, num_cams):
#             for i in range(target_counts[i1]):
#                 p1 = list[i1][i2][i].p1
#                 if p1 > NMAX or tusage[i1][p1] > 0:
#                     continue

#                 # if the candidate is the only one, there is no ambiguity.
#                 # we only take unambiguous pairs.
#                 if list[i1][i2][i].n != 1:
#                     continue

#                 p2 = list[i1][i2][i].p2[0]
#                 if p2 > NMAX or tusage[i2][p2] > 0:
#                     continue

#                 corr = list[i1][i2][i].corr[0] / list[i1][i2][i].dist[0]
#                 if corr <= accept_corr:
#                     continue

#                 # This to catch the excluded cameras
#                 for n in range(num_cams):
#                     scratch[matched].p[n] = -2

#                 scratch[matched].p[i1] = p1
#                 scratch[matched].p[i2] = p2
#                 scratch[matched].corr = corr

#                 matched += 1
#                 if matched == scratch_size:
#                     print("Overflow in correspondences.")
#                     return matched

#     return matched


def safely_allocate_target_usage_marks(num_cams):
    """
    Allocate space for per-camera arrays marking whether a certain target was used.

    If some allocation arrays marking whether a certain target was used. If some allocation
    failed, it cleans up memory and returns None. Allocated arrays are zeroed
    out initially by the C library.
    """
    try:
        tusage = [[0] * NMAX for _ in range(num_cams)]
        return tusage
    except MemoryError:
        return None
