"""Correspondences."""
from dataclasses import dataclass
from typing import List

import numpy as np

from .calibration import Calibration
from .constants import COORD_UNUSED, CORRES_NONE, MAXCAND, NMAX, PT_UNUSED
from .epi import Candidate, Coord2d, epi_mm, find_candidate
from .parameters import ControlPar, VolumePar
from .tracking_frame_buf import Frame, TargetArray
from .trafo import dist_to_flat, pixel_to_metric


@dataclass
class n_tupel:
    """n_tupel data structure."""

    p: List[int] = [0, 0, 0, 0]
    corr: float = 0.0


@dataclass
class Correspond:
    """Correspondence candidate data structure."""

    p1: int = 0  # point number of master point
    n: int = 0  # number of candidates
    p2: List[int] = [0] * MAXCAND  # point numbers of candidates
    corr: List[float] = [0.0] * MAXCAND  # feature-based correlation coefficient
    dist: List[float] = [0.0] * MAXCAND  # distance perpendicular to epipolar line


def quicksort_con(con, num):
    if num > 0:
        qs_con(con, 0, num - 1)


def qs_con(con, left, right):
    if left >= right:
        return

    pivot = con[(left + right) // 2].corr
    i, j = left, right
    while i <= j:
        while con[i].corr < pivot:
            i += 1
        while con[j].corr > pivot:
            j -= 1
        if i <= j:
            con[i], con[j] = con[j], con[i]
            i += 1
            j -= 1

    qs_con(con, left, j)
    qs_con(con, i, right)


def quicksort_target_y(pix, num):
    qs_target_y(pix, 0, num - 1)


def qs_target_y(pix, left, right):
    if left >= right:
        return

    pivot = pix[(left + right) // 2].y
    i, j = left, right
    while i <= j:
        while pix[i].y < pivot:
            i += 1
        while pix[j].y > pivot:
            j -= 1
        if i <= j:
            pix[i], pix[j] = pix[j], pix[i]
            i += 1
            j -= 1

    qs_target_y(pix, left, j)
    qs_target_y(pix, i, right)


def quicksort_coord2d_x(crd, num):
    qs_coord2d_x(crd, 0, num - 1)


def qs_coord2d_x(crd, left, right):
    if left >= right:
        return

    pivot = crd[(left + right) // 2].x
    i, j = left, right
    while i <= j:
        while crd[i].x < pivot and i < right:
            i += 1
        while pivot < crd[j].x and j > left:
            j -= 1
        if i <= j:
            crd[i], crd[j] = crd[j], crd[i]
            i += 1
            j -= 1

    qs_coord2d_x(crd, left, j)
    qs_coord2d_x(crd, i, right)


def deallocate_target_usage_marks(tusage, num_cams):
    for cam in range(num_cams):
        del tusage[cam][:]
    del tusage


def safely_allocate_target_usage_marks(num_cams, nmax=NMAX):
    tusage = []
    error = False

    try:
        for cam in range(num_cams):
            tusage_cam = [0] * nmax
            tusage.append(tusage_cam)
    except MemoryError:
        error = True

    if error:
        deallocate_target_usage_marks(tusage, num_cams)
        return None
    else:
        return tusage


def deallocate_adjacency_lists(
    lists: List[List[List[Correspond()]]], num_cams: int
) -> None:
    for c1 in range(num_cams - 1):
        for c2 in range(c1 + 1, num_cams):
            for i in range(len(lists[c1][c2])):
                lists[c1][c2][i] = None
            lists[c1][c2] = None


def safely_allocate_adjacency_lists(
    lists: List[List[List[Correspond()]]], num_cams: int, target_counts: List[int]
) -> int:
    error = False

    for c1 in range(num_cams - 1):
        for c2 in range(c1 + 1, num_cams):
            if not error:
                try:
                    lists[c1][c2] = [Correspond() for i in range(target_counts[c1])]
                except MemoryError:
                    error = True
                    continue

                for edge in range(target_counts[c1]):
                    lists[c1][c2][edge].n = 0
                    lists[c1][c2][edge].p1 = 0
            else:
                lists[c1][c2] = None

    if error:
        deallocate_adjacency_lists(lists, num_cams)
        return 0
    else:
        return 1


def four_camera_matching(list, base_target_count, accept_corr, scratch, scratch_size):
    matched = 0

    for i in range(base_target_count):
        p1 = list[0][1][i].p1
        for j in range(list[0][1][i].n):
            p2 = list[0][1][i].p2[j]
            for k in range(list[0][2][i].n):
                p3 = list[0][2][i].p2[k]
                for ll in range(list[0][3][i].n):
                    p4 = list[0][3][i].p2[ll]

                    for m in range(list[1][2][p2].n):
                        p31 = list[1][2][p2].p2[m]
                        if p3 != p31:
                            continue

                        for n in range(list[1][3][p2].n):
                            p41 = list[1][3][p2].p2[n]
                            if p4 != p41:
                                continue

                            for o in range(list[2][3][p3].n):
                                p42 = list[2][3][p3].p2[o]
                                if p4 != p42:
                                    continue

                                corr = (
                                    list[0][1][i].corr[j]
                                    + list[0][2][i].corr[k]
                                    + list[0][3][i].corr[ll]
                                    + list[1][2][p2].corr[m]
                                    + list[1][3][p2].corr[n]
                                    + list[2][3][p3].corr[o]
                                ) / (
                                    list[0][1][i].dist[j]
                                    + list[0][2][i].dist[k]
                                    + list[0][3][i].dist[ll]
                                    + list[1][2][p2].dist[m]
                                    + list[1][3][p2].dist[n]
                                    + list[2][3][p3].dist[o]
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
    list, num_cams, target_counts, accept_corr, scratch, scratch_size, tusage
):
    matched = 0
    nmax = NMAX

    for i1 in range(num_cams - 2):
        for i in range(target_counts[i1]):
            for i2 in range(i1 + 1, num_cams - 1):
                p1 = list[i1][i2][i].p1
                if p1 > nmax or tusage[i1][p1] > 0:
                    continue

                for j in range(list[i1][i2][i].n):
                    p2 = list[i1][i2][i].p2[j]
                    if p2 > nmax or tusage[i2][p2] > 0:
                        continue

                    for i3 in range(i2 + 1, num_cams):
                        for k in range(list[i1][i3][i].n):
                            p3 = list[i1][i3][i].p2[k]
                            if p3 > nmax or tusage[i3][p3] > 0:
                                continue

                            indices = np.where(list[i2][i3][p2].p2 == p3)[0]
                            if indices.size == 0:
                                continue

                            m = indices[0]
                            corr = (
                                list[i1][i2][i].corr[j]
                                + list[i1][i3][i].corr[k]
                                + list[i2][i3][p2].corr[m]
                            ) / (
                                list[i1][i2][i].dist[j]
                                + list[i1][i3][i].dist[k]
                                + list[i2][i3][p2].dist[m]
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
    list, num_cams, target_counts, accept_corr, scratch, scratch_size, tusage
):
    matched = 0
    # nmax = np.inf
    nmax = NMAX
    for i1 in range(num_cams - 1):
        for i2 in range(i1 + 1, num_cams):
            for i in range(target_counts[i1]):
                p1 = list[i1][i2][i].p1
                if p1 > nmax or tusage[i1][p1] > 0:
                    continue

                if list[i1][i2][i].n != 1:
                    continue

                p2 = list[i1][i2][i].p2[0]
                if p2 > nmax or tusage[i2][p2] > 0:
                    continue

                corr = list[i1][i2][i].corr[0] / list[i1][i2][i].dist[0]
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


def match_pairs(list, corrected, frm, vpar, cpar, calib):
    MAXCAND = 100
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

                # origin point in the list
                list[i1][i2][i].p1 = i
                pt1 = corrected[i1][i].pnr

                # search for a conjugate point in corrected[i2]
                cand = [Candidate() for _ in range(MAXCAND)]
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

                # write all corresponding candidates to the preliminary list of correspondences
                if count > MAXCAND:
                    count = MAXCAND

                for j in range(count):
                    list[i1][i2][i].p2[j] = cand[j].pnr
                    list[i1][i2][i].corr[j] = cand[j].corr
                    list[i1][i2][i].dist[j] = cand[j].tol

                list[i1][i2][i].n = count


def take_best_candidates(src, dst, num_cams, num_cands, tusage):
    taken = 0

    # sort candidates by match quality (.corr)
    src.sort(key=lambda x: x.corr, reverse=True)

    # take quadruplets from the top to the bottom of the sorted list
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


def correspondences(
    frm: Frame,
    corrected: List[List[Coord2d]],
    vpar: VolumePar,
    cpar: ControlPar,
    calib: List[List[Calibration]],
    match_counts: List[int],
) -> List[n_tupel]:
    # nmax = 1000
    nmax = NMAX

    # Allocation of scratch buffers for internal tasks and return-value space
    con0 = (nmax * cpar.num_cams) * [n_tupel(p=[-1] * cpar.num_cams, corr=0.0)]
    con = (nmax * cpar.num_cams) * [n_tupel(p=[-1] * cpar.num_cams, corr=0.0)]

    tim = safely_allocate_target_usage_marks(cpar.num_cams)
    if tim is None:
        print("out of memory")
        return None

    # allocate memory for lists of correspondences
    list = [[None] * 4 for _ in range(4)]
    if safely_allocate_adjacency_lists(list, cpar.num_cams, frm.num_targets) == 0:
        print("list is not allocated")
        deallocate_target_usage_marks(tim, cpar.num_cams)
        return None

    # if I understand correctly, the number of matches cannot be more than the number of
    # targets (dots) in the first image. In the future we'll replace it by the maximum
    # number of targets in any image (if we will implement the cyclic search) but for
    # a while we always start with the cam1
    for i in range(nmax):
        for j in range(cpar.num_cams):
            con0[i].p[j] = -1
        con0[i].corr = 0.0

    for i in range(4):
        match_counts[i] = 0

    # Generate adjacency lists: mark candidates for correspondence.
    # matching 1 -> 2,3,4 + 2 -> 3,4 + 3 -> 4
    match_pairs(list, corrected, frm, vpar, cpar, calib)

    # search consistent quadruplets in the list
    if cpar.num_cams == 4:
        match0 = four_camera_matching(
            list, frm.num_targets[0], vpar.corrmin, con0, 4 * nmax
        )

        match_counts[0] = take_best_candidates(con0, con, cpar.num_cams, match0, tim)
        match_counts[3] += match_counts[0]

    # search consistent triplets: 123, 124, 134, 234
    if (cpar.num_cams == 4 and cpar.allCam_flag == 0) or cpar.num_cams == 3:
        match0 = three_camera_matching(
            list, cpar.num_cams, frm.num_targets, vpar.corrmin, con0, 4 * nmax, tim
        )

        match_counts[1] = take_best_candidates(
            con0, con[match_counts[3] :], cpar.num_cams, match0, tim
        )
        match_counts[3] += match_counts[1]

    # Search consistent pairs: 12, 13, 14, 23, 24, 34
    if cpar.num_cams > 1 and cpar.allCam_flag == 0:
        match0 = consistent_pair_matching(
            list, cpar.num_cams, frm.num_targets, vpar.corrmin, con0, 4 * nmax, tim
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
    deallocate_adjacency_lists(list, cpar.num_cams)
    deallocate_target_usage_marks(tim, cpar.num_cams)
    del con0

    return con


class MatchedCoords:
    """Keep a block of 2D flat coordinates, each with a point number.

    the same as the number on one ``target`` from the block to which this block is kept
    matched. This block is x-sorted.

    NB: the data is not meant to be directly manipulated at this point. The
    coord_2d arrays are most useful as intermediate objects created and
    manipulated only by other liboptv functions. Although one can imagine a
    use case for direct manipulation in Python, it is rare and supporting it
    is a low priority.
    """

    buf = List[Coord2d]
    _num_pts: int

    def __init__(
        self,
        targs: TargetArray,
        cpar: ControlPar,
        cal: Calibration,
        tol: float = 0.00001,
        reset_numbers=True,
    ):
        """
        Allocate and initialize the memory, including coordinate conversion.

        and sorting.

        Arguments:
        ---------
        TargetArray targs - the TargetArray to be converted and matched.
        ControlPar cpar - parameters of image size etc. for conversion.
        Calibration cal - representation of the camera parameters to use in
            the flat/distorted transforms.
        double tol - optional tolerance for the lens distortion correction
            phase, see ``optv.transforms``.
        reset_numbers - if True (default) numbers the targets too, in their
            current order. This shouldn't be necessary since all TargetArray
            creators number the targets, but this gets around cases where they
            don't.
        """
        # cdef:
        #     target *targ

        self._num_pts = len(targs)
        self.buf = [Coord2d] * self._num_pts

        for tnum in range(self._num_pts):
            targ = targs._tarr[tnum]
            if reset_numbers:
                targ.pnr = tnum

            self.buf[tnum].x, self.buf[tnum].y = pixel_to_metric(targ.x, targ.y, cpar)
            self.buf[tnum].x, self.buf[tnum].y = dist_to_flat(
                cal, self.buf[tnum].x, self.buf[tnum].y, tol
            )
            self.buf[tnum].pnr = targ.pnr

        quicksort_coord2d_x(self.buf, self._num_pts)

    def as_arrays(self):
        """
        Return the data associated with the object (the matched coordinates.

        block) as NumPy arrays.

        Returns
        -------
        pos - (n,2) array, the (x,y) flat-coordinates position of n targets.
        pnr - n-length array, the corresponding target number for each point.
        """
        pos = np.empty((self._num_pts, 2))
        pnr = np.empty(self._num_pts, dtype=np.int_)

        for pt in range(self._num_pts):
            pos[pt, 0] = self.buf[pt].x
            pos[pt, 1] = self.buf[pt].y
            pnr[pt] = self.buf[pt].pnr

        return pos, pnr

    def get_by_pnrs(self, pnrs: np.ndarray):
        """
        Return the flat positions of points whose pnr property is given, as an.

        (n,2) flat position array. Assumes all pnrs are to be found, otherwise
        there will be garbage at the end of the position array.
        """
        pos = np.full((len(pnrs), 2), COORD_UNUSED, dtype=np.float64)
        for pt in range(self._num_pts):
            which = np.flatnonzero(self.buf[pt].pnr == pnrs)
            if len(which) > 0:
                which = which[0]
                pos[which, 0] = self.buf[pt].x
                pos[which, 1] = self.buf[pt].y
        return pos


def py_correspondences(
    img_pts: List[float],
    flat_coords: List[float],
    cals: List[Calibration],
    vparam: VolumePar,
    cparam: ControlPar,
):
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
    ControlPar cparam - an object holding general control parameters.

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
    num_cams = len(cals)

    # Special case of a single camera, follow the single_cam_correspondence docstring
    if num_cams == 1:
        sorted_pos, sorted_corresp, num_targs = single_cam_correspondence(
            img_pts, flat_coords
        )
        return sorted_pos, sorted_corresp, num_targs

    calib = [Calibration()] * num_cams
    corrected = [Coord2d()] * num_cams

    # np.ndarray[ndim=2, dtype=np.int_t] clique_ids
    # np.ndarray[ndim=3, dtype=np.float64_t] clique_targs

    # Return buffers:
    # int *match_counts = <int *> malloc(num_cams * sizeof(int))
    match_counts = np.zeros(num_cams, dtype=int)

    # n_tupel *corresp_buf

    # Initialize frame partially, without the extra momory used by init_frame.
    # .targets = <target**> calloc(num_cams, sizeof(target*))
    # frm.num_targets = <int *> calloc(num_cams, sizeof(int))

    frm = Frame(num_cams=num_cams)

    for cam in range(num_cams):
        calib[cam] = cals[cam]
        frm.targets[cam] = img_pts[cam]._tarr
        frm.num_targets[cam] = len(img_pts[cam])
        corrected[cam] = flat_coords[cam].buf

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
                    clique_targs[cam, pt, 0] = targ._targ.x
                    clique_targs[cam, pt, 1] = targ._targ.y

        last_count += num_points
        sorted_pos[clique_type] = clique_targs
        sorted_corresp[clique_type] = clique_ids

    # Clean up.
    num_targs = match_counts[num_cams - 1]
    # free(frm.targets)
    # free(frm.num_targets)
    # free(calib)
    # free(match_counts)
    # free(corresp_buf) # Note this for future returning of correspondences.

    return sorted_pos, sorted_corresp, num_targs


def single_cam_correspondence(img_pts: List[float], flat_coords: List[float]):
    """
    Single camera correspondence is not a real correspondence, it will be only a projection.

    of a 2D target from the image space into the 3D position, x,y,z using epi_mm_2d
    function. Here we only update the pointers of the targets and return it in a proper format.

    Arguments:
    ---------
    img_pts - a list of c := len(cals), containing TargetArray objects, each
        with the target coordinates of n detections in the respective image.
        The target arrays are clobbered: returned arrays have the tnr property
        set. the pnr property should be set to the target index in its array.
    flat_coords - a list of MatchedCoordinates objects, one per camera, holding
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

    sorted_pos = [None]
    sorted_corresp = [None]

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
            clique_targs[0, pt, 0] = targ._targ.x
            clique_targs[0, pt, 1] = targ._targ.x
            # we also update the tnr, see docstring of correspondences
            targ._targ.tnr = pt

    sorted_pos[0] = clique_targs
    sorted_corresp[0] = clique_ids

    return sorted_pos, sorted_corresp, num_points
