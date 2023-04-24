"""Tracking frame buffer."""
from collections import deque
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np

from .calibration import Calibration
from .constants import (
    COORD_UNUSED,
    MAX_TARGETS,
    MAXCAND,
    NEXT_NONE,
    POSI,
    PREV_NONE,
    PRIO_DEFAULT,
    PT_UNUSED,
)
from .epi import Coord2d
from .parameters import ControlPar
from .trafo import dist_to_flat, pixel_to_metric


@dataclass
class n_tupel:
    """n_tupel data structure."""

    p: List[int]
    corr: float


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


def sort_crd_y(crd):
    """Sort coordinates by y."""
    return sorted(crd, key=lambda p: p.y)


# def quicksort_target_y(pix, num):
#     """Quicksort for targets."""
#     pix = qs_target_y(pix, 0, num - 1)
#     return pix


# def qs_target_y(pix, left, right):
#     """Quicksort for targets subroutine."""
#     if left >= right:
#         return

#     pivot = pix[(left + right) // 2].y
#     i, j = left, right
#     while i <= j:
#         while pix[i].y < pivot:
#             i += 1
#         while pix[j].y > pivot:
#             j -= 1
#         if i <= j:
#             pix[i], pix[j] = pix[j], pix[i]
#             i += 1
#             j -= 1

#     pix = qs_target_y(pix, left, j)
#     pix = qs_target_y(pix, i, right)
#     return pix


# def quicksort_coord2d_x(crd, num):
#     """Quicksort for coordinates."""
#     crd = qs_coord2d_x(crd, 0, num - 1)
#     return crd


# def qs_coord2d_x(crd, left, right):
#     """Quicksort for coordinates subroutine."""
#     if left >= right:
#         return

#     pivot = crd[(left + right) // 2].x
#     i, j = left, right
#     while i <= j:
#         while crd[i].x < pivot and i < right:
#             i += 1
#         while pivot < crd[j].x and j > left:
#             j -= 1
#         if i <= j:
#             crd[i], crd[j] = crd[j], crd[i]
#             i += 1
#             j -= 1

#     crd = qs_coord2d_x(crd, left, j)
#     crd = qs_coord2d_x(crd, i, right)
#     return crd


def sort_crd_x(crd):
    """Sort coordinates by x."""
    return sorted(crd, key=lambda p: p.x)


@dataclass
class Target:
    """Target structure for tracking."""

    pnr: int = PT_UNUSED  # target number
    x: float = field(default_factory=float)  # pixel position
    y: float = field(default_factory=float)  # pixel position
    n: int = field(default_factory=int)  # number of pixels
    nx: int = field(default_factory=int)  # in x
    ny: int = field(default_factory=int)  # in y
    sumg: int = field(default_factory=int)  # sum of grey values
    tnr: int = field(default_factory=int)  # used in tracking

    def __eq__(self, other) -> bool:
        """Compare two targets."""
        return (
            self.pnr == other.pnr
            and self.x == other.x
            and self.y == other.y
            and self.n == other.n
            and self.nx == other.nx
            and self.ny == other.ny
            and self.sumg == other.sumg
            and self.tnr == other.tnr
        )

    def set_pos(self, x, y):
        """Set target position."""
        self.x = x
        self.y = y

    def set_pnr(self, pnr):
        """Set target number."""
        self.pnr = pnr

    def set_pixel_counts(self, n, nx, ny):
        """Set number of pixels and number of pixels in x and y."""
        self.n = n
        self.nx = nx
        self.ny = ny

    def set_sum_grey_value(self, sumg):
        """Set sum of grey values."""
        self.sumg = sumg


@dataclass
class TargetArray:
    """A list of targets and the number of targets in the list."""

    num_targs: int = field(default_factory=int)
    targs: List[Target] = field(
        default_factory=lambda: [Target() for _ in range(MAX_TARGETS)]
    )

    def __init__(self, num_targs: int):
        self.num_targs = num_targs
        self.targs = [Target() for _ in range(num_targs)]

    def set(self, targs: List[Target]):
        """Set targets."""
        self.targs = targs
        self.num_targs = len(targs)


def read_targets(file_base: str, frame_num: int) -> List[Target]:
    """Read targets from a file."""
    buffer = []

    if frame_num > 0:
        filename = f"{file_base}{frame_num:04}_targets"
    else:
        filename = f"{file_base}_targets"

    try:
        with open(filename, "r", encoding="utf-8") as file:
            num_targets = int(file.readline().strip())

            for _ in range(num_targets):
                line = file.readline().strip().split()

                if len(line) != 8:
                    raise ValueError(f"Bad format for file: {filename}")

                targ = Target(
                    pnr=int(line[0]),
                    x=float(line[1]),
                    y=float(line[2]),
                    n=int(line[3]),
                    nx=int(line[4]),
                    ny=int(line[5]),
                    sumg=int(line[6]),
                    tnr=int(line[7]),
                )

                buffer.append(targ)

    except IOError as err:
        print(f"Can't open ascii file: {filename}")
        raise err

    return buffer


def write_targets(
    targets: List[Target], num_targets: int, file_base: str, frame_num: int
) -> bool:
    """Write targets to a file."""
    success = False
    file_name = (
        file_base + "_targets"
        if frame_num == 0
        else f"{file_base}{frame_num:04d}_targets"
    )

    try:
        # Convert targets to a 2D numpy array
        target_arr = np.array(
            [(t.pnr, t.x, t.y, t.n, t.nx, t.ny, t.sumg, t.tnr) for t in targets]
        )
        # Save the target array to file using savetxt
        np.savetxt(
            file_name,
            target_arr,
            fmt="%4d %9.4f %9.4f %5d %5d %5d %5d %5d",
            header=f"{num_targets}",
            comments="",
        )
        success = True
    except IOError:
        print(f"Can't open ascii file: {file_name}")

    return success


@dataclass
class Pathinfo:
    """Pathinfo structure for tracking."""

    x: np.ndarray = np.zeros(3, dtype=np.float64)
    prev: int = PREV_NONE
    next: int = NEXT_NONE
    prio: int = PRIO_DEFAULT
    decis: List[float] = field(default_factory=lambda: [0.0] * POSI)
    finaldecis: float = 0.0
    linkdecis: List[int] = field(default_factory=lambda: [0.0] * POSI)
    inlist: int = 0

    def __eq__(self, other):
        if not isinstance(other, Pathinfo):
            return False
        return (
            self.x == other.x
            and self.prev == other.prev
            and self.next == other.next
            and self.prio == other.prio
            and self.decis == other.decis
            and self.finaldecis == other.finaldecis
            and self.linkdecis == other.linkdecis
            and self.inlist == other.inlist
        )

    def register_link_candidate(self, fitness: float, cand: int) -> None:
        """Register link candidate."""
        self.decis[self.inlist] = fitness
        self.linkdecis[self.inlist] = cand
        self.inlist += 1

    def reset_links(self) -> None:
        """Reset links."""
        self.prev = PREV_NONE
        self.next = NEXT_NONE
        self.prio = PRIO_DEFAULT


@dataclass
class Frame:
    """Frame structure for tracking."""

    num_cams: int
    max_targets: int = MAX_TARGETS
    path_info: Pathinfo | None = None
    correspond: Correspond | None = None
    targets: List[List[Target]] | None = None
    num_parts: int = 0
    num_targets: List[int] | None = None

    def read_frame(
        self,
        corres_file_base: Any,
        linkage_file_base: Any,
        prio_file_base: Any,
        target_file_base: List[Any],
        frame_num: int,
    ) -> bool:
        """Read a frame from the disk."""
        self.num_parts = read_path_frame(
            self.correspond,
            self.path_info,
            corres_file_base,
            linkage_file_base,
            prio_file_base,
            frame_num,
        )

        if self.num_parts == -1:
            return False

        if self.num_targets == [0] * self.num_cams:
            return False

        for cam in range(self.num_cams):
            self.targets[cam] = read_targets(target_file_base[cam], frame_num)
            self.num_targets[cam] = len(self.targets[cam])

            if self.num_targets[cam] == -1:
                return False

        return True

    def write_frame(
        self,
        corres_file_base: Any,
        linkage_file_base: Any,
        prio_file_base: Any,
        target_file_base: List[Any],
        frame_num: int,
    ) -> bool:
        """Write a frame to the disk."""
        status = write_path_frame(
            self.correspond,
            self.path_info,
            self.num_parts,
            corres_file_base,
            linkage_file_base,
            prio_file_base,
            frame_num,
        )

        if status == 0:
            return False

        for cam in range(self.num_cams):
            status = write_targets(
                self.targets[cam],
                self.num_targets[cam],
                target_file_base[cam],
                frame_num,
            )

            if status == 0:
                return False

        return True


class FrameBufBase:
    def __init__(self, buf_len, num_cams, max_targets):
        self.buf_len = buf_len
        self.num_cams = num_cams
        self.buf = deque(maxlen=buf_len)
        for _ in range(buf_len):
            self.buf.append(Frame(num_cams, max_targets))
        self.start = 0
        self.frame_num = 0

    def __del__(self):
        for frame in self.buf:
            del frame

    def read_frame_at_end(self, frame):
        self.buf.append(frame)

    def write_frame_from_start(self):
        pass
        # pickled_frame = pickle.dumps(frame)
        # self._advance_buffer_start()
        # return pickled_frame

    def _advance_buffer_start(self):
        self.start += 1
        if self.start >= self.buf_len:
            self.start = 0

    def fb_next(self):
        self.buf.rotate(-1)
        self._advance_buffer_start()

    def fb_prev(self):
        self.buf.rotate(1)
        self.start -= 1
        if self.start < 0:
            self.start = self.buf_len - 1


# def free_frame(frame):
#     # Free memory for the frame
#     frame = None


# class Frame:
#     def __init__(self, num_cams, max_targets):
#         self.num_cams = num_cams
#         self.max_targets = max_targets


# def frame_init(frame, num_cams, max_targets):
#     # Initialize the frame
#     frame.num_cams = num_cams
#     frame.max_targets = max_targets
#     return frame


@dataclass
class FrameBuf(FrameBufBase):
    buf_len: int
    num_cams: int
    max_targets: int
    corres_file_base: str = None
    linkage_file_base: str = None
    prio_file_base: str = None
    target_file_base: str = None

    def write_frame_from_start(self):
        """Write a frame to disk and advance the buffer."""
        # Write the frame to disk
        write_path_frame(
            self.buf[0],
            self.corres_file_base,
            len(self.buf),
            self.linkage_file_base,
            self.prio_file_base,
            self.target_file_base,
            self.frame_num,
        )

        # Advance the buffer
        self.buf.appendleft(Frame(self.num_cams, self.max_targets))

    def fb_disk_read_frame_at_end(self, read_links):
        if read_links:
            return read_path_frame(
                self.buf[-1],
                self.corres_file_base,
                self.linkage_file_base,
                self.prio_file_base,
                self.target_file_base,
                self.frame_num,
            )
        else:
            return read_path_frame(
                self.buf[-1],
                self.corres_file_base,
                None,
                None,
                self.target_file_base,
                self.frame_num,
            )


def frame_init(num_cams: int, max_targets: int):
    """Initialize a frame structure."""
    new_frame = Frame(max_targets=max_targets, num_cams=num_cams)
    for cam in range(num_cams):
        new_frame.targets[cam] = [Target() for _ in range(max_targets)]
        new_frame.num_targets[cam] = 0

    return new_frame


def read_path_frame(
    cor_buf, path_buf, corres_file_base, linkage_file_base, prio_file_base, frame_num
) -> List[Target]:
    """Read a frame from the disk.

    cor_buf = array of correspondences, pnr, 4 x cam_pnr

    """
    filein, linkagein, prioin = None, None, None
    fname = f"{corres_file_base}.{frame_num}"
    try:
        filein = open(fname, "r", encoding="utf-8")
    except IOError:
        print(f"Can't open ascii file: {fname}")
        return -1

    filein.readline()

    if linkage_file_base is not None:
        fname = f"{linkage_file_base}.{frame_num}"
        try:
            linkagein = open(fname, "r", encoding="utf-8")
        except IOError:
            print(f"Can't open linkage file: {fname}")
            return -1

        linkagein.readline()

    if prio_file_base is not None:
        fname = f"{prio_file_base}.{frame_num}"
        try:
            prioin = open(fname, "r", encoding="utf-8")
        except IOError:
            print(f"Can't open prio file: {fname}")
            return -1

        prioin.readline()

    targets = 0
    while True:
        line = filein.readline()
        if not line:
            break

        if linkagein is not None:
            linkage_line = linkagein.readline()
            linkage_vals = np.fromstring(linkage_line, dtype=float, sep=" ")
            path_buf["prev"] = linkage_vals[0].astype(int)
            path_buf["next"] = linkage_vals[1].astype(int)

        if prioin is not None:
            prio_line = prioin.readline()
            prio_vals = np.fromstring(prio_line, dtype=float, sep=" ")
            path_buf["prio"] = prio_vals[-1].astype(int)
        else:
            path_buf["prio"] = 4

        path_buf["inlist"] = 0
        path_buf["finaldecis"] = 1000000.0
        path_buf["decis"] = np.zeros(POSI)
        path_buf["linkdecis"] = np.zeros(POSI) - 999

        vals = np.fromstring(line, dtype=float, sep=" ")
        cor_buf["nr"] = targets + 1
        cor_buf["Pathinfo"] = vals[-4:].astype(int)
        path_buf["x"] = vals[:-4]

        cor_buf += 1
        path_buf += 1

        targets += 1

    filein.close()
    if linkagein is not None:
        linkagein.close()
    if prioin is not None:
        prioin.close()

    return targets


def write_path_frame(
    cor_buf: List[Target],
    path_buf: List[Pathinfo],
    num_parts: int,
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    frame_num: int,
) -> bool:
    """
    Write a frame of path and correspondence info.

    The correspondence and linkage information for a frame with the next and previous
    frames is written. The information is distributed among two files. The rt_is file holds
    correspondence and position data, and the ptv_is holds linkage data.

    Args:
    ----
        cor_buf: List of corres structs to write to the files.
        path_buf: List of path info structures.
        num_parts: Number of particles represented by the cor_buf and path_buf arrays.
        corres_file_base: Base name of the output correspondence file.
        linkage_file_base: Base name of the output linkage file.
        prio_file_base: Base name of the output priority file (optional).
        frame_num: Number of the frame to add to file_base.

    Returns:
    -------
        True on success, False on failure.
    """
    corres_fname = f"{corres_file_base}.{frame_num}"
    linkage_fname = f"{linkage_file_base}.{frame_num}"
    prio_fname = f"{prio_file_base}.{frame_num}" if prio_file_base else None

    try:
        np.savetxt(
            corres_fname,
            [
                [
                    pix + 1,
                    path_buf[pix].x[0],
                    path_buf[pix].x[1],
                    path_buf[pix].x[2],
                    cor_buf[pix].p[0],
                    cor_buf[pix].p[1],
                    cor_buf[pix].p[2],
                    cor_buf[pix].p[3],
                ]
                for pix in range(num_parts)
            ],
            fmt="%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d",
        )
        np.savetxt(
            linkage_fname,
            [
                [
                    path_buf[pix].prev,
                    path_buf[pix].next,
                    path_buf[pix].x[0],
                    path_buf[pix].x[1],
                    path_buf[pix].x[2],
                ]
                for pix in range(num_parts)
            ],
            fmt="%4d %4d %10.3f %10.3f %10.3f",
        )
        if prio_fname:
            np.savetxt(
                prio_fname,
                [
                    [
                        path_buf[pix].prev,
                        path_buf[pix].next,
                        path_buf[pix].x[0],
                        path_buf[pix].x[1],
                        path_buf[pix].x[2],
                        path_buf[pix].prio,
                    ]
                    for pix in range(num_parts)
                ],
                fmt="%4d %4d %10.3f %10.3f %10.3f %f",
            )
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

    return True


# cdef extern from "optv/tracking_frame_buf.h":
#     ctypedef struct target:
#         int pnr
#         double x, y
#         int n, nx, ny, sumg
#         int tnr

# ctypedef struct corres:
#     int nr
#     int p[4]

# cpdef enum:
#     CORRES_NONE = -1
#     PT_UNUSED = -999

# ctypedef struct path_inf "P":
#     vec3d x
#     int prev, next, prio

# ctypedef struct frame:
#     path_inf *path_info
#     corres *correspond
#     target **targets
#     int num_cams, max_targets, num_parts
#     int *num_targets

# ctypedef struct framebuf:
#     pass

# void fb_free(framebuf *self)


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

    buf: List[Coord2d]
    _num_pts: int

    def __init__(
        self,
        targs: List[Target],
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
        self.buf = [Coord2d() for _ in range(self._num_pts)]

        for tnum in range(self._num_pts):
            targ = targs[tnum]
            if reset_numbers:
                targ.pnr = tnum

            x, y = pixel_to_metric(targ.x, targ.y, cpar)
            self.buf[tnum].x, self.buf[tnum].y = dist_to_flat(x, y, cal, tol)
            self.buf[tnum].pnr = targ.pnr

        # self.buf = quicksort_coord2d_x(self.buf, self._num_pts)
        self.buf = sort_crd_x(self.buf)

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
