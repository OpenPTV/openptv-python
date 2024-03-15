"""Tracking frame buffer."""
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np

from .calibration import Calibration
from .constants import (
    COORD_UNUSED,
    CORRES_NONE,
    MAX_TARGETS,
    NEXT_NONE,
    POSI,
    PREV_NONE,
    PRIO_DEFAULT,
    # PT_UNUSED,
    TR_UNUSED,
)
from .epi import Coord2d_dtype
from .parameters import ControlPar
from .trafo import dist_to_flat, pixel_to_metric

n_tupel_dtype = np.dtype([
    ("p", np.int32, (4,)),
    ("corr", np.float64),
])

n_tupel = np.zeros(1, dtype=n_tupel_dtype)

def quicksort_n_tupel(arr: np.ndarray) -> np.ndarray:
    """
    Quicksorts a list of n_tupel instances based on the corr attribute.

    previously called quicksort_con

    Args:
    ----
      n_tupel_list: Array of n_tupel instances.

    Returns
    -------
      A list of n_tupel instances, sorted by the corr attribute.
    """
    # inline sorting
    return np.sort(arr, order = 'corr')




Corres_dtype = np.dtype([
    ("nr", np.int32),
    ("p", np.int32, (4,)) # -1 for no correspondence
])

Corres = np.array([(0, [-1, -1, -1, -1])], dtype=Corres_dtype)

    # def __eq__(self, other):
    #     return self['nr'] == other['nr'] and np.all(self['p'] == other['p'])


def compare_corres(c1: np.ndarray, c2: np.ndarray) -> bool:
    """
    Compare two Corres instances.

    Args:
    ----
        c1: A Corres array
        c2: A Corres array.

    Returns
    -------
        True if the Corres instances are equal, False otherwise.
    """
    return (c1 == c2).all()



Target_dtype = np.dtype([
    ("pnr", np.int32), # CORRES_NONE
    ("x", np.float64),
    ("y", np.float64),
    ("n", np.int32),
    ("nx", np.int32),
    ("ny", np.int32),
    ("sumg", np.int32),
    ("tnr", np.int32) # TR_UNUSED
    ])

Target = np.array([(CORRES_NONE, 0., 0., 0, 0, 0, 0, TR_UNUSED)], dtype=Target_dtype)

# @dataclass
# class Target:
#     """Target structure for tracking."""

#     pnr: int = PT_UNUSED  # target number
#     x: float = 0.0  # pixel position
#     y: float = 0.0  # pixel position
#     n: int = 0  # number of pixels
#     nx: int = 0  # in x
#     ny: int = 0  # in y
#     sumg: int = 0  # sum of grey values
#     tnr: int = -1  # used in tracking

# def set_pos(target: np.ndarray, pos: Tuple[float, float]) -> np.ndarray:
#     """Set target position."""
#     target['x'], target['y'] = pos

# def set_pnr(target: np.ndarray, pnr: int) -> np.ndarray:
#     """Set target number."""
#     target['pnr'] = pnr

# def set_pixel_counts(self, n, nx, ny):
#     """Set number of pixels and number of pixels in x and y."""
#     self.n = n
#     self.nx = nx
#     self.ny = ny

# def set_sum_grey_value(self, sumg):
#     """Set sum of grey values."""
#     self.sumg = sumg

# def sum_grey_value(self):
#     """Return sum of grey values."""
#     return self.sumg

# def pos(target: np.ndarray) -> Tuple[float, float]:
#     """Return target position."""
#     # return Coord2d(self.x, self.y)
#     return target['x'], target['y']

# def count_pixels(self):
#     """Return number of pixels."""
#     return (self.n, self.nx, self.ny)

#     def __eq__(self, __value) -> bool:
#         return (
#             self['pnr'] == __value['pnr']  # type: ignore
#             and self.x == __value.x  # type: ignore
#             and self.y == __value.y  # type: ignore
#             and self.n == __value.n  # type: ignore
#             and self.nx == __value.nx  # type: ignore
#             and self.ny == __value.ny  # type: ignore
#             and self.sumg == __value.sumg  # type: ignore
#             and self.tnr == __value.tnr  # type: ignore
#         )


def sort_target_y(targets: np.ndarray) -> np.ndarray:
    """Sort targets by y coordinate."""
    return np.sort(targets, order='y')

# class TargetArray(list):
#     """Target array class."""

#     def __init__(self, num_targets: int = 0):
#         super().__init__(Target() for item in range(num_targets))

#     def __setitem__(self, index, item):
#         super().__setitem__(index, item)

#     def insert(self, index, item):
#         super().insert(index, item)

#     def append(self, item):
#         super().append(str(item))

#     def extend(self, other):
#         if isinstance(other, type(self)):
#             super().extend(other)
#         else:
#             super().extend(item for item in other)

#     @property
#     def num_targs(self):
#         """Return the number of targets in the list."""
#         return len(self)


def read_targets(file_base: str, frame_num: int) -> np.ndarray:
    """Read targets from a file."""
    # # if file_base has an extension, remove it
    # file_base = file_base.split(".")[0]

    if frame_num > 0:
        # filename = f"{file_base}{frame_num:04d}_targets"
        fname = file_base % frame_num + '_targets'
    else:
        fname = f"{file_base}_targets"

    filename = Path(fname)
    print(f" filename: {filename}")
    if not filename.exists():
        raise FileNotFoundError(f"Can't open ascii file: {filename}")

    buffer = np.loadtxt(filename, dtype=Target_dtype, skiprows=1)
    return buffer


def write_targets(
    targets: np.ndarray, num_targets: int, file_base: str, frame_num: int) -> bool:
    """Write targets to a file."""
    success = False

    # fix old-type names, that are like cam1.# or just cam1.
    if '#' in file_base:
        file_base = file_base.replace('#', '%05d')
    if "%" not in file_base:
        file_base = file_base + "%05d"

    if frame_num == 0:
        frame_num = 123456789

    file_name =  file_base % frame_num + "_targets"

    try:
        # Convert targets to a 2D numpy array
        # target_arr = np.array(
        #     [(t['pnr'], t['x'], t['y'], t['n'], t['nx'], t['ny'], t['sumg'], t.tnr) for t in targets]
        # )
        # target_arr = np.array(targets.tolist())
        # Save the target array to file using savetxt
        np.savetxt(
            file_name,
            targets,
            fmt="%4d %9.4f %9.4f %5d %5d %5d %5d %5d",
            header=f"{num_targets}",
            comments="",
        )
        success = True
    except IOError:
        print(f"Can't open ascii file: {file_name}")

    return success


# def compare_targets(t1: Target, t2: Target):
#     """Check that target t1 is equal to target t2, i.e., all their fields are equal.

#     Arguments:
#     ---------
#     t1, t2 (tuple): the target structures to compare.

#     Returns
#     -------
#     bool: True if all fields are equal, False otherwise.
#     """
#     return t1 == t2

Pathinfo_dtype = np.dtype([
        ('x', np.float64, 3),
        ('prev_frame', np.int32),
        ('next_frame', np.int32),
        ('prio', np.int32),
        ('decis', np.float64, POSI),
        ('finaldecis', np.float64),
        ('linkdecis', np.int32, POSI),
        ('inlist', np.int32)
    ])
Pathinfo = np.array([(np.zeros(3), PREV_NONE, NEXT_NONE, PRIO_DEFAULT,
                         [0.0] * POSI, 1000000.0, [0] * POSI, 0)], dtype=Pathinfo_dtype)

# @dataclass
# class Pathinfo:
#     """Pathinfo structure for tracking."""

#     x: np.ndarray = field(default_factory=lambda: np.zeros(3))
#     prev_frame: int = PREV_NONE
#     next_frame: int = NEXT_NONE
#     prio: int = PRIO_DEFAULT
#     decis: List[float] = field(default_factory=lambda: [0.0] * POSI)
#     finaldecis: float = 0.0
#     linkdecis: List[int] = field(default_factory=lambda: [0] * POSI)
#     inlist: int = 0

#     def __eq__(self, other):
#         if not isinstance(other, Pathinfo):
#             return False
#         return (
#             (self.x == other.x).all()
#             and self['prev_frame'] == other['prev_frame']
#             and self['next_frame'] == other['next_frame']
#             and self['prio'] == other['prio']
#             and (self['decis'] == other['decis'])
#             and self['finaldecis'] == other['finaldecis']
#             and (self.linkdecis == other.linkdecis)
#             and self['inlist'] == other['inlist']
#         )

def register_link_candidate(pathinfo: np.ndarray, fitness: float, cand: int) -> np.ndarray:
    """Register link candidate."""
    pathinfo['decis'][pathinfo['inlist']] = fitness
    pathinfo['linkdecis'][pathinfo['inlist']] = cand
    pathinfo['inlist'] += 1
    return pathinfo

def reset_links(pathinfo: np.ndarray) -> np.ndarray:
    """Reset links."""
    pathinfo['prev_frame'] = PREV_NONE
    pathinfo['next_frame'] = NEXT_NONE
    pathinfo['prio'] = PRIO_DEFAULT
    return pathinfo


def compare_path_info(path_info1: np.ndarray, path_info2: np.ndarray) -> bool:
    """Compare path info."""
    return path_info1 == path_info2


class Frame:
    """Frame structure for tracking."""

    def __init__(self, num_cams: int, max_targets: int = MAX_TARGETS):
        """
        Initialize a frame object, allocates its arrays and sets up the frame data.

        Arguments:
        ---------
        num_cams - number of cameras per frame.
        max_targets - number of elements to allocate for the different buffers
            held by a frame.
        """
        self.path_info = np.tile(Pathinfo, max_targets)
        self.correspond = np.tile(Corres, max_targets)
        self.correspond['p'] = TR_UNUSED
        self.correspond['nr'] = 0

        self.targets = np.tile(Target, (num_cams, max_targets))
        self.num_targets = [0] * num_cams

        self.num_cams = num_cams
        self.max_targets = max_targets
        self.num_parts = 0

    def read(
        self,
        corres_file_base: str,
        linkage_file_base: str,
        prio_file_base: str,
        target_file_base: List[str],
        frame_num: int,
    ) -> bool:
        """Read a frame from the disk."""
        cor_buf, path_buf = read_path_frame(
            # self.correspond, self.path_info = read_path_frame(
            corres_file_base,
            linkage_file_base,
            prio_file_base,
            frame_num,
        )

        self.correspond = cor_buf
        self.path_info = path_buf
        self.num_parts = len(self.correspond)

        if self.num_parts == -1:
            return False

        for cam in range(self.num_cams):
            tmp = read_targets( target_file_base[cam], frame_num )
            self.targets[cam][:len(tmp)] = tmp
            self.num_targets[cam] = len(self.targets[cam])

            if self.num_targets[cam] == -1:
                return False

        return True

    def write(
        self,
        corres_file_base: str,
        linkage_file_base: str,
        prio_file_base: str,
        target_file_base: List[str],
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

    def positions(self) -> np.ndarray:
        """Return an (n,3) array 3D positions on n particles in the frame."""
        pos3d = np.zeros((self.num_parts, 3))
        for pt in range(self.num_parts):
            pos3d[pt] = self.path_info[pt].x

        return pos3d

    def target_positions_for_camera(self, cam: int) -> np.ndarray:
        """
        Get all targets in this frame as seen by the selected camere. The.

        targets are returned in the order corresponding to the particle order
        returned by ``positions()``.

        Arguments:
        ---------
        int cam - camera number, starting from 0.

        Returns
        -------
        an (n,2) array with the 2D position of targets detected in the image
            seen by camera ``cam``. for each 3D position. If no target in this
            camera belongs to the 3D position, its target is set to NaN.
        """
        pos2d = np.empty((self.num_parts, 2))
        for pt in range(self.num_parts):
            tix = self.correspond[pt]['p'][cam]

            if tix == CORRES_NONE:
                pos2d[pt] = np.nan
            else:
                pos2d[pt, 0] = self.targets[cam][tix].x
                pos2d[pt, 1] = self.targets[cam][tix].y

        return pos2d

    def __repr__(self):
        return f"<Frame num_parts={self.num_parts} num_cams={self.num_cams} max_targets={self.max_targets}>"

    # def frame_init(num_cams: int, max_targets: int):
    # """Initialize a frame structure."""
    # frame = Frame(max_targets=max_targets, num_cams=num_cams)
    # for cam in range(num_cams):
    #     frame.targets[cam] = [Target() for _ in range(max_targets)]
    #     frame.num_targets[cam] = 0

    # return frame


# class RingVector:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.data = [None] * capacity
#         self.size = 0
#         self.head = 0

#     def push(self, item):
#         if self.size < self.capacity:
#             self.data[(self.head + self.size) % self.capacity] = item
#             self.size += 1
#         else:
#             # Handle the case where the ring is full (optional)
#             print("Ring is full, cannot push item.")

#     def pop(self):
#         if self.size > 0:
#             item = self.data[self.head]
#             self.head = (self.head + 1) % self.capacity
#             self.size -= 1
#             return item
#         else:
#             # Handle the case where the ring is empty (optional)
#             print("Ring is empty, cannot pop item.")

# # Example usage:
# ring = RingVector(5)
# ring.push(1)
# ring.push(2)
# ring.push(3)
# ring.push(4)
# ring.push(5)

# print("Initial Ring:", ring.data)

# ring.push(6)  # This will print "Ring is full, cannot push item."

# popped_item = ring.pop()
# print("Popped item:", popped_item)
# print("Updated Ring:", ring.data)


class FrameBufBase:
    """Frame buffer class."""

    def __init__(
        self,
        buf: Deque[Frame],
        # _ring_vec: Deque[Frame],
        buf_len: int = 4,
        num_cams: int = 1,
    ):
        self.buf: Deque[Frame] = buf
        # self._ring_vec: Deque[Frame] = _ring_vec
        self.buf_len = buf_len
        self.num_cams = num_cams
        self.size = 0
        self.head = 0

    def push(self, item: Frame):
        """Push an item onto the buffer."""
        if self.size < self.buf_len:
            self.buf[(self.head + self.size) % self.buf_len] = item
            self.size += 1
        else:
            # Handle the case where the ring is full (optional)
            print("Ring is full, cannot push item.")

    def pop(self):
        """Pop an item from the buffer."""
        if self.size > 0:
            item = self.buf[self.head]
            self.head = (self.head + 1) % self.buf_len
            self.size -= 1
            return item
        else:
            # Handle the case where the ring is empty (optional)
            print("Ring is empty, cannot pop item.")

    def fb_next(self):
        """Advances the start pointer of the frame buffer and.

        resetting it to the beginning after exceeding the buffer length.

        Arguments:
        ---------
        self - the framebuf to advance.
        """
        # self.head += 1

        # if self.head >= self.buf_len: # reset, but this is not necessary for the deque
        #     self.buf = self._ring_vec

        # seems that this is just deque.rotate(1)
        self.buf.rotate(-1)  # [0,1,2,3] -> [1,2,3,0] -> will be [1,2,3,4]

    def fb_prev(self):
        """Back the start pointer of the frame buffer and.

        setting it to the end after exceeding the buffer start.

        Arguments:
        ---------
        self - the framebuf to advance.
        """
        # self.buf -= 1
        # if self.buf < self._ring_vec:
        #     self.buf = self._ring_vec + self.buf_len - 1

        self.buf.rotate(1)  # [0,1,2,3] -> [3, 0, 1, 2] -> [-1, 0, 1, 2]


class FrameBuf(FrameBufBase):
    """Frame buffer class."""

    def __init__(
        self,
        buf_len: int,
        num_cams: int,
        max_targets: int,
        corres_file_base: str,
        linkage_file_base: str,
        prio_file_base: str,
        target_file_base: List[str],
    ):
        super().__init__(
            buf=deque(
                [Frame(num_cams, max_targets) for _ in range(buf_len)], maxlen=buf_len
            ),
            # _ring_vec=deque(
            #     [Frame(num_cams, max_targets) for _ in range(buf_len)], maxlen=buf_len
            # ),
            buf_len=buf_len,
            num_cams=num_cams,
        )

        self.size = buf_len
        self.head = 0

        self.corres_file_base = corres_file_base
        self.linkage_file_base = linkage_file_base
        self.prio_file_base = prio_file_base
        self.target_file_base = target_file_base

    # def write_frame_from_start(self, frame_num):
    #     """Write a frame to disk and advance the buffer."""
    #     # Write the frame to disk
    #     frame = self.buf[0]  # first frame
    #     cor_buf = frame.correspond
    #     path_buf = frame.path_info
    #     num_parts = frame.num_parts

    #     write_path_frame(
    #         cor_buf,
    #         path_buf,
    #         num_parts,
    #         self.corres_file_base,
    #         self.linkage_file_base,
    #         self.prio_file_base,
    #         frame_num,
    #     )

    #     # Advance the buffer
    #     # self.buf.appendleft(Frame(self.num_cams, MAX_TARGETS))

    def read_frame_at_end(self, frame_num: int, read_links: bool = False) -> None:
        """Read a frame from the disk and add it to the end of the buffer."""
        frame = self.buf[-1]  # last frame

        if read_links:
            success = frame.read(
                self.corres_file_base,
                self.linkage_file_base,
                self.prio_file_base,
                self.target_file_base,
                frame_num,
            )
        else:
            success = frame.read(
                self.corres_file_base, "", "", self.target_file_base, frame_num
            )

        if not success:
            raise IOError("Could not read frame from disk")

    def disk_read_frame_at_end(self, frame_num: int, read_links: bool):
        """Read a frame to the last position in the ring.

        Arguments:
        ---------
        self_base - the framebuf object doing the reading.
        frame_num - number of the frame to read in the sequence of frames.
        read_links - whether or not to read data in the linkage/prio files.

        Returns
        -------
        True on success, false on failure.
        """
        frame = self.buf[-1]  # always the last one on the right
        if read_links:
            return frame.read(
                self.corres_file_base,
                self.linkage_file_base,
                self.prio_file_base,
                self.target_file_base,
                frame_num,
            )
        else:
            return frame.read(
                self.corres_file_base,
                '',
                '',
                self.target_file_base,
                frame_num,
            )

    def write_frame_from_start(self, frame_num: int):
        """Write the frame to the first position in the ring.

        Arguments:
        ---------
        self_base - the framebuf object doing the reading.
        frame_num - number of the frame to write in the sequence of frames.

        Returns
        -------
        True on success, false on failure.
        """
        frame = self.buf[0]  # always the first one on the left

        return frame.write(
            self.corres_file_base,
            self.linkage_file_base,
            self.prio_file_base,
            self.target_file_base,
            frame_num,
        )


def read_path_frame(
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    frame_num: int,
) -> Tuple[np.ndarray, np.ndarray]: #List[Corres]
    """Read a rt_is frames from the disk.

        /* Reads rt_is files. these files contain both the path info and the
        * information on correspondence between targets on the different images.
        * Sets fields not in file to default values.
        *
        * Arguments:
        * corres *cor_buf - a buffer of corres structs to fill in from the file.
        * P *path_buf - same for path info structures.
        * char* corres_file_base, *linkage_file_base - base names of the output
        *   correspondence and likage files respectively, to which a frame number
        *   is added. Without separator.
        * char *prio_file_base - for the linkage file with added 'prio' column.
        * int frame_num - number of frame to add to file_base. A value of 0 or less
        *   means that no frame number should be added. The '.' separator is added
        * between the name and the frame number.
        *
        * Returns:
        * The number of points read for this frame. -1 on failure.
    */

    """
    fname = f"{corres_file_base}.{frame_num}"
    # print(fname)

    try:
        filein = open(fname, "r", encoding="utf-8")
    except IOError:
        print(f"Can't open ascii file: {fname}")
        return Corres, Pathinfo

    # we do not need number of particles, reading till EOF
    n_particles = int(filein.readline())
    cor_buf = np.tile(Corres, n_particles)
    path_buf = np.tile(Pathinfo, n_particles)

    if linkage_file_base != "":
        fname = f"{linkage_file_base}.{frame_num}"
        try:
            linkagein = open(fname, "r", encoding="utf-8")
        except IOError:
            print(f"Can't open linkage file: {fname}")
            return Corres, Pathinfo

        linkagein.readline()
    else:
        linkagein = None

    if prio_file_base != "":
        fname = f"{prio_file_base}.{frame_num}"
        try:
            prioin = open(fname, "r", encoding="utf-8")
        except IOError:
            print(f"Can't open prio file: {fname}")
            return Corres, Pathinfo

        prioin.readline()
    else:
        prioin = None

    targets = 0
    while True:
        line = filein.readline()
        if not line:
            break

        if linkagein is not None:
            linkage_line = linkagein.readline()
            linkage_vals = np.fromstring(linkage_line, dtype=float, sep=" ")
            path_buf[targets]['prev_frame'] = linkage_vals[0].astype(int)
            path_buf[targets]['next_frame'] = linkage_vals[1].astype(int)
            # path_buf[targets].x = linkage_vals[2:]

        if prioin is not None:
            prio_line = prioin.readline()
            prio_vals = np.fromstring(prio_line, dtype=float, sep=" ")
            path_buf[targets]['prio'] = prio_vals[-1].astype(int)
        else:
            path_buf[targets]['prio'] = 4

        path_buf[targets]['inlist'] = 0
        path_buf[targets]['finaldecis'] = 1000000.0
        path_buf[targets]['decis'] = [0] * POSI  # type: ignore
        path_buf[targets]['linkdecis'] = [-999] * POSI

        vals = np.fromstring(line, dtype=float, sep=" ")
        cor_buf[targets]['nr'] = targets + 1
        cor_buf[targets]['p'] = vals[-4:].astype(int)
        path_buf[targets]['x'] = vals[1:-4]

        # print(cor_buf[targets]['nr'], cor_buf[targets]['p'], path_buf[targets].x)

        targets += 1

    filein.close()
    if linkagein is not None:
        linkagein.close()
    if prioin is not None:
        prioin.close()

    return cor_buf, path_buf


def write_path_frame(
    cor_buf: np.ndarray, #List[Corres],
    path_buf: np.ndarray,
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

    Returns
    -------
        True on success, False on failure.
    """
    corres_fname = f"{corres_file_base}.{frame_num}"
    linkage_fname = f"{linkage_file_base}.{frame_num}"
    prio_fname = f"{prio_file_base}.{frame_num}" if prio_file_base != "" else None
    success = False

    try:
        corres_file = open(corres_fname, "w", encoding="utf8")
        corres_file.write(f"{num_parts}\n")

        linkage_file = open(linkage_fname, "w", encoding="utf8")
        linkage_file.write(f"{num_parts}\n")

        if prio_file_base is not None:
            prio_file = open(prio_fname, "w", encoding="utf8")  # type: ignore
            prio_file.write(f"{num_parts}\n")

        for pix in range(num_parts):
            linkage_file.write(
                f"{path_buf[pix]['prev_frame']} {path_buf[pix]['next_frame']} "
                f"{path_buf[pix]['x'][0]:.3f} {path_buf[pix]['x'][1]:.3f} "
                f"{path_buf[pix]['x'][2]:.3f}\n"
            )

            corres_file.write(
                f"{pix + 1} {path_buf[pix]['x'][0]:.3f} "
                f"{path_buf[pix]['x'][1]:.3f} {path_buf[pix]['x'][2]:.3f} "
                f"{cor_buf[pix]['p'][0]} {cor_buf[pix]['p'][1]} "
                f"{cor_buf[pix]['p'][2]} {cor_buf[pix]['p'][3]}\n"
            )

            if prio_file_base is not None:
                prio_file.write(
                    f"{path_buf[pix]['prev_frame']} {path_buf[pix]['next_frame']} "
                    f"{path_buf[pix]['x'][0]:.3f} {path_buf[pix]['x'][1]:.3f} "
                    f"{path_buf[pix]['x'][2]:.3f} {path_buf[pix]['prio']}\n"
                )

        corres_file.close()
        linkage_file.close()
        if prio_file_base is not None:
            prio_file.close()

        success = True

    except IOError as e:
        print(f"Can't open file {e.filename} for writing")

    return success


def match_coords(
    targs: np.ndarray,
    cpar: ControlPar,
    cal: Calibration,
    tol: float = 1e-5,
    reset_numbers: bool = False,
) -> np.ndarray:
    """Match coordinates from all cameras into a single block.

    replaces MatchedCoords class in Cython

    The output is the same as the number on one ``target`` from the block
    to which this block is kept matched. This block is x-sorted.

    NB: the data is not meant to be directly manipulated at this point. The
    coord_2d arrays are most useful as intermediate objects created and
    manipulated only by other liboptv functions. Although one can imagine a
    use case for direct manipulation in Python, it is rare and supporting it
    is a low priority.

    """
    matched_coords = np.ndarray(len(targs), dtype=Coord2d_dtype)

    for tnum, targ in enumerate(targs):
        # targ = targs[tnum]
        if reset_numbers:
            targ['pnr'] = tnum

        x, y = pixel_to_metric(targ['x'], targ['y'], cpar)
        matched_coords[tnum]['x'], matched_coords[tnum]['y'] = dist_to_flat(x, y, cal, tol)
        matched_coords[tnum]['pnr'] = targ['pnr']

    matched_coords.sort(order='x')

    return matched_coords


def matched_coords_as_arrays(
    matched_coords: np.ndarray, #Coord2d
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the data associated with the object (the matched coordinates.

    block) as NumPy arrays.

    Returns
    -------
    pos - (n,2) array, the (x,y) flat-coordinates position of n targets.
    pnr - n-length array, the corresponding target number for each point.
    """
    num_pts = matched_coords.shape[0]
    pos = np.empty((num_pts, 2))
    pnr = np.empty(num_pts, dtype=np.int_)

    for pt, mc in enumerate(matched_coords):
        pos[pt, 0] = mc['x']
        pos[pt, 1] = mc['y']
        pnr[pt] = mc['pnr']

    return pos, pnr


def get_by_pnrs(matched_coords: List[np.ndarray], pnrs: np.ndarray) -> np.ndarray: #Coord2d
    """
    Return the flat positions of points whose pnr property is given, as an.

    (n,2) flat position array. Assumes all pnrs are to be found, otherwise
    there will be garbage at the end of the position array.
    """
    pos = np.full((len(pnrs), 2), COORD_UNUSED, dtype=np.float64)
    for pt in matched_coords:
        which = np.flatnonzero(pt['pnr'] == pnrs)
        if len(which) > 0:
            which = which[0]
            pos[which, 0] = pt['x']
            pos[which, 1] = pt['y']

    return pos
