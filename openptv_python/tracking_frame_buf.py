"""Tracking frame buffer."""
import pickle
from collections import deque
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from .constants import NEXT_NONE, POSI, PREV_NONE, PRIO_DEFAULT
from .correspondences import Correspond


@dataclass
class Target:
    pnr: int = 0
    x: float = 0.0
    y: float = 0.0
    n: int = 0
    nx: int = 0
    ny: int = 0
    sumg: int = 0
    tnr: int = 0

    def __eq__(self, other: "Target") -> bool:
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

                target = Target(
                    pnr=int(line[0]),
                    x=float(line[1]),
                    y=float(line[2]),
                    n=int(line[3]),
                    nx=int(line[4]),
                    ny=int(line[5]),
                    sumg=int(line[6]),
                    tnr=int(line[7]),
                )

                buffer.append(target)

    except IOError as err:
        print(f"Can't open ascii file: {filename}")
        raise err

    return buffer


def write_targets(targets, num_targets, file_base, frame_num):
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
    decis: List[float] = [] * POSI
    finaldecis: float = 0.0
    linkdecis: List[int] = [] * POSI
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

    num_cams: int = 1
    max_targets: int = 0
    path_info: Pathinfo = Pathinfo()
    correspond: Correspond = Correspond()
    targets: List[List[Target]] = [[]]
    num_parts: int = 0
    num_targets: List[int] = []

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
        for i in range(buf_len):
            self.buf.append(Frame(num_cams, max_targets))
        self.start = 0

    def __del__(self):
        for frame in self.buf:
            del frame

    def read_frame_at_end(self, frame):
        self.buf.append(frame)

    def write_frame_from_start(self, frame):
        pickled_frame = pickle.dumps(frame)
        self._advance_buffer_start()
        return pickled_frame

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


class FrameBuf(FrameBufBase):
    def __init__(
        self,
        buf_len,
        num_cams,
        max_targets,
        corres_file_base,
        linkage_file_base,
        prio_file_base,
        target_file_base,
    ):
        super().__init__(buf_len, num_cams, max_targets)
        self.corres_file_base = corres_file_base
        self.linkage_file_base = linkage_file_base
        self.prio_file_base = prio_file_base
        self.target_file_base = target_file_base

    def write_frame_from_start(self, frame_num):
        """Write a frame to disk and advance the buffer."""
        # Pickle the frame
        pickle.dumps(self.buf[0])

        # Advance the buffer
        self.buf.appendleft(Frame(self.num_cams, self.max_targets))

        # Write the frame to disk
        return write_path_frame(
            self.buf[0],
            self.corres_file_base,
            self.linkage_file_base,
            self.prio_file_base,
            self.target_file_base,
            frame_num,
        )

    def fb_disk_read_frame_at_end(self, frame_num, read_links):
        if read_links:
            return read_path_frame(
                self.buf[-1],
                self.corres_file_base,
                self.linkage_file_base,
                self.prio_file_base,
                self.target_file_base,
                frame_num,
            )
        else:
            return read_path_frame(
                self.buf[-1],
                self.corres_file_base,
                None,
                None,
                self.target_file_base,
                frame_num,
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
    cor_buf: List[Correspond],
    path_buf: List[Pathinfo],
    num_parts: int,
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    frame_num: int,
):
    """Write a frame of path and correspondence info.

    /* write_path_frame() writes the correspondence and linkage information for a
    * frame with the next and previous frames. The information is weirdly
    * distributed among two files. The rt_is file holds correspondence and
    * position data, and the ptv_is holds linkage data.
    *
    * Arguments:
    * corres *cor_buf - an array of corres structs to write to the files.
    * Pathinfo *path_buf - same for path info structures.
    * int num_parts - number of particles represented by the cor_buf and path_buf
    *   arrays, i.e. the arrays' length.
    * char* corres_file_base, *linkage_file_base - base names of the output
    *   correspondence and likage files respectively, to which a frame number
    *   is added. Without separator.
    * char *prio_file_base - for the linkage file with added 'prio' column.
    * int frame_num - number of frame to add to file_base. The '.' separator is
    * added between the name and the frame number.
    *
    * Returns:
    * True on success. 0 on failure.
    */

    Args:
    ----
        cor_buf (_type_): _description_
        path_buf (_type_): _description_
        num_parts (_type_): _description_
        corres_file_base (_type_): _description_
        linkage_file_base (_type_): _description_
        prio_file_base (_type_): _description_
        frame_num (_type_): _description_

    Returns:
    -------
        _type_: _description_
    """
    corres_fname = f"{corres_file_base}.{frame_num}"
    corres_file = open(corres_fname, "w", encoding="utf-8")
    if not corres_file:
        print(f"Can't open file {corres_fname} for writing")
        return 0

    linkage_fname = f"{linkage_file_base}.{frame_num}"
    linkage_file = open(linkage_fname, "w", encoding="utf-8")
    if not linkage_file:
        print(f"Can't open file {linkage_fname} for writing")
        corres_file.close()
        return 0

    print(f"{num_parts}\n", file=corres_file)
    print(f"{num_parts}\n", file=linkage_file)

    if prio_file_base:
        prio_fname = f"{prio_file_base}.{frame_num}"
        prio_file = open(prio_fname, "w", encoding="utf-8")
        if not prio_file:
            print(f"Can't open file {prio_fname} for writing")
            corres_file.close()
            linkage_file.close()
            return 0
        print(f"{num_parts}\n", file=prio_file)

    for pix in range(num_parts):
        print(
            f"{path_buf[pix].prev:4d} {path_buf[pix].next:4d} {path_buf[pix].x[0]:10.3f} \
            {path_buf[pix].x[1]:10.3f} {path_buf[pix].x[2]:10.3f}",
            file=linkage_file,
        )

        print(
            f"{pix+1:4d} {path_buf[pix].x[0]:9.3f} {path_buf[pix].x[1]:9.3f} {path_buf[pix].x[2]:9.3f} \
            {cor_buf[pix].Pathinfo[0]:4d} {cor_buf[pix].Pathinfo[1]:4d} {cor_buf[pix].Pathinfo[2]:4d} {cor_buf[pix].Pathinfo[3]:4d}",
            file=corres_file,
        )

        if not prio_file_base:
            continue
        print(
            f"{path_buf[pix].prev:4d} {path_buf[pix].next:4d} {path_buf[pix].x[0]:10.3f} \
            {path_buf[pix].x[1]:10.3f} {path_buf[pix].x[2]:10.3f} {path_buf[pix].prio}",
            file=prio_file,
        )

    corres_file.close()
    linkage_file.close()
    if prio_file_base:
        prio_file.close()

    return 1
