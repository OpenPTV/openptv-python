""" Tracking frame buffer."""
from dataclasses import dataclass
from typing import Any, List, Tuple
from openptv_python.correspondences import Correspond

# from openptv_python.parameters import ControlPar


POSI = 80
STR_MAX_LEN = 255
PT_UNUSED = -999
PREV_NONE = -1
NEXT_NONE = -2
PRIO_DEFAULT = 2


@dataclass
class Target:
    pnr: int
    x: float
    y: float
    n: int
    nx: int
    ny: int
    sumg: int
    tnr: int

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
    success = 0
    file_name = (
        file_base + "_targets"
        if frame_num == 0
        else f"{file_base}{frame_num:04d}_targets"
    )

    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"{num_targets}\n")
            for target in targets:
                f.write(
                    f"{target.pnr:4d} {target.x:9.4f} {target.y:9.4f} {target.n:5d} {target.nx:5d} {target.ny:5d} {target.sumg:5d} {target.tnr:5d}\n"
                )
        success = 1
    except IOError:
        print(f"Can't open ascii file: {file_name}")

    return success


@dataclass
class PathInfo:
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    prev_link: int = PREV_NONE
    next_link: int = NEXT_NONE
    priority: int = PRIO_DEFAULT
    decision_criteria: List[float] = []
    final_decision_criteria: float = 0.0
    link_candidates: List[int] = []
    candidate_count: int = 0

    def __eq__(self, other: "PathInfo") -> bool:
        if not (
            self.prev_link == other.prev_link
            and self.next_link == other.next_link
            and self.priority == other.priority
            and self.final_decision_criteria == other.final_decision_criteria
            and self.candidate_count == other.candidate_count
            and self.position == other.position
            and self.decision_criteria == other.decision_criteria
            and self.link_candidates == other.link_candidates
        ):
            return False

        for itr in range(POSI):
            if self.decision_criteria[itr] != other.decision_criteria[itr]:
                return False
            if self.link_candidates[itr] != other.link_candidates[itr]:
                return False

        return True

    def register_link_candidate(self, fitness: float, candidate_index: int) -> None:
        self.decision_criteria[self.candidate_count] = fitness
        self.link_candidates[self.candidate_count] = candidate_index
        self.candidate_count += 1

    def reset_links(self) -> None:
        self.prev_link = PREV_NONE
        self.next_link = NEXT_NONE
        self.priority = PRIO_DEFAULT


@dataclass
class Frame:
    path_info: PathInfo = PathInfo()
    correspond: Correspond = Correspond()
    targets: List[Target] = []
    num_cams: int = 1
    max_targets: int = 0
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


def read_path_frame(
    cor_buf: Any,
    path_buf: Any,
    corres_file_base: Any,
    linkage_file_base: Any,
    prio_file_base: Any,
    frame_num: Any,
) -> Any:
    """
    Read a frame of path and correspondence info.

    Args:
    ----
        cor_buf : _type_
            a buffer of corres structs to fill in from the file.
        path_buf : _type_
            a buffer of path info structures.
        corres_file_base : _type_
            base name of the output correspondence file to which a frame number
            is added without separator.
        linkage_file_base : _type_
            base name of the output linkage file to which a frame number
            is added without separator.
        prio_file_base : _type_
            base name of the output priorities file to which a frame number
            is added without separator.
        frame_num : _type_
            number of frame to add to file_base. A value of 0 or less means that no
            frame number should be added. The '.' separator is added between the name
            and the frame number.

    Returns:
    -------
        _type_:
            The number of points read for this frame. -1 on failure.
    """
    filein = None
    linkagein = None
    prioin = None
    fname = ""
    read_res, targets, alt_link = 0, -1, 0

    # File format: first line contains the number of points, then each line
    # is a record of path and correspondence info. We don't need the number of
    # points because we read to EOF anyway.
    fname = f"{corres_file_base}.{frame_num}"
    try:
        filein = open(fname, "r", encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Can't open ascii file: {fname}") from e

    read_res = int(filein.readline().strip())
    if not read_res:
        return targets

    if linkage_file_base is not None:
        fname = f"{linkage_file_base}.{frame_num}"
        try:
            linkagein = open(fname, "r", encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Can't open linkage file: {fname}") from e

        read_res = int(linkagein.readline().strip())
        if not read_res:
            return targets

    if prio_file_base is not None:
        fname = f"{prio_file_base}.{frame_num}"
        try:
            prioin = open(fname, "r", encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Can't open prioin file: {fname}") from e

        read_res = int(prioin.readline().strip())
        if not read_res:
            return targets

    targets = 0
    while True:
        if linkagein is not None:
            line = linkagein.readline().strip()
            if not line:
                break

            data = list(map(float, line.split()))
            path_buf.prev = int(data[0])
            path_buf.next = int(data[1])
        else:
            # Defaults:
            path_buf.prev = -1
            path_buf.next = -2

        if prioin is not None:
            line = prioin.readline().strip()
            if not line:
                break

            data = list(map(float, line.split()))
            path_buf.prio = int(data[5])
        else:
            path_buf.prio = 4

        # Initialize tracking-related transient variables. These never get
        # saved or restored.
        path_buf.inlist = 0
        path_buf.finaldecis = 1000000.0

        for alt_link in range(POSI):
            path_buf.decis[alt_link] = 0.0
            path_buf.linkdecis[alt_link] = -999

        # Rest of values:
        line = filein.readline().strip()
        if not line:
            break

        data = list(map(float, line.split()))
        path_buf.x[0] = data[1]
        path_buf.x[1] = data[2]
        path_buf.x[2] = data[3]
        cor_buf.p[0] = int(data[4])
        cor_buf.p[1] = int(data[5])
        cor_buf.p[2] = int(data[6])
        cor_buf.p[3] = int(data[7])

        targets += 1
        cor_buf += 1
        path_buf += 1

    if filein is not None:
        filein.close()
    if linkagein is not None:
        linkagein.close()
    if prioin is not None:
        prioin.close()

    return targets


def write_path_frame(
    cor_buf,
    path_buf,
    num_parts,
    corres_file_base,
    linkage_file_base,
    prio_file_base,
    frame_num,
):
    """Write a frame of path and correspondence info.

    /* write_path_frame() writes the correspondence and linkage information for a
    * frame with the next and previous frames. The information is weirdly
    * distributed among two files. The rt_is file holds correspondence and
    * position data, and the ptv_is holds linkage data.
    *
    * Arguments:
    * corres *cor_buf - an array of corres structs to write to the files.
    * P *path_buf - same for path info structures.
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
            {cor_buf[pix].p[0]:4d} {cor_buf[pix].p[1]:4d} {cor_buf[pix].p[2]:4d} {cor_buf[pix].p[3]:4d}",
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


@dataclass
class FramebufBase:
    buf_len: int = 4
    num_cams: int = 1
    _ring_vec: List[Any] = [None] * (2 * buf_len)
    buf: List[Frame] = [Frame()] * buf_len
    _vptr: Any = None

    def fb_read_frame_at_end(self, frame_num, read_links):
        return self._vptr.read_frame_at_end(self, frame_num, read_links)

    def fb_write_frame_from_start(self, frame_num):
        return self._vptr.write_frame_from_start(self, frame_num)

    def fb_base_init(self, buf_len, num_cams, max_targets):
        self.buf_len = buf_len
        self.num_cams = num_cams

        self._ring_vec = [Frame() for _ in range(buf_len * 2)]
        self.buf = self._ring_vec[buf_len:]

        for _ in self.buf:
            # self.frame_init(frame, num_cams, max_targets)
            _ = Frame(num_cams, max_targets)

        # self._vptr = fb_vtable()


@dataclass
class Framebuf:
    buf_len: int
    num_cams: int
    max_targets: int
    corres_file_base: str
    linkage_file_base: str
    prio_file_base: str
    target_file_base: str

    def __post_init__(self):
        self.base = FramebufBase(self.buf_len, self.num_cams)
        # self.base._vptr = self
        # self.base.base_init(self.max_targets)

    def free(self):
        pass

    def read_frame_at_end(self, frame_num, read_links):
        pass

    def write_frame_from_start(self, frame_num):
        pass


class FramebufDisk(Framebuf):
    def free(self):
        pass

    def read_frame_at_end(self, frame_num, read_links):
        pass

    def write_frame_from_start(self, frame_num):
        pass


# # Define the virtual function table as a dictionary of function pointers.
# fb_vtable = {
#     'free': None,
#     'read_frame_at_end': None,
#     'write_frame_from_start': None
# }

# # Define a decorator that adds a virtual method to the vtable.
# def virtual_method(name):
#     def decorator(func):
#         fb_vtable[name] = func
#         return func
#     return decorator

# # Define the base class for frame buffer objects.
# class framebuf_base:
#     def __init__(self, buf_len, num_cams):
#         self._vptr = fb_vtable.copy()
#         self.buf = [None] * buf_len
#         self._ring_vec = [None] * (buf_len * 2)
#         self.buf_len = buf_len
#         self.num_cams = num_cams

#     @virtual_method('free')
#     def free(self):
#         pass

#     @virtual_method('read_frame_at_end')
#     def read_frame_at_end(self, frame_num, read_links):
#         pass

#     @virtual_method('write_frame_from_start')
#     def write_frame_from_start(self, frame_num):
#         pass

#     def next(self):
#         pass

#     def prev(self):
#         pass

# # Define the child class that reads from _target files.
# class framebuf(framebuf_base):
#     def __init__(self, buf_len, num_cams, max_targets, corres_file_base,
#                  linkage_file_base, prio_file_base, target_file_base):
#         super().__init__(buf_len, num_cams)
#         self.corres_file_base = corres_file_base
#         self.linkage_file_base = linkage_file_base
#         self.prio_file_base = prio_file_base
#         self.target_file_base = target_file_base

#     @virtual_method('free')
#     def disk_free(self):
#         pass

#     @virtual_method('read_frame_at_end')
#     def disk_read_frame_at_end(self, frame_num, read_links):
#         pass

#     @virtual_method('write_frame_from_start')
#     def disk_write_frame_from_start(self, frame_num):
#         pass

# # Define a function to initialize a frame buffer.
# def fb_init(buf_len, num_cams, max_targets, corres_file_base,
#             linkage_file_base, prio_file_base, target_file_base):
#     buf = framebuf(buf_len, num_cams, max_targets, corres_file_base,
#                    linkage_file_base, prio_file_base, target_file_base)
#     buf.free = buf.disk_free
#     buf.read_frame_at_end = buf.disk_read_frame_at_end
#     buf.write_frame_from_start = buf.disk_write_frame_from_start
#     return buf
