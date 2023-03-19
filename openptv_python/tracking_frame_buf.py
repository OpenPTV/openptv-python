from typing import List

from openptv_python.vec_utils import vec3d

POSI = 80


class Target:
    def __init__(
        self,
        pnr: int,
        x: float,
        y: float,
        n: int,
        nx: int,
        ny: int,
        sumg: int,
        tnr: int,
    ):
        self.pnr = pnr
        self.x = x
        self.y = y
        self.n = n
        self.nx = nx
        self.ny = ny
        self.sumg = sumg
        self.tnr = tnr

    def __eq__(self, other):
        """Compare two targets.

        /* Check that target self is equal to target other, i.e. all their fields are equal.
            *
            * Arguments:
            * target *self, *other - the target structures to compare.
            *
            * Returns:
            * true if all fields are equal, false otherwise.
            */

        Args:
        ----
            self (target): target
            other (target): target

        Returns:
        -------
            _type_: _description_
        """
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
    tix = 0
    num_targets = 0
    filein = ""

    if frame_num > 0:
        filein = f"{file_base}{frame_num:04}_targets"
    else:
        filein = f"{file_base}_targets"

    try:
        with open(filein, "r", encoding="utf-8") as f:
            num_targets = int(f.readline().strip())
            buffer = [Target] * num_targets

            for tix in range(num_targets):
                line = f.readline().strip().split()
                if len(line) != 8:
                    raise ValueError("Bad format for file: {}".format(filein))
                buffer[tix].pnr = int(line[0])
                buffer[tix].x = float(line[1])
                buffer[tix].y = float(line[2])
                buffer[tix].n = int(line[3])
                buffer[tix].nx = int(line[4])
                buffer[tix].ny = int(line[5])
                buffer[tix].sumg = int(line[6])
                buffer[tix].tnr = int(line[7])
    except IOError:
        print(f"Can't open ascii file: {filein}")
    except ValueError as err:
        print(str(err))

    return buffer


def write_targets(
    buffer: List[Target], num_targets: int, file_base: str, frame_num: int
) -> int:
    FILEOUT = None
    tix, printf_ok, success = 0, 0, 0
    fileout = ""

    if frame_num == 0:
        fileout = file_base + "_targets"
    else:
        fileout = f"{file_base}{frame_num:04d}_targets"

    try:
        FILEOUT = open(fileout, "w", encoding="utf-8")
    except IOError:
        print(f"Can't open ascii file: {fileout}")
        return success

    if FILEOUT.write(f"{num_targets}\n") <= 0:
        print(f"Write error in file {fileout}")
        return success

    for tix in range(num_targets):
        t = buffer[tix]
        printf_ok = FILEOUT.write(
            f"{t.pnr:4d} {t.x:9.4f} {t.y:9.4f} {t.n:5d} {t.nx:5d} {t.ny:5d} {t.sumg:5d} {t.tnr:5d}\n"
        )
        if printf_ok != 0:
            print(f"Write error in file {fileout}")
            return success
    success = 1
    return success


class Correspondence:
    def __init__(self, nr: int, p: List[int]):
        self.nr = nr
        self.p = p

    def __eq__(self, other) -> bool:
        """Compare two Correspondence objects.

        Args:
        ----
            __o (Correspondence): _description_

        Returns:
        -------
            bool: True if the two objects are equal, False otherwise.
        """
        return (
            (self.nr == other.nr)
            and (self.p[0] == other.p[0])
            and (self.p[1] == other.p[1])
            and (self.p[2] == other.p[2])
            and (self.p[3] == other.p[3])
        )


class PathInfo:
    POSI = 80
    PREV_NONE = -1
    NEXT_NONE = -2
    PRIO_DEFAULT = 2

    def __init__(
        self,
        x: vec3d = [0.0, 0.0, 0.0],  # position
        prv: int = PREV_NONE,  # pointer to prev or next link
        nxt: int = NEXT_NONE,  # next is preserved in python
        prio: int = PRIO_DEFAULT,  # Prority of link is used for differen levels
        decis: List[
            float
        ] = [],  # Bin for decision critera of possible links to next dataset
        finaldecis: float = 0.0,  # final decision critera by which the link was established
        linkdecis: List[int] = [],  # pointer of possible links to next data set
        inlist: int = 0,  # Counter of number of possible links to next data set
    ):
        self.x = x
        self.prv = prv  # previous
        self.nxt = nxt
        self.prio = prio
        self.decis = decis
        self.finaldecis = finaldecis
        self.linkdecis = linkdecis
        self.inlist = inlist

    def __eq__(self, other) -> bool:
        """Compare two PathInfo objects."""
        if not (
            self.prv == other.prv  # previous
            and self.nxt == other.nxt
            and self.prio == other.prio
            and self.finaldecis == other.finaldecis
            and self.inlist == other.inlist
            and self.x == other.x
            and self.decis == other.decis
            and self.linkdecis == other.linkdecis
        ):
            return False

        for itr in range(self.POSI):
            if self.decis[itr] != other.decis[itr]:
                return False
            if self.linkdecis[itr] != other.linkdecis[itr]:
                return False

        return True

    def register_link_candidate(self, fitness: float, cand: int):
        """Register a link candidate for a target.

        /* register_link_candidate() adds information on a possible link to a path info
        * structure.
        *
        * Arguments:
        * P *self - the path info structure to modify.
        * fitness_t fitness - the likelihood of the link candidate to actually be the
        *     correct link, based on physical criteria.
        * int cand - the candidate's index in nxt frame's target list.
        */
        """
        self.decis[self.inlist] = fitness
        self.linkdecis[self.inlist] = cand
        self.inlist += 1

    def reset_links(self):
        self.prv = self.PREV_NONE  # previous
        self.nxt = self.NEXT_NONE
        self.prio = self.PRIO_DEFAULT


class Frame:
    def __init__(
        self,
        path_info: PathInfo,
        correspond: Correspondence,
        targets: List[Target],
        num_cams: int,
        max_targets: int,
        num_parts: int,  # Number of 3D particles in the correspondence buffer
        num_targets: List[int],  # Pointer to array of 2D particle counts per image.
    ):
        self.path_info = path_info
        self.correspond = correspond
        self.targets = targets
        self.num_cams = num_cams
        self.max_targets = max_targets
        self.num_parts = num_parts
        self.num_targets = num_targets

    def read_frame(
        self,
        corres_file_base,
        linkage_file_base,
        prio_file_base,
        target_file_base,
        frame_num,
    ):
        """Read a frame from the disk.

        Args:
        ----
            corres_file_base (_type_): _description_
            linkage_file_base (_type_): _description_
            prio_file_base (_type_): _description_
            target_file_base (_type_): _description_
            frame_num (_type_): _description_

        Returns:
        -------
            _type_: _description_
        """
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

        if self.num_targets == 0:
            return False

        for cam in range(self.num_cams):
            self.targets[cam] = read_targets(target_file_base[cam], frame_num)
            self.num_targets[cam] = len(self.targets[cam])
            if self.num_targets[cam] == -1:
                return False

        return True

    def write_frame(
        self,
        corres_file_base,
        linkage_file_base,
        prio_file_base,
        target_file_base,
        frame_num,
    ):
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
    cor_buf, path_buf, corres_file_base, linkage_file_base, prio_file_base, frame_num
):
    """Read a frame of path and correspondence info.

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

    Args:
    ----
        cor_buf (_type_): _description_
        path_buf (_type_): _description_
        corres_file_base (_type_): _description_
        linkage_file_base (_type_): _description_
        prio_file_base (_type_): _description_
        frame_num (_type_): _description_

    Returns:
    -------
        _type_: _description_
    """
    filein, linkagein, prioin = None, None, None
    fname = ""
    read_res, targets, alt_link = 0, -1, 0

    # File format: first line contains the number of points, then each line is
    # a record of path and correspondence info. We don't need the nuber of points
    # because we read to EOF anyway.
    fname = f"{corres_file_base}.{frame_num}"
    try:
        filein = open(fname, "r")
    except:
        print(f"Can't open ascii file: {fname}")
        return targets

    read_res = int(filein.readline().strip())
    if not read_res:
        return targets

    if linkage_file_base is not None:
        fname = f"{linkage_file_base}.{frame_num}"
        try:
            linkagein = open(fname, "r")
        except:
            print(f"Can't open linkage file: {fname}")
            return targets

        read_res = int(linkagein.readline().strip())
        if not read_res:
            return targets

    if prio_file_base is not None:
        fname = f"{prio_file_base}.{frame_num}"
        try:
            prioin = open(fname, "r")
        except:
            print(f"Can't open prio file: {fname}")
            return targets

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
    corres_file = open(corres_fname, "w")
    if not corres_file:
        print(f"Can't open file {corres_fname} for writing")
        return 0

    linkage_fname = f"{linkage_file_base}.{frame_num}"
    linkage_file = open(linkage_fname, "w")
    if not linkage_file:
        print(f"Can't open file {linkage_fname} for writing")
        corres_file.close()
        return 0

    print(f"{num_parts}\n", file=corres_file)
    print(f"{num_parts}\n", file=linkage_file)

    if prio_file_base:
        prio_fname = f"{prio_file_base}.{frame_num}"
        prio_file = open(prio_fname, "w")
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


class FramebufBase:
    def __init__(self, buf_len, num_cams):
        self._vptr = None
        self.buf = [None] * buf_len
        self._ring_vec = [None] * (2 * buf_len)
        self.buf_len = buf_len
        self.num_cams = num_cams

    # def fb_free(self):
    #     self._vptr.free(self)

    def fb_read_frame_at_end(self, frame_num, read_links):
        return self._vptr.read_frame_at_end(self, frame_num, read_links)

    def fb_write_frame_from_start(self, frame_num):
        return self._vptr.write_frame_from_start(self, frame_num)

    def fb_base_init(self, buf_len, num_cams, max_targets):
        self.buf_len = buf_len
        self.num_cams = num_cams

        self._ring_vec = [Frame() for _ in range(buf_len * 2)]
        self.buf = self._ring_vec[buf_len:]

        for frame in self.buf:
            # self.frame_init(frame, num_cams, max_targets)
            Frame(num_cams, max_targets)

        self._vptr = fb_vtable()


class Framebuf:
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
        self.base = FramebufBase(buf_len, num_cams)
        self.base._vptr = self
        self.corres_file_base = corres_file_base
        self.linkage_file_base = linkage_file_base
        self.prio_file_base = prio_file_base
        self.target_file_base = target_file_base
        self.base.base_init(max_targets)

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
