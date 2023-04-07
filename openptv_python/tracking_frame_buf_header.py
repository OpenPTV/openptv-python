from dataclasses import dataclass
from typing import List, Optional
from openptv_python.vec_utils import vec3d
from dvg_ringbuffer import RingBuffer


POSI = 80
STR_MAX_LEN = 255
PT_UNUSED = -999


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


def compare_targets(t1: Target, t2: Target) -> int:
    return t1.pnr - t2.pnr


def read_targets(file_base: str, frame_num: int) -> List[Target]:
    buffer = []
    # Implementation left to the user
    return buffer


def write_targets(
    buffer: List[Target], num_targets: int, file_base: str, frame_num: int
):
    # Implementation left to the user
    pass


@dataclass
class Corres:
    nr: int
    p: List[int]


def compare_corres(c1: Corres, c2: Corres) -> int:
    return c1.nr - c2.nr


CORRES_NONE = -1


fitness_t = float


@dataclass
class P:
    x: vec3d
    prev: int
    next: int
    prio: int = 2
    decis: List[fitness_t] = [0.0] * POSI
    finaldecis: float = 0.0
    linkdecis: List[int] = [0] * POSI
    inlist: int = 0


def compare_path_info(p1: P, p2: P) -> int:
    return id(p1) - id(p2)


def register_link_candidate(self: P, fitness: fitness_t, cand: int):
    # Implementation left to the user
    pass


PREV_NONE = -1
NEXT_NONE = -2


def reset_links(self: P):
    # Implementation left to the user
    pass


def read_path_frame(
    cor_buf: List[Corres],
    path_buf: List[P],
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    frame_num: int,
):
    # Implementation left to the user
    pass


def write_path_frame(
    cor_buf: List[Corres],
    path_buf: List[P],
    num_parts: int,
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    frame_num: int,
):
    # Implementation left to the user
    pass


@dataclass
class Frame:
    path_info: List[P]
    correspond: List[Corres]
    targets: List[Optional[List[Target]]]
    num_cams: int
    max_targets: int
    num_parts: int
    num_targets: List[int]


def frame_init(new_frame: Frame, num_cams: int, max_targets: int):
    new_frame.path_info = []
    new_frame.correspond = []
    new_frame.targets = [None] * num_cams
    new_frame.num_cams = num_cams
    new_frame.max_targets = max_targets
    new_frame.num_parts = 0
    new_frame.num_targets = [0] * num_cams


def free_frame(self: Frame):
    # Implementation left to the user
    pass


def read_frame(
    self: Frame,
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    target_file_base: List[str],
    frame_num: int,
) -> int:
    # Implementation left to the user
    pass


def write_frame(
    self: Frame,
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    target_file_base: List[str],
    frame_num: int,
) -> int:
    # Implementation left to the user
    pass


class FramebufBase:
    _vptr = None
    buf = None
    _ring_vec = None
    buf_len = None
    num_cams = None

    def __init__(self, buf_len: int, num_cams: int, max_targets: int):
        self.buf_len = buf_len
        self.num_cams = num_cams
        fb_base_init(self, buf_len, num_cams, max_targets)

    def free(self):
        self._vptr.free(self)

    def read_frame_at_end(self, frame_num: int, read_links: int) -> int:
        return self._vptr.read_frame_at_end(self, frame_num, read_links)

    def write_frame_from_start(self, frame_num: int) -> int:
        return self._vptr.write_frame_from_start(self, frame_num)

    def next(self):
        fb_next(self)

    def prev(self):
        fb_prev(self)


class Framebuf(FramebufBase):
    corres_file_base = ""
    linkage_file_base = ""
    prio_file_base = ""
    target_file_base = []

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
        FramebufBase.__init__(self, buf_len, num_cams, max_targets)
        fb_init(
            self,
            buf_len,
            num_cams,
            max_targets,
            corres_file_base,
            linkage_file_base,
            prio_file_base,
            target_file_base,
        )

    def free(self):
        fb_disk_free(self)


# Base class virtual function implementations
def fb_free(self):
    pass


def fb_read_frame_at_end(self, frame_num: int, read_links: int) -> int:
    return -1


def fb_write_frame_from_start(self, frame_num: int) -> int:
    return -1


# Base class non-virtual methods
def fb_base_init(new_buf: FramebufBase, buf_len: int, num_cams: int, max_targets: int):
    new_buf.buf = [None] * buf_len
    new_buf._ring_vec = [None] * (buf_len * 2)
    new_buf.num_cams = num_cams
    for i in range(buf_len):
        new_buf._ring_vec[i] = Frame()
        (
            new_buf._ring_vec[i].path_info,
            new_buf._ring_vec[i].correspond,
            new_buf._ring_vec[i].num_targets,
        ) = ([], [], [0] * num_cams)
        for j in range(num_cams):
            new_buf._ring_vec[i].targets.append([None] * max_targets)
    new_buf.buf = new_buf._ring_vec[buf_len:]


def fb_next(self: FramebufBase):
    self.buf.append(self._ring_vec.pop(0))


def fb_prev(self: FramebufBase):
    self._ring_vec.insert(0, self.buf.pop())


# Child class virtual function implementations
def fb_disk_free(self: FramebufBase):
    for i in range(self.buf_len):
        free_frame(self.buf[i])
    del self.buf


def fb_disk_read_frame_at_end(
    self: FramebufBase, frame_num: int, read_links: int
) -> int:
    # Implementation left to the user
    return 0


def fb_disk_write_frame_from_start(self: FramebufBase, frame_num: int) -> int:
    # Implementation left to the user
    return 0


# Child class vtable definition
virtual_funcs = fb_vtable(
    fb_disk_free, fb_disk_read_frame_at_end, fb_disk_write_frame_from_start
)


# Derived class initialization
def fb_init(
    new_buf: Framebuf,
    buf_len: int,
    num_cams: int,
    max_targets: int,
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    target_file_base: List[str],
):
    FramebufBase.__init__(new_buf, buf_len, num_cams, max_targets)
    new_buf._vptr = virtual_funcs
    new_buf.corres_file_base = corres_file_base
    new_buf.linkage_file_base = linkage_file_base
    new_buf.prio_file_base = prio_file_base
    new_buf.target_file_base = target_file_base


# another boilerplate code
s
from collections import deque


class fb_vtable:
    def free(self, self_ref):
        pass

    def read_frame_at_end(self, self_ref, frame_num, read_links):
        pass

    def write_frame_from_start(self, self_ref, frame_num):
        pass


class framebuf_base:
    def __init__(self, buf_len):
        self._vptr = fb_vtable()
        self._ring_vec = deque(maxlen=buf_len * 2)
        self._fb_ptr = 0
        self._max_len = buf_len

    def fb_next(self):
        if self._fb_ptr < len(self._ring_vec) - 1:
            self._fb_ptr += 1

    def fb_prev(self):
        if self._fb_ptr > 0:
            self._fb_ptr -= 1

    def fb_read_frame_at_end(self, frame_num, read_links):
        self._vptr.read_frame_at_end(self, frame_num, read_links)

    def fb_write_frame_from_start(self, frame_num):
        self._vptr.write_frame_from_start(self, frame_num)


class framebuf_target(framebuf_base):
    def __init__(self, buf_len, target_file):
        super().__init__(buf_len)
        self._target_file = target_file
        self._vptr.read_frame_at_end = self.read_frame_at_end
        self._vptr.write_frame_from_start = self.write_frame_from_start
        self._vptr.free = self.free

    def read_frame_at_end(self, self_ref, frame_num, read_links):
        # Implementation for reading frame information from target file
        pass

    def write_frame_from_start(self, self_ref, frame_num):
        # Implementation for writing frame information to target file
        pass

    def free(self, self_ref):
        # Implementation for freeing memory
        pass


# we can use https://pypi.org/project/dvg-ringbuffer/


@dataclass
class Test:
    x: int
    y: float


x = RingBuffer(capacity=3, dtype=Test)

x.append(Test(1, 2.0))
x[0].x
x[0].y
