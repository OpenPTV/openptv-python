"""Test the Frame class."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from openptv_python.tracking_frame_buf import (
    Frame,
)


class TestFrame(unittest.TestCase):
    """Test the Frame class."""

    def test_read_frame(self):
        """Reading a frame.

        num_cams: int = field(default_factory=int)
        max_targets: int = MAX_TARGETS
        path_info: Pathinfo | None = None
        correspond: Correspond | None = None
        targets: List[List[Target]] | None = None
        num_parts: int = 0
        num_targets: List[int] | None = None

            corres_file_base: Any,
            linkage_file_base: Any,
            prio_file_base: Any,
            target_file_base: List[Any],
            frame_num: int,

        """
        targ_files = [f"tests/testing_folder/frame/cam{c:d}.%04d" for c in range(1, 5)]
        frm = Frame(num_cams=4)
        frm.read(
            corres_file_base="tests/testing_folder/frame/rt_is.%d",
            linkage_file_base="tests/testing_folder/frame/ptv_is.%d",
            prio_file_base="tests/testing_folder/frame/added.%d",
            target_file_base=targ_files,
            frame_num=333,
        )

        pos = frm.positions()
        self.assertEqual(pos.shape, (10, 3))

        targs = frm.target_positions_for_camera(3)
        self.assertEqual(targs.shape, (10, 2))

        targs_correct = np.array(
            [
                [426.0, 199.0],
                [429.0, 60.0],
                [431.0, 327.0],
                [509.0, 315.0],
                [345.0, 222.0],
                [465.0, 139.0],
                [487.0, 403.0],
                [241.0, 178.0],
                [607.0, 209.0],
                [563.0, 238.0],
            ]
        )
        np.testing.assert_array_equal(targs, targs_correct)

    def test_read_frame_returns_false_when_linkage_file_missing(self):
        """Missing linkage data must fail the frame read instead of producing an empty frame."""
        source_dir = Path("tests/testing_folder/frame")

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            for path in source_dir.iterdir():
                shutil.copy2(path, work_dir / path.name)

            (work_dir / "ptv_is.333").unlink()

            targ_files = [str(work_dir / f"cam{c:d}.%04d") for c in range(1, 5)]
            frm = Frame(num_cams=4)

            self.assertFalse(
                frm.read(
                    corres_file_base=str(work_dir / "rt_is.%d"),
                    linkage_file_base=str(work_dir / "ptv_is.%d"),
                    prio_file_base=str(work_dir / "added.%d"),
                    target_file_base=targ_files,
                    frame_num=333,
                )
            )


if __name__ == "__main__":
    unittest.main()
