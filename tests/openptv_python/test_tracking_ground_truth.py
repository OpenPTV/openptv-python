import io
import os
import re
import shutil
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np

import openptv_python.track as track
from openptv_python._native_compat import HAS_OPTV
from openptv_python._native_convert import to_native_calibration, to_native_control_par
from openptv_python.calibration import Calibration
from openptv_python.constants import CORRES_NONE
from openptv_python.imgcoord import image_coordinates
from openptv_python.parameters import ControlPar, read_volume_par
from openptv_python.track import (
    track_forward_start,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv_python.tracking_frame_buf import (
    Frame,
    Pathinfo,
    Target,
    read_path_frame,
    write_path_frame,
    write_targets,
)
from openptv_python.tracking_run import tr_new
from openptv_python.trafo import metric_to_pixel

try:
    from optv.parameters import SequenceParams as NativeSequenceParams
    from optv.parameters import TrackingParams as NativeTrackingParams
    from optv.parameters import VolumeParams as NativeVolumeParams
    from optv.tracker import Tracker as NativeTracker

    HAS_NATIVE_TRACKING = True
except ImportError:
    NativeSequenceParams = None
    NativeTrackingParams = None
    NativeVolumeParams = None
    NativeTracker = None
    HAS_NATIVE_TRACKING = False


FRAMES = tuple(range(10001, 10007))
PERMISSIVE_BOUNDS = {
    "dacc": 2.0,
    "dangle": 100.0,
    "dvxmin": -0.02,
    "dvxmax": 0.02,
    "dvymin": -0.02,
    "dvymax": 0.02,
    "dvzmin": -0.02,
    "dvzmax": 0.02,
}
STRICT_BOUNDS = {
    "dacc": 2.0,
    "dangle": 100.0,
    "dvxmin": -0.005,
    "dvxmax": 0.005,
    "dvymin": -0.005,
    "dvymax": 0.005,
    "dvzmin": -0.005,
    "dvzmax": 0.005,
}
EXPECTED_PERMISSIVE_SNAPSHOT = {
    10001: [
        (-1, 0, (2, 1, 1, -1), (0.0, 0.0, 0.0)),
        (-1, 1, (1, 0, 0, -1), (0.02, 0.01, 0.0)),
    ],
    10002: [
        (0, 0, (2, 1, 1, -1), (0.01, 0.0, 0.0)),
        (1, 1, (1, 0, 0, -1), (0.03, 0.01, 0.0)),
    ],
    10003: [
        (0, 0, (2, 1, 1, -1), (0.02, 0.0, 0.0)),
        (1, 1, (1, 0, 0, -1), (0.04, 0.01, 0.0)),
    ],
    10004: [
        (0, 0, (2, 1, 1, -1), (0.03, 0.0, 0.0)),
        (1, 1, (1, 0, 0, -1), (0.05, 0.01, 0.0)),
    ],
    10005: [
        (0, 0, (2, 1, 1, -1), (0.04, 0.0, 0.0)),
        (1, 1, (1, 0, 0, -1), (0.06, 0.01, 0.0)),
    ],
    10006: [
        (0, -2, (2, 1, 1, -1), (0.05, 0.0, 0.0)),
        (1, -2, (1, 0, 0, -1), (0.07, 0.01, 0.0)),
    ],
}
EXPECTED_STRICT_SNAPSHOT = {
    frame: [
        (-1, -2, (2, 1, 1, -1), (round(0.01 * (frame - 10001), 5), 0.0, 0.0)),
        (-1, -2, (1, 0, 0, -1), (round(0.02 + 0.01 * (frame - 10001), 5), 0.01, 0.0)),
    ]
    for frame in FRAMES
}
EXPECTED_PERMISSIVE_STEP_STATS = {frame: (2, 0) for frame in FRAMES[:-1]}
EXPECTED_STRICT_STEP_STATS = {frame: (0, 2) for frame in FRAMES[:-1]}
BAD_FILE_LOOKUP_PATTERN = re.compile(r"Can't open ascii file|\d{10}_targets")

CONSTANT_DIAGONAL_TRACKS = {
    0: [
        np.array([0.01 * index, 0.01 * index, 0.01 * index])
        for index in range(len(FRAMES))
    ],
}
TURNING_TRACKS = {
    0: [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.01, 0.0, 0.001]),
        np.array([0.018, 0.006, 0.002]),
        np.array([0.023, 0.014, 0.003]),
        np.array([0.025, 0.023, 0.004]),
        np.array([0.024, 0.033, 0.005]),
    ],
}
JUMP_SWITCH_TRACKS = {
    0: [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.01, 0.01, 0.01]),
        np.array([0.02, 0.02, 0.02]),
        None,
        None,
        None,
        None,
    ],
    1: [
        None,
        None,
        np.array([0.05, 0.05, 0.05]),
        np.array([0.06, 0.06, 0.06]),
        np.array([0.07, 0.07, 0.07]),
        np.array([0.08, 0.08, 0.08]),
    ],
}
BRANCH_SWITCH_TRACKS = {
    0: [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.01, 0.0, 0.001]),
        np.array([0.018, 0.006, 0.002]),
        None,
        None,
        None,
    ],
    1: [
        None,
        None,
        None,
        np.array([0.024, 0.022, 0.004]),
        np.array([0.030, 0.032, 0.005]),
        np.array([0.036, 0.042, 0.006]),
    ],
}


@dataclass(frozen=True)
class TrackingSnapshotResult:
    snapshot: dict[
        int,
        list[tuple[int, int, tuple[int, int, int, int], tuple[float, float, float]]],
    ]
    log: str
    step_stats: dict[int, tuple[int, int]]


def _make_bounds(*, dv_limit: float, dacc: float, dangle: float) -> dict[str, float]:
    return {
        "dacc": dacc,
        "dangle": dangle,
        "dvxmin": -dv_limit,
        "dvxmax": dv_limit,
        "dvymin": -dv_limit,
        "dvymax": dv_limit,
        "dvzmin": -dv_limit,
        "dvzmax": dv_limit,
    }


def _rounded_position(pos: np.ndarray) -> tuple[float, float, float]:
    values = tuple(round(float(value), 5) for value in pos)
    return cast(tuple[float, float, float], values)


def _snapshot_graph(snapshot):
    return {
        frame: [(prev, next_, pos) for prev, next_, _corres, pos in rows]
        for frame, rows in snapshot.items()
    }


def _candidate_passes_tracking_gate(
    prev_pos: np.ndarray,
    curr_pos: np.ndarray,
    cand_pos: np.ndarray,
    bounds: dict[str, float],
) -> bool:
    diff_pos = cand_pos - curr_pos
    if not (
        bounds["dvxmin"] < diff_pos[0] < bounds["dvxmax"]
        and bounds["dvymin"] < diff_pos[1] < bounds["dvymax"]
        and bounds["dvzmin"] < diff_pos[2] < bounds["dvzmax"]
    ):
        return False

    pred_pos = track.search_volume_center_moving.py_func(prev_pos, curr_pos)
    angle, acc = track.angle_acc.py_func(curr_pos, pred_pos, cand_pos)
    return (acc < bounds["dacc"] and angle < bounds["dangle"]) or (
        acc < bounds["dacc"] / 10
    )


def _single_track_chain_graph(positions: list[np.ndarray]):
    return {
        frame: [
            (
                -1 if index == 0 else 0,
                -2 if index == len(FRAMES) - 1 else 0,
                _rounded_position(pos),
            )
        ]
        for index, (frame, pos) in enumerate(zip(FRAMES, positions))
    }


def _single_track_delayed_chain_graph(positions: list[np.ndarray]):
    return {
        FRAMES[0]: [(-1, -2, _rounded_position(positions[0]))],
        FRAMES[1]: [(-1, 0, _rounded_position(positions[1]))],
        FRAMES[2]: [(0, 0, _rounded_position(positions[2]))],
        FRAMES[3]: [(0, 0, _rounded_position(positions[3]))],
        FRAMES[4]: [(0, 0, _rounded_position(positions[4]))],
        FRAMES[5]: [(0, -2, _rounded_position(positions[5]))],
    }


def _single_track_unlinked_graph(positions: list[np.ndarray]):
    return {
        frame: [(-1, -2, _rounded_position(pos))]
        for frame, pos in zip(FRAMES, positions)
    }


def _switch_graph(tracks, *, inherited: bool):
    first_positions = tracks[0]
    second_positions = tracks[1]
    return {
        FRAMES[0]: [(-1, 0, _rounded_position(first_positions[0]))],
        FRAMES[1]: [(0, 0, _rounded_position(first_positions[1]))],
        FRAMES[2]: [(0, 0 if inherited else -2, _rounded_position(first_positions[2]))],
        FRAMES[3]: [((-1, 0)[inherited], 0, _rounded_position(second_positions[3]))],
        FRAMES[4]: [(0, 0, _rounded_position(second_positions[4]))],
        FRAMES[5]: [(0, -2, _rounded_position(second_positions[5]))],
    }


EXPECTED_CONSTANT_DIAGONAL_CHAIN_GRAPH = _single_track_chain_graph(
    CONSTANT_DIAGONAL_TRACKS[0]
)
EXPECTED_CONSTANT_DIAGONAL_REJECT_GRAPH = _single_track_unlinked_graph(
    CONSTANT_DIAGONAL_TRACKS[0]
)
EXPECTED_TURNING_CHAIN_GRAPH = _single_track_delayed_chain_graph(TURNING_TRACKS[0])
EXPECTED_TURNING_REJECT_GRAPH = _single_track_unlinked_graph(TURNING_TRACKS[0])
EXPECTED_JUMP_FRESH_START_GRAPH = _switch_graph(JUMP_SWITCH_TRACKS, inherited=False)
EXPECTED_JUMP_FALSE_LINK_GRAPH = _switch_graph(JUMP_SWITCH_TRACKS, inherited=True)
EXPECTED_BRANCH_FRESH_START_GRAPH = _switch_graph(BRANCH_SWITCH_TRACKS, inherited=False)
EXPECTED_BRANCH_FALSE_LINK_GRAPH = _switch_graph(BRANCH_SWITCH_TRACKS, inherited=True)
EXPECTED_SINGLE_TRACK_LINK_STATS = {frame: (1, 0) for frame in FRAMES[:-1]}
EXPECTED_SINGLE_TRACK_REJECT_STATS = {frame: (0, 1) for frame in FRAMES[:-1]}
EXPECTED_TURNING_DELAYED_LINK_STATS = {
    FRAMES[0]: (0, 1),
    FRAMES[1]: (1, 0),
    FRAMES[2]: (1, 0),
    FRAMES[3]: (1, 0),
    FRAMES[4]: (1, 0),
}


def _to_native_volume_par(vpar):
    """Convert Python volume parameters into native VolumeParams."""
    if NativeVolumeParams is None:
        raise RuntimeError("optv VolumeParams is not available")

    native_vpar = NativeVolumeParams()
    native_vpar.set_X_lay(list(vpar.x_lay))
    native_vpar.set_Zmin_lay(list(vpar.z_min_lay))
    native_vpar.set_Zmax_lay(list(vpar.z_max_lay))
    native_vpar.set_cn(vpar.cn)
    native_vpar.set_cnx(vpar.cnx)
    native_vpar.set_cny(vpar.cny)
    native_vpar.set_csumg(vpar.csumg)
    native_vpar.set_eps0(vpar.eps0)
    native_vpar.set_corrmin(vpar.corrmin)
    return native_vpar


def _synthetic_track_positions():
    """Return deterministic per-frame 3D positions for two tracks."""
    return {
        0: [np.array([0.00 + 0.01 * index, 0.0, 0.0]) for index in range(len(FRAMES))],
        1: [np.array([0.02 + 0.01 * index, 0.01, 0.0]) for index in range(len(FRAMES))],
    }


def _write_sequence_range(sequence_path: Path) -> None:
    lines = sequence_path.read_text(encoding="utf-8").splitlines()
    lines[-2] = str(FRAMES[0])
    lines[-1] = str(FRAMES[-1])
    sequence_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_tracking_step_stats(log: str) -> dict[int, tuple[int, int]]:
    return {
        int(frame): (int(links), int(lost))
        for frame, links, lost in re.findall(
            r"step:\s*(\d+),.*?links:\s*(-?\d+),\s*lost:\s*(-?\d+)",
            log,
            re.DOTALL,
        )
    }


def _write_synthetic_tracking_fixture(
    workdir: Path,
    tracks: dict[int, list[np.ndarray | None]] | None = None,
    *,
    add_distractor: bool = True,
) -> list[Calibration]:
    """Write a clean deterministic tracking fixture into a temporary workspace."""
    source = Path(__file__).resolve().parents[1] / "testing_folder"
    if not source.exists():
        raise FileNotFoundError(f"Missing test fixture directory: {source}")
    shutil.copytree(source, workdir)
    os.chdir(workdir)

    calibrations = [
        Calibration().from_file(
            Path(f"cal/sym_cam{cam + 1}.tif.ori"),
            Path("cal/cam1.tif.addpar"),
        )
        for cam in range(3)
    ]

    os.chdir(workdir / "track")
    Path("res").mkdir()
    Path("newpart").mkdir(exist_ok=True)
    _write_sequence_range(Path("parameters/sequence_newpart.par"))

    cpar = ControlPar(3).from_file(Path("parameters/control_newpart.par"))
    tracks = tracks or _synthetic_track_positions()

    for frame_index, frame_num in enumerate(FRAMES):
        frame = Frame(3, 16)
        cam_targets: list[list[Target]] = [[] for _ in range(3)]
        correspond_indices = {
            track_id: [CORRES_NONE] * 4
            for track_id, positions in tracks.items()
            if positions[frame_index] is not None
        }

        for track_id, positions in tracks.items():
            pos = positions[frame_index]
            if pos is None:
                continue
            projections = [
                image_coordinates(np.array([pos]), cal, cpar.mm)[0]
                for cal in calibrations
            ]
            for cam, xy in enumerate(projections):
                px, py = metric_to_pixel(xy[0], xy[1], cpar)
                cam_targets[cam].append(
                    Target(
                        pnr=len(cam_targets[cam]),
                        x=float(px),
                        y=float(py),
                        n=10,
                        nx=3,
                        ny=3,
                        sumg=100,
                        tnr=track_id,
                    )
                )

        # Keep one unmatched distractor near the search area in a single camera.
        if add_distractor:
            cam_targets[0].append(
                Target(
                    pnr=99,
                    x=960.0,
                    y=540.0 + frame_index,
                    n=8,
                    nx=2,
                    ny=2,
                    sumg=50,
                    tnr=CORRES_NONE,
                )
            )

        for cam in range(3):
            cam_targets[cam].sort(key=lambda target: target.y)
            write_targets(
                cam_targets[cam],
                len(cam_targets[cam]),
                f"newpart/cam{cam + 1}.%05d",
                frame_num,
            )
            for idx, target in enumerate(cam_targets[cam]):
                if target.tnr in correspond_indices:
                    correspond_indices[target.tnr][cam] = idx

        active_track_ids = sorted(correspond_indices)
        frame.num_parts = len(active_track_ids)
        frame.correspond = np.recarray((frame.num_parts,), dtype=frame.correspond.dtype)
        frame.path_info = [Pathinfo() for _ in range(frame.num_parts)]

        for row_index, track_id in enumerate(active_track_ids):
            frame.correspond[row_index].nr = track_id + 1
            frame.correspond[row_index].p = np.array(
                correspond_indices[track_id],
                dtype=np.int32,
            )
            position = tracks[track_id][frame_index]
            assert position is not None
            frame.path_info[row_index].x = position

        write_path_frame(
            frame.correspond,
            frame.path_info,
            frame.num_parts,
            "res/particles",
            "res/linkage",
            "res/whatever",
            frame_num,
        )

    return calibrations


def _snapshot_tracking_outputs() -> dict[
    int, list[tuple[int, int, tuple[int, int, int, int], tuple[float, float, float]]]
]:
    """Read the written tracking outputs into a stable frame-by-frame snapshot."""
    snapshots: dict[
        int,
        list[tuple[int, int, tuple[int, int, int, int], tuple[float, float, float]]],
    ] = {}
    for frame in FRAMES:
        correspond, path_info = read_path_frame(
            "res/particles",
            "res/linkage",
            "res/whatever",
            frame,
        )
        snapshots[frame] = [
            (
                int(path.prev_frame),
                int(path.next_frame),
                cast(
                    tuple[int, int, int, int],
                    tuple(int(value) for value in correspond[idx].p),
                ),
                cast(
                    tuple[float, float, float],
                    tuple(round(float(value), 5) for value in path.x),
                ),
            )
            for idx, path in enumerate(path_info)
        ]
    return snapshots


def _run_python_tracking_snapshot(
    mode: str,
    bounds: dict[str, float],
    *,
    tracks: dict[int, list[np.ndarray | None]] | None = None,
    add_distractor: bool = True,
) -> TrackingSnapshotResult:
    """Run the Python tracker in either compiled or patched-Python mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workdir = Path(tmp_dir) / "testing_folder"
        original_cwd = Path.cwd()
        try:
            calibrations = _write_synthetic_tracking_fixture(
                workdir,
                tracks=tracks,
                add_distractor=add_distractor,
            )
            run = tr_new(
                Path("parameters/sequence_newpart.par"),
                Path("parameters/track.par"),
                Path("parameters/criteria.par"),
                Path("parameters/control_newpart.par"),
                4,
                16,
                "res/particles",
                "res/linkage",
                "res/whatever",
                calibrations,
                0.1,
            )
            run.tpar = run.tpar._replace(add=0, **bounds)

            with ExitStack() as stack:
                if mode == "python":
                    stack.enter_context(
                        patch.object(
                            track,
                            "search_volume_center_moving",
                            track.search_volume_center_moving.py_func,
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            track,
                            "pos3d_in_bounds",
                            track.pos3d_in_bounds.py_func,
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            track,
                            "angle_acc",
                            track.angle_acc.py_func,
                        )
                    )

                def run_tracker() -> None:
                    track_forward_start(run)
                    for step in range(run.seq_par.first, run.seq_par.last):
                        trackcorr_c_loop(run, step)
                    trackcorr_c_finish(run, run.seq_par.last)

                log_buffer = io.StringIO()
                with redirect_stdout(log_buffer):
                    run_tracker()
                log = log_buffer.getvalue()
            return TrackingSnapshotResult(
                snapshot=_snapshot_tracking_outputs(),
                log=log,
                step_stats=_parse_tracking_step_stats(log),
            )
        finally:
            os.chdir(original_cwd)


def _run_native_tracking_snapshot(
    bounds: dict[str, float],
    *,
    tracks: dict[int, list[np.ndarray | None]] | None = None,
    add_distractor: bool = True,
):
    """Run the native optv tracker on the same synthetic fixture."""
    if not HAS_NATIVE_TRACKING or NativeTracker is None or NativeTrackingParams is None:
        raise RuntimeError("optv Tracker is not available")

    with tempfile.TemporaryDirectory() as tmp_dir:
        workdir = Path(tmp_dir) / "testing_folder"
        original_cwd = Path.cwd()
        try:
            calibrations = _write_synthetic_tracking_fixture(
                workdir,
                tracks=tracks,
                add_distractor=add_distractor,
            )
            cpar = ControlPar(3).from_file(Path("parameters/control_newpart.par"))
            cpar.img_base_name = [f"newpart/cam{cam + 1}." for cam in range(3)]
            vpar = read_volume_par(Path("parameters/criteria.par"))

            native_tpar = NativeTrackingParams()
            native_tpar.read_track_par("parameters/track.par")
            native_tpar.set_add(0)
            native_tpar.set_dacc(bounds["dacc"])
            native_tpar.set_dangle(bounds["dangle"])
            native_tpar.set_dvxmin(bounds["dvxmin"])
            native_tpar.set_dvxmax(bounds["dvxmax"])
            native_tpar.set_dvymin(bounds["dvymin"])
            native_tpar.set_dvymax(bounds["dvymax"])
            native_tpar.set_dvzmin(bounds["dvzmin"])
            native_tpar.set_dvzmax(bounds["dvzmax"])

            native_spar = NativeSequenceParams(num_cams=3)
            native_spar.read_sequence_par("parameters/sequence_newpart.par", 3)
            native_spar.set_first(FRAMES[0])
            native_spar.set_last(FRAMES[-1])
            for cam in range(3):
                native_spar.set_img_base_name(cam, f"newpart/cam{cam + 1}.")

            tracker = NativeTracker(
                to_native_control_par(cpar),
                _to_native_volume_par(vpar),
                native_tpar,
                native_spar,
                [to_native_calibration(cal) for cal in calibrations],
                naming={
                    "corres": "res/particles",
                    "linkage": "res/linkage",
                    "prio": "res/whatever",
                },
                flatten_tol=0.1,
            )
            tracker.full_forward()
            return TrackingSnapshotResult(
                snapshot=_snapshot_tracking_outputs(),
                log="",
                step_stats={},
            )
        finally:
            os.chdir(original_cwd)


class TestTrackingGroundTruth(unittest.TestCase):
    """Ground-truth tracking comparisons across Python and native backends."""

    def assert_python_modes_match_graph(
        self,
        *,
        tracks,
        bounds,
        expected_graph,
        expected_step_stats=None,
        add_distractor=False,
    ):
        compiled_snapshot = _run_python_tracking_snapshot(
            "python+numba",
            bounds,
            tracks=tracks,
            add_distractor=add_distractor,
        )
        python_snapshot = _run_python_tracking_snapshot(
            "python",
            bounds,
            tracks=tracks,
            add_distractor=add_distractor,
        )

        self.assertEqual(_snapshot_graph(compiled_snapshot.snapshot), expected_graph)
        self.assertEqual(_snapshot_graph(python_snapshot.snapshot), expected_graph)
        self.assertEqual(compiled_snapshot.snapshot, python_snapshot.snapshot)
        if expected_step_stats is not None:
            self.assertEqual(compiled_snapshot.step_stats, expected_step_stats)
            self.assertEqual(python_snapshot.step_stats, expected_step_stats)
        self.assertNotRegex(compiled_snapshot.log, BAD_FILE_LOOKUP_PATTERN)
        self.assertNotRegex(python_snapshot.log, BAD_FILE_LOOKUP_PATTERN)

    def test_tracking_exact_link_graph_matches_ground_truth_across_backends(self):
        """Match the exact permissive link graph across Python, Numba, and optv."""
        expected = EXPECTED_PERMISSIVE_SNAPSHOT
        compiled_snapshot = _run_python_tracking_snapshot(
            "python+numba", PERMISSIVE_BOUNDS
        )
        python_snapshot = _run_python_tracking_snapshot("python", PERMISSIVE_BOUNDS)

        self.assertEqual(compiled_snapshot.snapshot, expected)
        self.assertEqual(python_snapshot.snapshot, expected)
        self.assertEqual(compiled_snapshot.snapshot, python_snapshot.snapshot)
        self.assertEqual(compiled_snapshot.step_stats, EXPECTED_PERMISSIVE_STEP_STATS)
        self.assertEqual(python_snapshot.step_stats, EXPECTED_PERMISSIVE_STEP_STATS)
        self.assertNotRegex(compiled_snapshot.log, BAD_FILE_LOOKUP_PATTERN)
        self.assertNotRegex(python_snapshot.log, BAD_FILE_LOOKUP_PATTERN)

        if HAS_OPTV and HAS_NATIVE_TRACKING:
            native_snapshot = _run_native_tracking_snapshot(PERMISSIVE_BOUNDS)
            self.assertEqual(native_snapshot.snapshot, expected)
            self.assertEqual(native_snapshot.snapshot, compiled_snapshot.snapshot)

    def test_tracking_strict_bounds_reject_links_by_design(self):
        """Reject links consistently across backends when the velocity window excludes the motion."""
        expected = EXPECTED_STRICT_SNAPSHOT
        compiled_snapshot = _run_python_tracking_snapshot("python+numba", STRICT_BOUNDS)
        python_snapshot = _run_python_tracking_snapshot("python", STRICT_BOUNDS)

        self.assertEqual(compiled_snapshot.snapshot, expected)
        self.assertEqual(python_snapshot.snapshot, expected)
        self.assertEqual(compiled_snapshot.snapshot, python_snapshot.snapshot)
        self.assertEqual(compiled_snapshot.step_stats, EXPECTED_STRICT_STEP_STATS)
        self.assertEqual(python_snapshot.step_stats, EXPECTED_STRICT_STEP_STATS)
        self.assertNotRegex(compiled_snapshot.log, BAD_FILE_LOOKUP_PATTERN)
        self.assertNotRegex(python_snapshot.log, BAD_FILE_LOOKUP_PATTERN)

    def test_tracking_velocity_window_has_valid_range_between_reject_and_false_link(
        self,
    ):
        """A symmetric 3D velocity window must be wide enough for truth but narrow enough to block jumps."""
        self.assert_python_modes_match_graph(
            tracks=CONSTANT_DIAGONAL_TRACKS,
            bounds=_make_bounds(dv_limit=0.005, dacc=0.2, dangle=100.0),
            expected_graph=EXPECTED_CONSTANT_DIAGONAL_REJECT_GRAPH,
            expected_step_stats=EXPECTED_SINGLE_TRACK_REJECT_STATS,
        )
        self.assert_python_modes_match_graph(
            tracks=CONSTANT_DIAGONAL_TRACKS,
            bounds=_make_bounds(dv_limit=0.015, dacc=0.2, dangle=100.0),
            expected_graph=EXPECTED_CONSTANT_DIAGONAL_CHAIN_GRAPH,
            expected_step_stats=EXPECTED_SINGLE_TRACK_LINK_STATS,
        )
        self.assertFalse(
            _candidate_passes_tracking_gate(
                np.array([0.01, 0.01, 0.01]),
                np.array([0.02, 0.02, 0.02]),
                np.array([0.05, 0.05, 0.04]),
                _make_bounds(dv_limit=0.015, dacc=0.2, dangle=100.0),
            )
        )
        self.assertTrue(
            _candidate_passes_tracking_gate(
                np.array([0.01, 0.01, 0.01]),
                np.array([0.02, 0.02, 0.02]),
                np.array([0.05, 0.05, 0.04]),
                _make_bounds(dv_limit=0.05, dacc=0.2, dangle=100.0),
            )
        )

    def test_tracking_dacc_has_valid_range_between_reject_and_false_link(self):
        """Acceleration bounds should admit smooth curvature but still reject identity-switch jumps."""
        self.assert_python_modes_match_graph(
            tracks=TURNING_TRACKS,
            bounds=_make_bounds(dv_limit=0.02, dacc=0.002, dangle=40.0),
            expected_graph=EXPECTED_TURNING_REJECT_GRAPH,
            expected_step_stats=EXPECTED_SINGLE_TRACK_REJECT_STATS,
        )
        self.assert_python_modes_match_graph(
            tracks=TURNING_TRACKS,
            bounds=_make_bounds(dv_limit=0.02, dacc=0.02, dangle=40.0),
            expected_graph=EXPECTED_TURNING_CHAIN_GRAPH,
            expected_step_stats=EXPECTED_TURNING_DELAYED_LINK_STATS,
        )
        self.assertFalse(
            _candidate_passes_tracking_gate(
                np.array([0.01, 0.01, 0.01]),
                np.array([0.02, 0.02, 0.02]),
                np.array([0.05, 0.05, 0.04]),
                _make_bounds(dv_limit=0.05, dacc=0.02, dangle=100.0),
            )
        )
        self.assertTrue(
            _candidate_passes_tracking_gate(
                np.array([0.01, 0.01, 0.01]),
                np.array([0.02, 0.02, 0.02]),
                np.array([0.05, 0.05, 0.04]),
                _make_bounds(dv_limit=0.05, dacc=0.06, dangle=100.0),
            )
        )

    def test_tracking_dangle_has_valid_range_between_reject_and_false_link(self):
        """Angular bounds should admit smooth turning but still reject a sharp branch switch."""
        self.assert_python_modes_match_graph(
            tracks=TURNING_TRACKS,
            bounds=_make_bounds(dv_limit=0.02, dacc=0.02, dangle=10.0),
            expected_graph=EXPECTED_TURNING_REJECT_GRAPH,
            expected_step_stats=EXPECTED_SINGLE_TRACK_REJECT_STATS,
        )
        self.assert_python_modes_match_graph(
            tracks=TURNING_TRACKS,
            bounds=_make_bounds(dv_limit=0.02, dacc=0.02, dangle=30.0),
            expected_graph=EXPECTED_TURNING_CHAIN_GRAPH,
            expected_step_stats=EXPECTED_TURNING_DELAYED_LINK_STATS,
        )
        self.assertFalse(
            _candidate_passes_tracking_gate(
                np.array([0.01, 0.0, 0.001]),
                np.array([0.018, 0.006, 0.002]),
                np.array([0.024, 0.022, 0.004]),
                _make_bounds(dv_limit=0.02, dacc=0.02, dangle=30.0),
            )
        )
        self.assertTrue(
            _candidate_passes_tracking_gate(
                np.array([0.01, 0.0, 0.001]),
                np.array([0.018, 0.006, 0.002]),
                np.array([0.024, 0.022, 0.004]),
                _make_bounds(dv_limit=0.02, dacc=0.02, dangle=40.0),
            )
        )


if __name__ == "__main__":
    unittest.main()
