"""Stress benchmarks comparing native and non-native backends."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np

import openptv_python.correspondences as correspondences
import openptv_python.image_processing as image_processing
import openptv_python.orientation as orientation
import openptv_python.segmentation as segmentation
from openptv_python._native_compat import (
    HAS_NATIVE_PREPROCESS,
    HAS_NATIVE_SEGMENTATION,
    HAS_OPTV,
)
from openptv_python._native_convert import to_native_calibration, to_native_control_par
from openptv_python.calibration import Calibration
from openptv_python.epi import Coord2d_dtype
from openptv_python.imgcoord import image_coordinates, img_coord
from openptv_python.parameters import (
    ControlPar,
    TargetPar,
    VolumePar,
    read_control_par,
    read_sequence_par,
    read_volume_par,
)
from openptv_python.track import (
    track_forward_start,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv_python.tracking_frame_buf import Target
from openptv_python.tracking_run import tr_new
from openptv_python.trafo import dist_to_flat, metric_to_pixel, pixel_to_metric

try:
    from optv.orientation import (
        multi_cam_point_positions as native_multi_cam_point_positions,
    )
    from optv.orientation import point_positions as native_point_positions
    from optv.parameters import VolumeParams as NativeVolumeParams

    HAS_NATIVE_RECONSTRUCTION = True
except ImportError:
    native_multi_cam_point_positions = None
    native_point_positions = None
    NativeVolumeParams = None
    HAS_NATIVE_RECONSTRUCTION = False

try:
    from optv.correspondences import (
        MatchedCoords as NativeMatchedCoords,
    )
    from optv.correspondences import (
        correspondences as native_correspondences,
    )
    from optv.tracking_framebuf import TargetArray as NativeTargetArray

    HAS_NATIVE_STEREOMATCHING = True
except ImportError:
    NativeMatchedCoords = None
    NativeTargetArray = None
    native_correspondences = None
    HAS_NATIVE_STEREOMATCHING = False

try:
    from optv.parameters import (
        SequenceParams as NativeSequenceParams,
    )
    from optv.parameters import (
        TrackingParams as NativeTrackingParams,
    )
    from optv.tracker import Tracker as NativeTracker

    HAS_NATIVE_SEQUENCE = True
    HAS_NATIVE_TRACKING = NativeVolumeParams is not None
except ImportError:
    NativeSequenceParams = None
    NativeTrackingParams = None
    NativeTracker = None
    HAS_NATIVE_SEQUENCE = False
    HAS_NATIVE_TRACKING = False


def _env_flag_enabled(name: str) -> bool:
    """Interpret common truthy environment variable values."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


RUN_STRESS_BENCHMARKS = not _env_flag_enabled("OPENPTV_SKIP_STRESS_BENCHMARKS")


def _benchmark(function, *, warmups: int = 1, runs: int = 3):
    """Run a callable repeatedly and return the final result and timings."""
    result = None
    for _ in range(warmups):
        result = function()

    timings = []
    for _ in range(runs):
        start = perf_counter()
        result = function()
        timings.append(perf_counter() - start)

    return result, timings


def _timing_summary(label: str, timings: list[float]) -> str:
    """Return a compact timing summary for benchmark output."""
    return (
        f"{label}: median={median(timings):.6f}s "
        f"min={min(timings):.6f}s max={max(timings):.6f}s"
    )


def _serialize_targets(
    targets,
) -> list[tuple[int, float, float, int, int, int, int, int]]:
    """Convert target objects into stable tuples for comparisons."""
    return [
        (
            int(target.pnr),
            round(float(target.x), 9),
            round(float(target.y), 9),
            int(target.n),
            int(target.nx),
            int(target.ny),
            int(target.sumg),
            int(target.tnr),
        )
        for target in targets
    ]


def _build_segmentation_stress_image(
    width: int = 512,
    height: int = 512,
    spacing: int = 24,
) -> np.ndarray:
    """Create a deterministic image with many isolated synthetic targets."""
    image = np.zeros((height, width), dtype=np.uint8)
    patch = np.array(
        [
            [0, 220, 230, 220, 0],
            [220, 240, 248, 240, 220],
            [230, 248, 255, 248, 230],
            [220, 240, 248, 240, 220],
            [0, 220, 230, 220, 0],
        ],
        dtype=np.uint8,
    )

    for center_y in range(12, height - 12, spacing):
        for center_x in range(12, width - 12, spacing):
            image[center_y - 2 : center_y + 3, center_x - 2 : center_x + 3] = patch

    return image


def _build_reconstruction_stress_case(
    num_points: int = 4096,
) -> tuple[np.ndarray, np.ndarray, ControlPar, list[Calibration]]:
    """Build a deterministic multi-camera reconstruction workload."""
    rng = np.random.default_rng(20260307)
    points = np.empty((num_points, 3), dtype=np.float64)
    points[:, 0] = rng.uniform(-25.0, 25.0, size=num_points)
    points[:, 1] = rng.uniform(20.0, 65.0, size=num_points)
    points[:, 2] = rng.uniform(-15.0, 15.0, size=num_points)

    cpar = ControlPar(4).from_file(
        Path("tests/testing_folder/control_parameters/control.par")
    )
    cpar.mm.set_n1(1.0)
    cpar.mm.set_layers([1.0], [1.0])
    cpar.mm.set_n3(1.0)

    add_file = Path("tests/testing_folder/calibration/cam1.tif.addpar")
    calibs = [
        Calibration().from_file(
            ori_file=Path(f"tests/testing_folder/calibration/sym_cam{cam_num}.tif.ori"),
            add_file=add_file,
        )
        for cam_num in range(1, 5)
    ]

    projections = [image_coordinates(points, cal, cpar.mm) for cal in calibs]
    targets = np.asarray(projections, dtype=np.float64).transpose(1, 0, 2)
    return points, targets, cpar, calibs


def _build_multilayer_reconstruction_stress_case(
    num_points: int = 4096,
) -> tuple[np.ndarray, np.ndarray, ControlPar, list[Calibration]]:
    """Build a deterministic multi-layer multi-camera reconstruction workload."""
    points, _targets, cpar, calibs = _build_reconstruction_stress_case(num_points)
    cpar.mm.set_n1(1.0)
    cpar.mm.set_layers([1.49, 1.10], [5.0, 10.0])
    cpar.mm.set_n3(1.33)

    projections = [image_coordinates(points, cal, cpar.mm) for cal in calibs]
    targets = np.asarray(projections, dtype=np.float64).transpose(1, 0, 2)
    return points, targets, cpar, calibs


def _python_point_positions_reference(
    targets: np.ndarray,
    mm_par,
    cals: list[Calibration],
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct points with the original Python per-target loop."""
    num_targets = targets.shape[0]
    points = np.empty((num_targets, 3), dtype=np.float64)
    rcm = np.empty(num_targets, dtype=np.float64)

    for pt in range(num_targets):
        rcm[pt], points[pt] = orientation.point_position(
            targets[pt],
            len(cals),
            mm_par,
            cals,
        )

    return points, rcm


def _build_stereomatching_stress_case(
    grid_width: int = 8,
    grid_height: int = 8,
    spacing: float = 5.0,
):
    """Build a deterministic multi-camera correspondence workload."""
    cpar = read_control_par(Path("tests/testing_folder/parameters/ptv.par"))
    vpar = read_volume_par(Path("tests/testing_folder/parameters/criteria.par"))
    cpar.mm.n2[0] = 1.0001
    cpar.mm.n3 = 1.0001

    calibs = [
        Calibration().from_file(
            ori_file=Path(f"tests/testing_folder/cal/sym_cam{cam_num}.tif.ori"),
            add_file=Path("tests/testing_folder/cal/cam1.tif.addpar"),
        )
        for cam_num in range(1, cpar.num_cams + 1)
    ]

    img_pts: list[list[Target]] = []
    for cam in range(cpar.num_cams):
        cam_targets = [Target() for _ in range(grid_width * grid_height)]
        for row in range(grid_height):
            for col in range(grid_width):
                pnr = row * grid_width + col
                if cam % 2:
                    pnr = grid_width * grid_height - 1 - pnr

                pos3d = np.array([col * spacing, row * spacing, 0.0], dtype=np.float64)
                x, y = img_coord(pos3d, calibs[cam], cpar.mm)
                x, y = metric_to_pixel(x, y, cpar)

                cam_targets[pnr] = Target(
                    pnr=pnr,
                    x=float(x),
                    y=float(y),
                    n=25,
                    nx=5,
                    ny=5,
                    sumg=10,
                    tnr=-1,
                )

        img_pts.append(cam_targets)

    corrected = np.recarray(
        (cpar.num_cams, len(img_pts[0])),
        dtype=Coord2d_dtype,
    )
    for cam in range(cpar.num_cams):
        for part, targ in enumerate(img_pts[cam]):
            x, y = pixel_to_metric(targ.x, targ.y, cpar)
            x, y = dist_to_flat(x, y, calibs[cam], 0.0001)
            corrected[cam][part].pnr = targ.pnr
            corrected[cam][part].x = x
            corrected[cam][part].y = y

        corrected[cam].sort(order="x")

    return img_pts, corrected, calibs, vpar, cpar


def _to_native_target_array(targets: list[Target]):
    """Convert Python targets into a native TargetArray."""
    if NativeTargetArray is None:
        raise RuntimeError("optv TargetArray is not available")

    native_targets = NativeTargetArray(len(targets))
    for index, target in enumerate(targets):
        native_target = native_targets[index]
        native_target.set_pnr(int(target.pnr))
        native_target.set_tnr(int(target.tnr))
        native_target.set_pos((float(target.x), float(target.y)))
        native_target.set_pixel_counts(int(target.n), int(target.nx), int(target.ny))
        native_target.set_sum_grey_value(int(target.sumg))

    return native_targets


def _to_native_volume_par(vpar: VolumePar):
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


def _normalize_correspondence_output(sorted_pos, sorted_corresp):
    """Sort correspondence outputs into a stable order for comparisons."""
    normalized_pos = []
    normalized_corresp = []

    for positions, identifiers in zip(sorted_pos, sorted_corresp):
        if identifiers.shape[1] == 0:
            normalized_pos.append(positions)
            normalized_corresp.append(identifiers)
            continue

        sortable_ids = np.where(identifiers < 0, np.iinfo(np.int64).max, identifiers)
        order = np.lexsort(sortable_ids[::-1])
        normalized_pos.append(positions[:, order, :])
        normalized_corresp.append(identifiers[:, order])

    return normalized_pos, normalized_corresp


@contextmanager
def _working_directory(path: Path):
    """Temporarily change the current working directory."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _read_track_calibrations(workdir: Path, num_cams: int) -> list[Calibration]:
    """Read all tracking calibrations from a copied tracking fixture."""
    return [
        Calibration().from_file(
            ori_file=workdir / f"cal/cam{cam_num}.tif.ori",
            add_file=workdir / f"cal/cam{cam_num}.tif.addpar",
        )
        for cam_num in range(1, num_cams + 1)
    ]


def _snapshot_text_outputs(root: Path) -> dict[str, tuple[int, str]]:
    """Summarize benchmark output files into a stable comparison dictionary."""
    return {
        str(path.relative_to(root)): (
            len(path.read_text(encoding="utf-8").splitlines()),
            path.read_text(encoding="utf-8").splitlines()[0]
            if path.read_text(encoding="utf-8").splitlines()
            else "",
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run_python_tracking_fixture() -> dict[str, tuple[int, str]]:
    """Execute the Python tracking loop in a temporary fixture directory."""
    source = Path("tests/testing_folder/track")
    with tempfile.TemporaryDirectory() as tmp_dir:
        workdir = Path(tmp_dir) / "track"
        shutil.copytree(source, workdir)
        shutil.copytree(workdir / "img_orig", workdir / "img")
        output_dir = workdir / "py_res"
        shutil.copytree(workdir / "res_orig", output_dir)

        with _working_directory(workdir):
            cpar = read_control_par(Path("parameters/ptv.par"))
            calibs = _read_track_calibrations(workdir, cpar.num_cams)
            run = tr_new(
                Path("parameters/sequence.par"),
                Path("parameters/track.par"),
                Path("parameters/criteria.par"),
                Path("parameters/ptv.par"),
                4,
                20000,
                "py_res/rt_is",
                "py_res/ptv_is",
                "py_res/added",
                calibs,
                10000.0,
            )
            run.seq_par.first = 10240
            run.seq_par.last = 10250
            run.tpar = run.tpar._replace(add=1)

            track_forward_start(run)
            trackcorr_c_loop(run, run.seq_par.first)
            for step in range(run.seq_par.first + 1, run.seq_par.last):
                trackcorr_c_loop(run, step)
            trackcorr_c_finish(run, run.seq_par.last)

        return _snapshot_text_outputs(output_dir)


def _run_native_tracking_fixture() -> dict[str, tuple[int, str]]:
    """Execute the native tracker in a temporary fixture directory."""
    if (
        NativeTracker is None
        or NativeSequenceParams is None
        or NativeTrackingParams is None
    ):
        raise RuntimeError("optv Tracker is not available")

    source = Path("tests/testing_folder/track")
    with tempfile.TemporaryDirectory() as tmp_dir:
        workdir = Path(tmp_dir) / "track"
        shutil.copytree(source, workdir)
        shutil.copytree(workdir / "img_orig", workdir / "img")
        output_dir = workdir / "native_res"
        shutil.copytree(workdir / "res_orig", output_dir)

        with _working_directory(workdir):
            py_cpar = read_control_par(Path("parameters/ptv.par"))
            py_vpar = read_volume_par(Path("parameters/criteria.par"))
            calibs = _read_track_calibrations(workdir, py_cpar.num_cams)

            native_cpar = to_native_control_par(py_cpar)
            native_vpar = _to_native_volume_par(py_vpar)
            native_tpar = NativeTrackingParams()
            native_tpar.read_track_par("parameters/track.par")
            native_tpar.set_add(1)

            native_spar = NativeSequenceParams(num_cams=py_cpar.num_cams)
            native_spar.read_sequence_par("parameters/sequence.par", py_cpar.num_cams)
            native_spar.set_first(10240)
            native_spar.set_last(10250)
            for cam, img_base_name in enumerate(py_cpar.img_base_name):
                native_spar.set_img_base_name(cam, img_base_name.replace("%05d", ""))

            tracker = NativeTracker(
                native_cpar,
                native_vpar,
                native_tpar,
                native_spar,
                [to_native_calibration(cal) for cal in calibs],
                naming={
                    "corres": "native_res/rt_is",
                    "linkage": "native_res/ptv_is",
                    "prio": "native_res/added",
                },
                flatten_tol=10000.0,
            )
            tracker.full_forward()

        return _snapshot_text_outputs(output_dir)


@unittest.skip("stress benchmarks are disabled while focusing on core test failures")
class TestNativeStressPerformance(unittest.TestCase):
    """Stress tests comparing native and non-native runtime paths."""

    @unittest.skipUnless(
        HAS_NATIVE_SEQUENCE,
        "optv native SequenceParams is not available",
    )
    def test_sequence_parameter_read_stress_timing(self):
        """Compare sequence parameter loading timing against native bindings."""
        sequence_file = Path("tests/testing_folder/sequence_parameters/sequence.par")

        def python_path():
            return read_sequence_par(sequence_file, 4)

        def native_path():
            assert NativeSequenceParams is not None
            seq = NativeSequenceParams(num_cams=4)
            seq.read_sequence_par(str(sequence_file), 4)
            return seq

        python_result, python_timings = _benchmark(python_path, runs=5)
        native_result, native_timings = _benchmark(native_path, runs=5)

        self.assertEqual(python_result.first, native_result.get_first())
        self.assertEqual(python_result.last, native_result.get_last())
        self.assertEqual(
            python_result.img_base_name,
            [native_result.get_img_base_name(cam) for cam in range(4)],
        )

        speedup = median(python_timings) / median(native_timings)
        print(
            "sequence stress benchmark: "
            f"{_timing_summary('python', python_timings)}; "
            f"{_timing_summary('native', native_timings)}; "
            f"speedup={speedup:.2f}x"
        )

    @unittest.skipUnless(
        HAS_NATIVE_PREPROCESS,
        "optv native preprocess_image is not available",
    )
    def test_preprocess_image_stress_timing(self):
        """Compare native preprocessing timing against the Python path."""
        rng = np.random.default_rng(20260307)
        image = rng.integers(0, 256, size=(1024, 1024), dtype=np.uint8)

        cpar = ControlPar(1)
        cpar.set_image_size((image.shape[1], image.shape[0]))

        def python_path():
            return image_processing.prepare_image(
                image,
                dim_lp=1,
                filter_hp=0,
                filter_file="",
            )

        def native_path():
            return image_processing.preprocess_image(
                image,
                filter_hp=0,
                cpar=cpar,
                dim_lp=1,
            )

        python_result, python_timings = _benchmark(python_path)
        with patch.object(
            image_processing,
            "native_preprocess_image",
            wraps=image_processing.native_preprocess_image,
        ) as native_call:
            native_result, native_timings = _benchmark(native_path)

        np.testing.assert_array_equal(native_result, python_result)
        self.assertGreaterEqual(native_call.call_count, 3)

        speedup = median(python_timings) / median(native_timings)
        print(
            "preprocess stress benchmark: "
            f"{_timing_summary('python', python_timings)}; "
            f"{_timing_summary('native', native_timings)}; "
            f"speedup={speedup:.2f}x"
        )

    @unittest.skipUnless(
        HAS_NATIVE_SEGMENTATION,
        "optv native target recognition is not available",
    )
    def test_target_recognition_stress_timing(self):
        """Compare native target recognition timing against Python+Numba."""
        image = _build_segmentation_stress_image()

        cpar = ControlPar(1)
        cpar.set_image_size((image.shape[1], image.shape[0]))
        tpar = TargetPar(
            gvthresh=[200],
            discont=20,
            nnmin=1,
            nnmax=100,
            sumg_min=1,
            nxmin=1,
            nxmax=10,
            nymin=1,
            nymax=10,
        )

        def python_numba_path():
            with patch.object(segmentation, "HAS_NATIVE_SEGMENTATION", False):
                return segmentation.target_recognition(image, tpar, 0, cpar)

        def native_path():
            return segmentation.target_recognition(image, tpar, 0, cpar)

        python_targets, python_timings = _benchmark(python_numba_path)
        with patch.object(
            segmentation,
            "native_target_recognition",
            wraps=segmentation.native_target_recognition,
        ) as native_call:
            native_targets, native_timings = _benchmark(native_path)

        self.assertEqual(
            _serialize_targets(native_targets),
            _serialize_targets(python_targets),
        )
        self.assertGreaterEqual(native_call.call_count, 3)

        speedup = median(python_timings) / median(native_timings)
        print(
            "segmentation stress benchmark: "
            f"{_timing_summary('python+numba', python_timings)}; "
            f"{_timing_summary('native', native_timings)}; "
            f"speedup={speedup:.2f}x"
        )

    @unittest.skipUnless(
        HAS_NATIVE_STEREOMATCHING,
        "optv native correspondences is not available",
    )
    def test_stereomatching_stress_timing(self):
        """Compare native correspondences timing against the Python path."""
        img_pts, corrected, calibs, vpar, cpar = _build_stereomatching_stress_case()
        native_cpar = to_native_control_par(cpar)
        native_vpar = _to_native_volume_par(vpar)
        native_cals = [to_native_calibration(cal) for cal in calibs]
        native_img_pts = [_to_native_target_array(targets) for targets in img_pts]
        native_flat_coords = [
            NativeMatchedCoords(
                native_img_pts[cam], native_cpar, native_cals[cam], 0.0001
            )
            for cam in range(cpar.num_cams)
        ]

        def python_path():
            return correspondences.py_correspondences(
                img_pts, corrected, calibs, vpar, cpar
            )

        def native_path():
            assert native_correspondences is not None
            return native_correspondences(
                native_img_pts,
                native_flat_coords,
                native_cals,
                native_vpar,
                native_cpar,
            )

        python_result, python_timings = _benchmark(python_path)
        native_result, native_timings = _benchmark(native_path)
        python_pos, python_corresp, python_num_targs = python_result
        native_pos, native_corresp, native_num_targs = native_result
        python_pos, python_corresp = _normalize_correspondence_output(
            python_pos,
            python_corresp,
        )
        native_pos, native_corresp = _normalize_correspondence_output(
            native_pos,
            native_corresp,
        )

        self.assertEqual(python_num_targs, native_num_targs)
        self.assertEqual(len(python_corresp), len(native_corresp))
        for python_positions, native_positions in zip(python_pos, native_pos):
            np.testing.assert_allclose(native_positions, python_positions, atol=1e-9)
        for python_ids, native_ids in zip(python_corresp, native_corresp):
            np.testing.assert_array_equal(native_ids, python_ids)

        speedup = median(python_timings) / median(native_timings)
        print(
            "stereomatching stress benchmark: "
            f"{_timing_summary('python', python_timings)}; "
            f"{_timing_summary('native', native_timings)}; "
            f"speedup={speedup:.2f}x"
        )

    @unittest.skipUnless(
        HAS_OPTV and HAS_NATIVE_RECONSTRUCTION,
        "optv native point_positions is not available",
    )
    def test_point_reconstruction_stress_timing(self):
        """Compare compiled Python, legacy Python, and native reconstruction."""
        expected_points, targets, cpar, calibs = _build_reconstruction_stress_case()
        native_cpar = to_native_control_par(cpar)
        native_cals = [to_native_calibration(cal) for cal in calibs]
        native_vpar = NativeVolumeParams()
        python_vpar = VolumePar()

        def compiled_python_path():
            return orientation.point_positions(targets, cpar.mm, calibs, python_vpar)

        def legacy_python_path():
            return _python_point_positions_reference(targets, cpar.mm, calibs)

        def native_path():
            assert native_point_positions is not None
            return native_point_positions(
                targets, native_cpar, native_cals, native_vpar
            )

        compiled_python_result, compiled_python_timings = _benchmark(
            compiled_python_path
        )
        legacy_python_result, legacy_python_timings = _benchmark(legacy_python_path)
        native_result, native_timings = _benchmark(native_path)
        python_points, python_rcm = compiled_python_result
        legacy_points, legacy_rcm = legacy_python_result
        native_points, native_rcm = native_result

        np.testing.assert_allclose(python_points, expected_points, atol=1e-6)
        np.testing.assert_allclose(legacy_points, expected_points, atol=1e-6)
        np.testing.assert_allclose(native_points, expected_points, atol=1e-6)
        np.testing.assert_allclose(legacy_points, python_points, atol=1e-9)
        np.testing.assert_allclose(legacy_rcm, python_rcm, atol=1e-9)
        np.testing.assert_allclose(native_points, python_points, atol=1e-9)
        np.testing.assert_allclose(native_rcm, python_rcm, atol=1e-9)

        compiled_vs_legacy = median(legacy_python_timings) / median(
            compiled_python_timings
        )
        compiled_vs_native = median(compiled_python_timings) / median(native_timings)
        print(
            "reconstruction stress benchmark: "
            f"{_timing_summary('compiled-python', compiled_python_timings)}; "
            f"{_timing_summary('legacy-python', legacy_python_timings)}; "
            f"{_timing_summary('native', native_timings)}; "
            f"compiled-vs-legacy={compiled_vs_legacy:.2f}x; "
            f"compiled-vs-native={compiled_vs_native:.2f}x"
        )

    @unittest.skipUnless(
        HAS_OPTV and HAS_NATIVE_RECONSTRUCTION,
        "optv native point_positions is not available",
    )
    def test_multilayer_point_reconstruction_stress_timing(self):
        """Compare compiled Python, legacy Python, and native multilayer reconstruction."""
        _expected_points, targets, cpar, calibs = (
            _build_multilayer_reconstruction_stress_case()
        )
        native_cpar = to_native_control_par(cpar)
        native_cals = [to_native_calibration(cal) for cal in calibs]

        def compiled_python_path():
            return orientation.multi_cam_point_positions(targets, cpar.mm, calibs)

        def legacy_python_path():
            return _python_point_positions_reference(targets, cpar.mm, calibs)

        def native_path():
            assert native_multi_cam_point_positions is not None
            return native_multi_cam_point_positions(targets, native_cpar, native_cals)

        compiled_python_result, compiled_python_timings = _benchmark(
            compiled_python_path
        )
        legacy_python_result, legacy_python_timings = _benchmark(legacy_python_path)
        native_result, native_timings = _benchmark(native_path)
        python_points, python_rcm = compiled_python_result
        legacy_points, legacy_rcm = legacy_python_result
        native_points, native_rcm = native_result

        np.testing.assert_allclose(legacy_points, python_points, atol=1e-9)
        np.testing.assert_allclose(legacy_rcm, python_rcm, atol=1e-9)
        np.testing.assert_allclose(native_points, python_points, atol=1e-9)
        np.testing.assert_allclose(native_rcm, python_rcm, atol=1e-9)

        compiled_vs_legacy = median(legacy_python_timings) / median(
            compiled_python_timings
        )
        compiled_vs_native = median(compiled_python_timings) / median(native_timings)
        print(
            "multilayer reconstruction stress benchmark: "
            f"{_timing_summary('compiled-python', compiled_python_timings)}; "
            f"{_timing_summary('legacy-python', legacy_python_timings)}; "
            f"{_timing_summary('native', native_timings)}; "
            f"compiled-vs-legacy={compiled_vs_legacy:.2f}x; "
            f"compiled-vs-native={compiled_vs_native:.2f}x"
        )

    @unittest.skipUnless(
        HAS_NATIVE_TRACKING,
        "optv native Tracker is not available",
    )
    def test_tracking_sequence_stress_timing(self):
        """Compare native tracking over a short sequence against Python."""

        def python_path():
            return _run_python_tracking_fixture()

        def native_path():
            return _run_native_tracking_fixture()

        python_outputs, python_timings = _benchmark(python_path, warmups=0, runs=1)
        native_outputs, native_timings = _benchmark(native_path, warmups=0, runs=1)

        self.assertTrue(python_outputs)
        self.assertEqual(native_outputs, python_outputs)

        speedup = median(python_timings) / median(native_timings)
        print(
            "tracking stress benchmark: "
            f"{_timing_summary('python', python_timings)}; "
            f"{_timing_summary('native', native_timings)}; "
            f"speedup={speedup:.2f}x"
        )


if __name__ == "__main__":
    unittest.main()
