"""Utilities for comparing calibration folders camera by camera."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from .calibration import Calibration, read_calibration


@dataclass
class CalibrationDelta:
    """Numeric parameter deltas for one camera calibration pair."""

    camera_key: str
    position_delta: np.ndarray
    angle_delta: np.ndarray
    primary_point_delta: np.ndarray
    glass_delta: np.ndarray
    added_par_delta: np.ndarray


def _camera_key_from_ori_path(ori_path: Path) -> str:
    """Return the camera key shared by the .ori and .addpar file pair."""
    if not ori_path.name.endswith(".ori"):
        raise ValueError(f"Expected a .ori file, got {ori_path}")
    return ori_path.name[: -len(".ori")]


def _discover_calibration_pairs(folder: Path) -> Dict[str, tuple[Path, Path | None]]:
    """Discover calibration files in a folder keyed by camera base name."""
    if not folder.exists():
        raise FileNotFoundError(folder)

    pairs: Dict[str, tuple[Path, Path | None]] = {}
    for ori_path in sorted(folder.glob("*.ori")):
        if "(copy)" in ori_path.stem:
            continue
        key = _camera_key_from_ori_path(ori_path)
        addpar_path = folder / f"{key}.addpar"
        pairs[key] = (ori_path, addpar_path if addpar_path.exists() else None)

    if not pairs:
        raise ValueError(f"No calibration .ori files found in {folder}")

    return pairs


def _load_calibration_pair(pair: tuple[Path, Path | None]) -> Calibration:
    """Load one calibration pair from disk."""
    ori_path, addpar_path = pair
    return read_calibration(ori_path, addpar_path)


def compare_calibration_folders(
    reference_dir: Path | str, candidate_dir: Path | str
) -> Dict[str, CalibrationDelta]:
    """Compare two calibration folders and return numeric deltas per camera."""
    reference_dir = Path(reference_dir)
    candidate_dir = Path(candidate_dir)

    reference_pairs = _discover_calibration_pairs(reference_dir)
    candidate_pairs = _discover_calibration_pairs(candidate_dir)

    if set(reference_pairs) != set(candidate_pairs):
        missing_from_candidate = sorted(set(reference_pairs) - set(candidate_pairs))
        missing_from_reference = sorted(set(candidate_pairs) - set(reference_pairs))
        raise ValueError(
            "Calibration folders contain different camera sets: "
            f"missing_from_candidate={missing_from_candidate}, "
            f"missing_from_reference={missing_from_reference}"
        )

    deltas: Dict[str, CalibrationDelta] = {}
    for camera_key in sorted(reference_pairs):
        reference_cal = _load_calibration_pair(reference_pairs[camera_key])
        candidate_cal = _load_calibration_pair(candidate_pairs[camera_key])

        deltas[camera_key] = CalibrationDelta(
            camera_key=camera_key,
            position_delta=candidate_cal.get_pos() - reference_cal.get_pos(),
            angle_delta=candidate_cal.get_angles() - reference_cal.get_angles(),
            primary_point_delta=(
                candidate_cal.get_primary_point() - reference_cal.get_primary_point()
            ),
            glass_delta=candidate_cal.glass_par - reference_cal.glass_par,
            added_par_delta=candidate_cal.added_par - reference_cal.added_par,
        )

    return deltas


def format_calibration_comparison(
    deltas: Dict[str, CalibrationDelta],
    reference_dir: Path | str | None = None,
    candidate_dir: Path | str | None = None,
) -> str:
    """Format camera-by-camera calibration deltas as readable text."""
    lines: list[str] = []
    if reference_dir is not None and candidate_dir is not None:
        lines.append(f"Reference: {Path(reference_dir)}")
        lines.append(f"Candidate: {Path(candidate_dir)}")

    for camera_key in sorted(deltas):
        delta = deltas[camera_key]
        lines.append(f"{camera_key}:")
        lines.append(
            "  position_delta: "
            + " ".join(f"{value:+.9f}" for value in delta.position_delta)
        )
        lines.append(
            "  angle_delta: " + " ".join(f"{value:+.9f}" for value in delta.angle_delta)
        )
        lines.append(
            "  primary_point_delta: "
            + " ".join(f"{value:+.9f}" for value in delta.primary_point_delta)
        )
        lines.append(
            "  glass_delta: " + " ".join(f"{value:+.9f}" for value in delta.glass_delta)
        )
        lines.append(
            "  addpar_delta: "
            + " ".join(f"{value:+.9f}" for value in delta.added_par_delta)
        )

    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    """Command-line entry point for comparing calibration folders."""
    parser = argparse.ArgumentParser(
        description="Compare two calibration folders camera by camera."
    )
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    deltas = compare_calibration_folders(args.reference_dir, args.candidate_dir)
    print(
        format_calibration_comparison(
            deltas,
            reference_dir=args.reference_dir,
            candidate_dir=args.candidate_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
