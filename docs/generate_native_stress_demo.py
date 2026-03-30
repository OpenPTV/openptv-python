"""Regenerate the native stress benchmark data assets for the docs.

This script reruns tests/test_native_stress_performance.py, parses the timing
summary lines printed by the test module, and writes machine-specific demo
artifacts under docs/_static/.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Structured benchmark summary for one workload."""

    key: str
    label: str
    python_label: str
    python_seconds: float
    native_seconds: float
    speedup: float
    legacy_seconds: float | None = None
    compiled_vs_legacy: float | None = None


LINE_PATTERNS = {
    "sequence": re.compile(
        r"^sequence stress benchmark: python: median=(?P<python>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?speedup=(?P<speedup>[0-9.]+)x"
    ),
    "preprocess": re.compile(
        r"^preprocess stress benchmark: python: median=(?P<python>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?speedup=(?P<speedup>[0-9.]+)x"
    ),
    "segmentation": re.compile(
        r"^segmentation stress benchmark: python\+numba: median=(?P<python>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?speedup=(?P<speedup>[0-9.]+)x"
    ),
    "stereomatching": re.compile(
        r"^stereomatching stress benchmark: python: median=(?P<python>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?speedup=(?P<speedup>[0-9.]+)x"
    ),
    "reconstruction": re.compile(
        r"^reconstruction stress benchmark: compiled-python: median=(?P<python>[0-9.]+)s .*?"
        r"legacy-python: median=(?P<legacy>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?"
        r"compiled-vs-legacy=(?P<compiled_vs_legacy>[0-9.]+)x; "
        r"compiled-vs-native=(?P<speedup>[0-9.]+)x"
    ),
    "multilayer_reconstruction": re.compile(
        r"^multilayer reconstruction stress benchmark: compiled-python: median=(?P<python>[0-9.]+)s .*?"
        r"legacy-python: median=(?P<legacy>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?"
        r"compiled-vs-legacy=(?P<compiled_vs_legacy>[0-9.]+)x; "
        r"compiled-vs-native=(?P<speedup>[0-9.]+)x"
    ),
    "tracking": re.compile(
        r"^tracking stress benchmark: python: median=(?P<python>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?speedup=(?P<speedup>[0-9.]+)x"
    ),
}


WORKLOAD_METADATA = {
    "sequence": ("Sequence params", "python"),
    "preprocess": ("Preprocess image", "python"),
    "segmentation": ("Segmentation", "python+numba"),
    "stereomatching": ("Stereo matching", "python"),
    "reconstruction": ("Reconstruction", "compiled-python"),
    "multilayer_reconstruction": ("Multilayer reconstruction", "compiled-python"),
    "tracking": ("Tracking", "python"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root containing tests/test_native_stress_performance.py",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(__file__).resolve().parent / "_static" / "native-stress-demo.json",
        help="Output path for the parsed benchmark JSON",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=Path(__file__).resolve().parent / "_static" / "native-stress-demo.log",
        help="Output path for the raw benchmark log",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to run pytest",
    )
    return parser.parse_args()


def run_benchmarks(repo_root: Path, python_executable: str) -> str:
    command = [
        python_executable,
        "-m",
        "pytest",
        "-q",
        "-s",
        "tests/test_native_stress_performance.py",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            "stress benchmark run failed with exit code "
            f"{completed.returncode}\n\n{output}"
        )
    return output


def parse_benchmarks(output: str) -> list[BenchmarkResult]:
    result_map: dict[str, BenchmarkResult] = {}
    for raw_line in output.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        line = line.lstrip(".").strip()
        for key, pattern in LINE_PATTERNS.items():
            match = pattern.match(line)
            if match is None:
                continue
            label, python_label = WORKLOAD_METADATA[key]
            groups = match.groupdict()
            result_map[key] = BenchmarkResult(
                key=key,
                label=label,
                python_label=python_label,
                python_seconds=float(groups["python"]),
                native_seconds=float(groups["native"]),
                speedup=float(groups["speedup"]),
                legacy_seconds=(
                    float(groups["legacy"]) if groups.get("legacy") else None
                ),
                compiled_vs_legacy=(
                    float(groups["compiled_vs_legacy"])
                    if groups.get("compiled_vs_legacy")
                    else None
                ),
            )
            break

    expected_order = list(WORKLOAD_METADATA)
    missing = [key for key in expected_order if key not in result_map]
    if missing:
        raise RuntimeError(
            "failed to parse benchmark summaries for: " + ", ".join(missing)
        )
    return [result_map[key] for key in expected_order]


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = run_benchmarks(repo_root, args.python)
    results = parse_benchmarks(output)
    generated_at = datetime.now(timezone.utc)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(
            {
                "generated_at": generated_at.isoformat(),
                "python": args.python,
                "command": [
                    args.python,
                    "-m",
                    "pytest",
                    "-q",
                    "-s",
                    "tests/test_native_stress_performance.py",
                ],
                "results": [asdict(result) for result in results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.log_output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
