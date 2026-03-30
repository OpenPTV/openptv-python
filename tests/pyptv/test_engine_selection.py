from pathlib import Path
import sys

from pyptv import pyptv_batch, pyptv_batch_parallel


def test_batch_cli_parses_engine_override(monkeypatch, tmp_path):
    """The batch CLI should accept an engine override."""

    yaml_file = tmp_path / "parameters.yaml"
    yaml_file.write_text(
        "\n".join(
            [
                "num_cams: 1",
                "engine: optv",
                "sequence:",
                "  first: 10",
                "  last: 12",
                "  base_name:",
                "    - img/cam1_%04d.tif",
            ]
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyptv_batch.py",
            str(yaml_file),
            "100",
            "120",
            "--engine",
            "python",
            "--mode",
            "tracking",
        ],
    )

    parsed_yaml, first_frame, last_frame, mode, engine = pyptv_batch.parse_command_line_args()

    assert parsed_yaml == Path(yaml_file).resolve()
    assert first_frame == 100
    assert last_frame == 120
    assert mode == "tracking"
    assert engine == "python"


def test_batch_cli_uses_yaml_engine_when_not_overridden(monkeypatch, tmp_path):
    """The batch CLI should respect the engine saved in YAML by default."""

    yaml_file = tmp_path / "parameters.yaml"
    yaml_file.write_text(
        "\n".join(
            [
                "num_cams: 1",
                "engine: python",
                "sequence:",
                "  first: 10",
                "  last: 12",
                "  base_name:",
                "    - img/cam1_%04d.tif",
            ]
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyptv_batch.py",
            str(yaml_file),
            "100",
            "120",
        ],
    )

    _, _, _, mode, engine = pyptv_batch.parse_command_line_args()

    assert mode == "both"
    assert engine == "python"


def test_parallel_batch_cli_uses_yaml_engine_when_not_overridden(monkeypatch, tmp_path):
    """The parallel batch CLI should also respect the YAML engine preference."""

    yaml_file = tmp_path / "parameters.yaml"
    yaml_file.write_text(
        "\n".join(
            [
                "num_cams: 1",
                "engine: python",
                "sequence:",
                "  first: 10",
                "  last: 12",
                "  base_name:",
                "    - img/cam1_%04d.tif",
            ]
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyptv_batch_parallel.py",
            str(yaml_file),
            "100",
            "120",
            "2",
        ],
    )

    _, _, _, _, mode, engine = pyptv_batch_parallel.parse_command_line_args()

    assert mode == "both"
    assert engine == "python"