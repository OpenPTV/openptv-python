# openptv-python

Unified repository for the merged `openptv_python` core library and `pyptv` GUI.

Current version: 0.5.0

## What This Repo Is

This repository exposes one public API and one shared version stream.

- `openptv_python`: the core Python library, with Numba-accelerated kernels and a Python fallback path for debugging and inspection.
- `pyptv`: the GUI and batch layer that calls the same shared API.
- `optv`: the preferred native engine when installed; it is used automatically for supported operations and treated as the fast reference path.

The engine choice is exposed in both GUI and CLI:

- default engine: `optv`
- override for debugging/testing: `python`
- the selected engine is stored in the experiment YAML as `engine`

When `optv` is unavailable, the shared API falls back to the Python/Numba implementation and reports the reason once per session.

## Versioning

The canonical version source is [src/openptv_python/version.py](src/openptv_python/version.py).

To bump the repository version, run:

```bash
python scripts/bump_version.py --patch
```

Use `--minor` or `--major` for larger increments. The script updates the shared version module and `pyproject.toml` together.

## Installation

Python support: `>=3.11,<3.14`

### Recommended: uv

Runtime install:

```bash
uv venv
source .venv/bin/activate
uv sync
```

GUI extras:

```bash
uv sync --extra gui
```

Developer setup:

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

### Alternative: pip

```bash
pip install .
```

GUI extras:

```bash
pip install ".[gui]"
```

Developer extras:

```bash
pip install ".[dev]"
```

## Usage

### Python API

```python
import openptv_python
print(openptv_python.__version__)
```

### GUI

```bash
uv run pyptv --engine optv
```

Use `--engine python` only for debugging, testing, or visualization work where you want to force the fallback backend.

### Batch

Serial batch processing:

```bash
uv run python -m pyptv.pyptv_batch path/to/parameters.yaml 10000 10010 --engine optv
```

Parallel batch processing:

```bash
uv run python -m pyptv.pyptv_batch_parallel path/to/parameters.yaml 10000 10010 4 --engine optv
```

Both batch entrypoints accept `--engine python` for debugging/testing.

## Documentation

Start with [docs/index.md](docs/index.md) for the full documentation tree.

The PyPTV topic guides are in [docs/pyptv/README.md](docs/pyptv/README.md).

## Testing

Run the focused test suites with:

```bash
uv run --with pytest pytest -q tests/openptv_python tests/pyptv
```

The test datasets live in [tests/testing_folder](tests/testing_folder/).

## Repository Layout

- `src/openptv_python/`: shared Python library, engine selector, and canonical version module
- `src/pyptv/`: GUI, batch tools, and compatibility layer
- `docs/`: Sphinx documentation
- `tests/`: library, GUI, and fixture tests

## License

Apache License, Version 2.0.
