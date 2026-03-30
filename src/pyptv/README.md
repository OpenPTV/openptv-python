# PyPTV source tree

This directory is part of the unified repository documented in the top-level [README.md](../README.md).

The GUI and batch tools use the shared engine policy from `openptv_python`:

- default engine: `optv`
- override for debugging/testing: `python`
- the selected engine is stored in experiment YAML as `engine`

The canonical version now lives in [src/openptv_python/version.py](../openptv_python/version.py), and the GUI reads from that shared source.

For setup, usage, engine selection, and version bumping, use the repository root README.
