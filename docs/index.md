# Welcome to openptv_python's documentation!

This is the documentation tree for the unified OpenPTV Python repository.

Start with the root [README.md](../README.md) for installation, versioning, and engine-selection details.

Short version:

- install runtime dependencies with `uv sync`
- install contributor tooling with `uv sync --extra dev`
- use `optv` by default for supported operations
- use `--engine python` in GUI or batch runs when you want to force the fallback backend for debugging/testing
- the shared API automatically falls back to Python/Numba when `optv` is unavailable

```{toctree}
:caption: 'Contents:'
:maxdepth: 2

README.md
bundle_adjustment.md
native_backend_unification.md
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
