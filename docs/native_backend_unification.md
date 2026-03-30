# Unified Python API Across Python, Numba, and optv

This page explains the backend model that `openptv-python` is moving toward:
one public Python API for the user, with execution routed internally to pure
Python, Python plus Numba, or native `optv` when available.

The important user-facing contract is simple:

- the script imports `openptv_python`, not `optv`
- the same Python calls should work with and without `optv` installed
- native acceleration is an implementation detail, not a second API surface

## Implementation Checklist

Use this checklist when adding or changing backend routing:

- [ ] Import backend-facing symbols from `pyptv._backend`, not directly from `optv` or `openptv_python`, in pipeline code.
- [ ] Ensure the active engine is selected through `_native_compat.set_engine()` and queried through `should_use_native(...)` or the backend helper functions.
- [ ] When `optv` is installed and selected, use the native module objects and native parameter/calibration classes.
- [ ] When `optv` is missing or disabled, keep the Python/Numba fallback path callable with the same function signatures.
- [ ] Convert Python objects to native objects at backend boundaries only; do not leak backend-specific types into unrelated code paths.
- [ ] Convert native objects back to Python objects only when a downstream caller still expects the Python implementation model.
- [ ] Preserve input and output shapes between backends for detection, correspondences, calibration, reconstruction, and tracking.
- [ ] Keep a fallback path for calibration routines that fail in the native backend, but make sure the fallback sees the same data model as the native path.
- [ ] Add or update tests whenever a backend boundary changes so the native and Python paths stay behaviorally aligned.

The runtime flow is:

| Layer | Responsibility |
| --- | --- |
| User script | Imports `openptv_python` and calls the normal Python API. |
| Public API | Keeps the stable Python function signatures used by scripts and GUI code. |
| Dispatch layer | Chooses between pure Python, Python plus Numba, or native `optv` based on availability and the specific operation. |
| Backend implementation | Executes the selected kernel while preserving the same user-facing behavior. |

In other words, the user writes one Python workflow and the repository decides
internally whether the current operation should run through the fallback Python
path, an accelerated Numba path, or a native `optv` path.

## How the API is unified today

The repository already has the basic adapter structure needed for a transparent
backend model:

- optional native feature detection lives in `openptv_python/_native_compat.py`
- Python-to-native object conversion lives in `openptv_python/_native_convert.py`
- public runtime entry points keep their Python signatures and decide internally
  whether to call Python or native implementations

Today, transparent native delegation is already active for two high-value entry
points:

- `openptv_python.image_processing.preprocess_image()`
- `openptv_python.segmentation.targ_rec()`

## Native optv submodules

When `optv` is installed, the backend can use these submodules directly:

| optv submodule | Used for |
| --- | --- |
| `optv.calibration` | Calibration objects and calibration-file handling |
| `optv.correspondences` | Stereo correspondences and single-camera correspondences |
| `optv.epipolar` | Epipolar-curve generation |
| `optv.image_processing` | High-pass preprocessing |
| `optv.imgcoord` | Image-coordinate conversion helpers |
| `optv.orientation` | Point reconstruction and calibration routines |
| `optv.parameters` | Control, sequence, target, tracking, and volume parameters |
| `optv.segmentation` | Target recognition |
| `optv.tracker` | Tracking orchestration |
| `optv.tracking_framebuf` | Targets, target arrays, frames, and target-file IO |
| `optv.transforms` | Pixel/metric conversion helpers |
| `optv.version` | Package version metadata |

The current backend preference is to route to these native modules whenever the
`optv` engine is selected and the submodule is available, while preserving the
Python compatibility path as the fallback.

The rest of the 3D-PTV pipeline is already callable from Python without native
bindings, and the stress suite shows that several later stages also have native
parity paths available:

- reconstruction via `openptv_python.orientation.point_positions()`
- correspondence search via `openptv_python.correspondences.py_correspondences()`
- tracking orchestration via `openptv_python.tracking_run.TrackingRun`

That split is exactly what we want for batch jobs:

1. one Python script controls the workflow
1. pure Python remains the guaranteed fallback path
1. Numba accelerates selected kernels automatically
1. `optv` can be used underneath the same API when installed

## Dispatch status by stage

The current rollout state is clearer as a status table:

| Stage | Public entry point | Current status | Notes |
| --- | --- | --- | --- |
| Image preprocessing | `openptv_python.image_processing.preprocess_image()` | Auto-delegated today | Same Python signature; dispatches internally through native compatibility and conversion helpers when `optv` is available. |
| Full-frame target recognition | `openptv_python.segmentation.targ_rec()` | Auto-delegated today | Whole-image path routes transparently to native and converts results back into Python `Target` objects. |
| Point reconstruction | `openptv_python.orientation.point_positions()` and `multi_cam_point_positions()` | Parity-tested, not yet transparent | Native parity exists in the stress suite, but routine runtime dispatch still uses the Python implementation by default. |
| Correspondences / stereo matching | `openptv_python.correspondences.py_correspondences()` | Parity-tested, not yet transparent | Benchmarked against `optv` correspondences, but not auto-routed yet. |
| Tracking workflow | `openptv_python.tracking_run.TrackingRun` | Parity-tested, not yet transparent | Python workflow is compared with native `optv` `Tracker`, but dispatch remains separate today. |
| Sequence parameter I/O | sequence parameter loading path | Parity-tested, not yet transparent | Native bindings exist and are benchmarked, but Python remains the default runtime entry point. |

The important distinction is not whether a native implementation exists. It is
whether the normal `openptv_python` runtime path already chooses that native
implementation transparently for the user.

### Contract Rules

- `pyptv._backend` is the only layer that should decide which implementation is active.
- Public helpers must keep one stable signature, regardless of backend.
- Backend-specific object construction must stay inside conversion helpers or backend dispatch code.
- If a native routine expects a native object type, the wrapper must build it before the call.
- If a fallback routine expects Python dataclasses, the wrapper must convert back before the call.
- A failed native path may fall back to Python, but only after restoring or rebuilding the input objects the fallback expects.

## Why `tests/test_native_stress_performance.py` matters

The best executable description of this backend strategy is
`tests/test_native_stress_performance.py`.

That test module does three useful things:

1. it validates parity between Python and native paths for several stages
1. it benchmarks Python, compiled Python, and native implementations side by side
1. it demonstrates which stages already have a viable native provider behind the
   same workflow semantics

The benchmarked workloads in that file currently cover:

- sequence parameter loading
- image preprocessing
- full-frame target recognition
- stereo matching / correspondences
- point reconstruction
- multilayer point reconstruction
- short-sequence tracking

## Demo from this machine

The timings below were generated from a real run on this machine using the docs
helper script:

```bash
python docs/generate_native_stress_demo.py
```

That script reruns:

```bash
python -m pytest -q -s tests/test_native_stress_performance.py
```

and then writes two machine-specific artifacts under `docs/_static/`:

- `native-stress-demo.json`
- `native-stress-demo.log`

Because it reruns the full stress suite, it is expected to take around a couple
of minutes on a normal developer machine.

The reported speedups compare native execution against the routed Python path
used by the test:

- for preprocessing, segmentation, stereomatching, and tracking this is Python
  versus native
- for reconstruction workloads this is compiled Python plus Numba versus native

The parsed machine-specific results are summarized in the next section.

## Measured median timings

```{include} _generated/native-stress-demo.mdinc
```

This is the key architectural message of the page: the repository does not need
two user APIs to get performance. It needs one stable Python API with multiple
internal execution paths.

## What this means for the roadmap

The test results support a practical backend policy:

1. keep Python as the orchestration and scripting surface
1. preserve current public function signatures so scripts and GUI code do not
   care which backend is active
1. keep the pure Python implementation as the reference behavior
1. continue using Numba for the medium-performance path
1. route additional heavy stages to `optv` or future compiled kernels behind
   the same API

In other words, users should be able to write one batch script and get:

- correct execution with no native dependencies installed
- better throughput when Numba is active
- still better throughput when `optv` is installed and the operation has a
  native provider

That is the unifying API story already visible in the codebase and demonstrated
by the stress suite.
