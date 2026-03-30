"""Backend compatibility layer for PyPTV.

This module exposes the shared openptv_python API and the active engine choice
behind the legacy names expected by the GUI and batch layers.
"""

from __future__ import annotations

from typing import Any

import openptv_python
from openptv_python._native_compat import (
    HAS_OPTV,
    get_active_engine,
    get_engine_preference,
    get_engine_reason,
    get_engine_status,
    set_engine,
    should_use_native,
)

BACKEND: str = "openptv_python"
BACKEND_MODULE: Any = openptv_python


# =============================================================================
# Helper functions for converting between naming conventions
# =============================================================================


def _to_camel_case(snake_str: str) -> str:
    """Convert snake_case to CamelCase."""
    return "".join(word.capitalize() for word in snake_str.split("_"))


def _to_optv_params_name(name: str) -> str:
    """Convert openptv_python naming (ControlPar) to legacy naming."""
    if name.endswith("Par"):
        return name[:-3] + "Params"
    return name


def _to_openptv_name(name: str) -> str:
    """Convert legacy naming to openptv_python naming."""
    if name.endswith("Params"):
        return name[:-6] + "Par"
    return name


# =============================================================================
# Module imports with compatibility wrapping
# =============================================================================

from openptv_python.parameters import MultimediaPar

if HAS_OPTV:
    from optv.calibration import Calibration
    from optv.parameters import ControlParams
    from optv.parameters import SequenceParams
    from optv.parameters import TargetParams
    from optv.parameters import TrackingParams
    from optv.parameters import VolumeParams
    from optv.correspondences import MatchedCoords
    from optv.correspondences import correspondences as py_correspondences
    from optv.epipolar import epipolar_curve
    from optv.image_processing import preprocess_image
    from optv.imgcoord import image_coordinates
    from optv.orientation import (
        external_calibration,
        full_calibration,
        match_detection_to_ref,
        multi_cam_point_positions,
        point_positions,
    )
    from optv.segmentation import target_recognition
    from optv.tracker import Tracker, default_naming
    from optv.tracking_framebuf import Frame, Target, TargetArray
    from optv.transforms import (
        convert_arr_metric_to_pixel,
        convert_arr_pixel_to_metric,
    )
    correspondences = py_correspondences
else:
    from openptv_python.calibration import Calibration
    from openptv_python.parameters import ControlPar as ControlParams
    from openptv_python.parameters import SequencePar as SequenceParams
    from openptv_python.parameters import TargetPar as TargetParams
    from openptv_python.parameters import TrackPar as TrackingParams
    from openptv_python.parameters import VolumePar as VolumeParams
    from openptv_python.epi import epipolar_curve
    from openptv_python.image_processing import preprocess_image
    from openptv_python.imgcoord import image_coordinates, img_coord
    from openptv_python.orientation import (
        external_calibration,
        full_calibration,
        match_detection_to_ref,
        multi_cam_point_positions,
        point_positions,
    )
    from openptv_python.segmentation import target_recognition
    from openptv_python.tracking_frame_buf import Frame, Target, TargetArray
    from openptv_python.track import Tracker, default_naming
    from openptv_python.trafo import arr_pixel_to_metric as convert_arr_pixel_to_metric
    from openptv_python.trafo import arr_metric_to_pixel as convert_arr_metric_to_pixel
    from openptv_python.correspondences import (
        MatchedCoords,
        py_correspondences,
        correspondences,
    )
from openptv_python.tracking_frame_buf import sort_target_y
from openptv_python.imgcoord import img_coord


# =============================================================================
# Backend info
# =============================================================================


def get_backend() -> str:
    """Return the currently active execution engine."""
    return get_active_engine()


def get_engine() -> str:
    """Return the currently selected engine preference."""
    return get_engine_preference()


def get_backend_module() -> Any:
    """Return the backend module being used."""
    return BACKEND_MODULE


def get_backend_reason() -> str:
    """Return a human-readable explanation of the active engine."""
    return get_engine_reason()


def get_backend_status() -> str:
    """Return a compact backend status string."""
    return get_engine_status()


# =============================================================================
# Parameter compatibility helpers
# =============================================================================


def create_control_params(num_cams: int, **kwargs) -> ControlParams:
    """Create ControlParams with backend-appropriate initialization."""
    cpar = ControlParams(num_cams)
    for key, value in kwargs.items():
        if key == "mm" and hasattr(cpar, "get_multimedia_params"):
            mm = cpar.get_multimedia_params()
            if mm is not None and isinstance(value, dict):
                for mm_key, mm_value in value.items():
                    setter = getattr(mm, f"set_{mm_key}", None)
                    if callable(setter):
                        setter(mm_value)
                    elif hasattr(mm, mm_key):
                        setattr(mm, mm_key, mm_value)
            continue
        if hasattr(cpar, key):
            setattr(cpar, key, value)
    return cpar


def create_sequence_params(num_cams: int, **kwargs) -> SequenceParams:
    """Create SequenceParams with backend-appropriate initialization."""
    try:
        spar = SequenceParams(num_cams=num_cams)
    except TypeError:
        spar = SequenceParams()
        if hasattr(spar, "set_img_base_name"):
            spar.set_img_base_name(["" for _ in range(num_cams)])

    for key, value in kwargs.items():
        if hasattr(spar, key):
            setattr(spar, key, value)
    return spar


def create_tracking_params(**kwargs) -> TrackingParams:
    """Create TrackingParams with backend-appropriate initialization."""
    tpar = TrackingParams()

    for key, value in kwargs.items():
        if hasattr(tpar, key):
            setattr(tpar, key, value)
    return tpar


def create_volume_params(**kwargs) -> VolumeParams:
    """Create VolumeParams with backend-appropriate initialization."""
    vpar = VolumeParams()

    for key, value in kwargs.items():
        if hasattr(vpar, key):
            setattr(vpar, key, value)
    return vpar


def create_target_params(**kwargs) -> TargetParams:
    """Create TargetParams with backend-appropriate initialization."""
    tpar = TargetParams()

    for key, value in kwargs.items():
        if hasattr(tpar, key):
            setattr(tpar, key, value)
    return tpar


# =============================================================================
# Export all symbols
# =============================================================================

__all__ = [
    # Backend info
    "get_backend",
    "get_engine",
    "get_backend_module",
    "get_backend_reason",
    "get_backend_status",
    "BACKEND",
    "BACKEND_MODULE",
    "set_engine",
    "should_use_native",
    # Classes
    "Calibration",
    "ControlParams",
    "SequenceParams",
    "TrackingParams",
    "TargetParams",
    "VolumeParams",
    "MultimediaPar",
    "TargetArray",
    "Target",
    "Frame",
    "Tracker",
    "MatchedCoords",
    "py_correspondences",
    "correspondences",
    # Functions
    "preprocess_image",
    "target_recognition",
    "correspondences",
    "point_positions",
    "multi_cam_point_positions",
    "default_naming",
    "sort_target_y",
    "convert_arr_pixel_to_metric",
    "convert_arr_metric_to_pixel",
    "image_coordinates",
    "img_coord",
    "epipolar_curve",
    # Factory functions
    "create_control_params",
    "create_sequence_params",
    "create_tracking_params",
    "create_volume_params",
    "create_target_params",
]
