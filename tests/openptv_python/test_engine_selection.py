import numpy as np

from openptv_python import _native_compat as compat
from openptv_python import image_processing


def test_force_python_uses_python_fallback(monkeypatch):
    """Forcing Python should bypass the native preprocess path."""

    compat.set_engine("python", warn_once=False)

    sentinel = object()

    def fake_prepare_image(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(image_processing, "prepare_image", fake_prepare_image)

    result = image_processing.preprocess_image(
        np.zeros((4, 4), dtype=np.uint8),
        0,
        None,
        1,
    )

    assert result is sentinel
    assert compat.get_active_engine() == "python"
    assert not compat.should_use_native("preprocess_image")


def test_optv_preference_falls_back_when_optv_is_missing(monkeypatch):
    """If optv is unavailable, the active engine should fall back to Python."""

    monkeypatch.setattr(compat, "HAS_OPTV", False)
    monkeypatch.setattr(compat, "HAS_NATIVE_PREPROCESS", False)

    compat.set_engine("optv", warn_once=False)

    assert compat.get_active_engine() == "python"
    assert compat.get_engine_preference() == "optv"
    assert "unavailable" in compat.get_engine_reason().lower()


def teardown_module():
    compat.set_engine("optv", warn_once=False)