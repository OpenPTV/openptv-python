import openptv_python


def test_version() -> None:
    assert openptv_python.__version__ != "999"
