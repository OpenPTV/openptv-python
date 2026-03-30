import pytest
from pathlib import Path
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture to set up test data directory"""
    test_dir = Path(__file__).parent.parent / "testing_folder"
    if not test_dir.exists():
        pytest.skip(f"Test data directory {test_dir} not found")
    return test_dir


@pytest.fixture(scope="session", autouse=True)
def clean_test_environment(test_data_dir):
    """Clean up test environment before and after tests"""
    # Clean up any existing test results anywhere under the shared fixture tree.
    for results_dir in test_data_dir.rglob("res"):
        if results_dir.is_dir():
            shutil.rmtree(results_dir)

    # Create a fresh top-level results directory for tests that expect one.
    (test_data_dir / "res").mkdir(exist_ok=True)

    yield

    # Cleanup after tests
    for results_dir in test_data_dir.rglob("res"):
        if results_dir.is_dir():
            shutil.rmtree(results_dir)


@pytest.fixture
def copy_test_case(test_data_dir, tmp_path):
    """Copy one named test case from the shared fixture tree into a temp directory."""

    def _copy(case_name: str):
        source = test_data_dir / case_name
        if not source.exists():
            pytest.skip(f"Test case directory {source} not found")

        destination = tmp_path / case_name
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return destination

    return _copy


def pytest_runtest_setup(item):
    if 'qt' in item.keywords:
        try:
            import PySide6  # or PySide6, depending on your package
        except ImportError:
            pytest.skip("Skipping Qt-dependent test: Qt not available")
