import os
import unittest
from pathlib import Path

from openptv_python.parameters import MultiPlanesPar


class TestMultiPlanesParameters(unittest.TestCase):
    """Test the MultiPlanesPar class."""

    def setUp(self):
        """Set up."""
        # Create a temporary file for testing
        self.temp_file = "tests/testing_fodder/parameters/multi_planes.par"

    def test_read_from_file(self):
        """Read from file."""
        instance = MultiPlanesPar().from_file(self.temp_file)
        self.assertEqual(instance.num_planes, 3)
        self.assertEqual(
            instance.filename, ["img/calib_a_cam", "img/calib_b_cam", "img/calib_c_cam"]
        )

    def test_write_to_yaml_and_read_back(self):
        """Write to YAML and read back."""
        instance = MultiPlanesPar().from_file(self.temp_file)

        # Write to YAML
        yaml_file = Path("multi_planes_parameters.yaml")
        instance.to_yaml(yaml_file)

        # Read back from YAML
        read_instance = MultiPlanesPar().from_yaml(yaml_file)

        # Verify equality
        self.assertEqual(instance.to_dict(), read_instance.to_dict())

        os.remove(yaml_file)


if __name__ == "__main__":
    unittest.main()
