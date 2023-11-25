import os
import unittest

from openptv_python.parameters import OrientationPar


class TestOrientationParameters(unittest.TestCase):
    """Test the OrientationPar class."""

    def setUp(self):
        # Create a temporary file for testing
        self.temp_file = 'tests/testing_fodder/parameters/orient.par'


    def test_read_from_file(self):
        """Read from file."""
        instance = OrientationPar.from_file(self.temp_file)
        self.assertEqual(instance.useflag, 0)
        self.assertEqual(instance.ccflag, 0)
        self.assertEqual(instance.xhflag, 0)
        self.assertEqual(instance.yhflag, 0)
        self.assertEqual(instance.k1flag, 0)
        self.assertEqual(instance.k2flag, 0)
        self.assertEqual(instance.k3flag, 0)
        self.assertEqual(instance.p1flag, 0)
        self.assertEqual(instance.p2flag, 0)
        self.assertEqual(instance.scxflag, 0)
        self.assertEqual(instance.sheflag, 0)
        self.assertEqual(instance.interfflag, 0)

    def test_write_to_yaml_and_read_back(self):
        """Write to YAML and read back."""
        instance = OrientationPar.from_file(self.temp_file)

        # Write to YAML
        yaml_file = 'orientation_parameters.yaml'
        instance.to_yaml(yaml_file)

        # Read back from YAML
        read_instance = OrientationPar.from_yaml(yaml_file)

        # Verify equality
        self.assertEqual(instance, read_instance)
        os.remove(yaml_file)

if __name__ == '__main__':
    unittest.main()
