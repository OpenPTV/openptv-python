import unittest

from openptv_python.parameters import MultimediaPar


class TestMultimediaPar(unittest.TestCase):
    def test_multimedia_par(self):
        # Test case 1: Create an instance with default values
        multimedia_instance = MultimediaPar()
        self.assertEqual(multimedia_instance.nlay, 1)
        self.assertEqual(multimedia_instance.n1, 1.0)
        self.assertEqual(multimedia_instance.n2, [1.0])
        self.assertEqual(multimedia_instance.d, [1.0])
        self.assertEqual(multimedia_instance.n3, 1.0)

        # Test case 2: Create an instance with custom values
        custom_values = {
            "nlay": 2,
            "n1": 2.5,
            "n2": [2.0, 3.0],
            "d": [0.5, 1.0],
            "n3": 2.0,
        }
        multimedia_instance_custom = MultimediaPar(**custom_values)
        self.assertEqual(multimedia_instance_custom.nlay, 2)
        self.assertEqual(multimedia_instance_custom.n1, 2.5)
        self.assertEqual(multimedia_instance_custom.n2, [2.0, 3.0])
        self.assertEqual(multimedia_instance_custom.d, [0.5, 1.0])
        self.assertEqual(multimedia_instance_custom.n3, 2.0)

        # Test case 3: Ensure that an exception is raised for invalid lengths of n2 and d
        invalid_values = {
            "nlay": 2,
            "n1": 2.5,
            "n2": [2.0, 3.0, 4.0],  # Invalid length
            "d": [0.5, 1.0],
            "n3": 2.0,
        }
        with self.assertRaises(ValueError):
            MultimediaPar(**invalid_values)

    def test_to_dict_from_dict(self):
        # Test case for to_dict and from_dict methods
        data = {
            "nlay": 3,
            "n1": 1.5,
            "n2": [1.0, 2.0, 3.0],
            "d": [0.5, 1.0, 1.5],
            "n3": 1.8,
        }
        multimedia_instance = MultimediaPar(**data)

        # Convert to dictionary and then create a new instance from the dictionary
        data_from_instance = multimedia_instance.to_dict()
        new_instance = MultimediaPar().from_dict(data_from_instance)

        # Ensure the new instance is equal to the original one
        self.assertEqual(multimedia_instance.nlay, new_instance.nlay)
        self.assertEqual(multimedia_instance.n1, new_instance.n1)
        self.assertEqual(multimedia_instance.n2, new_instance.n2)
        self.assertEqual(multimedia_instance.d, new_instance.d)
        self.assertEqual(multimedia_instance.n3, new_instance.n3)


if __name__ == "__main__":
    unittest.main()
