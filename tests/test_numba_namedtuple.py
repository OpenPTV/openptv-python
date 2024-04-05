from collections import namedtuple

import numpy as np
from numba import njit

# Define a namedtuple
Point = namedtuple('Point', ['x', 'y'])

# Create a list of points
points = [Point(x, y) for x in range(1000) for y in range(1000)]

# Calculate the distance between each point and the origin
@njit
def distance_to_origin(point):
    """Calculate the distance between a point and the origin."""
    return np.sqrt(point.x**2 + point.y**2)


# Calculate the distance between each point and the origin using the function
distances = np.array([distance_to_origin(point) for point in points])

# Calculate the distance between each point and the origin manually
expected_distances = np.array([np.sqrt(point.x**2 + point.y**2) for point in points])

# Compare the results
np.testing.assert_allclose(distances, expected_distances)
