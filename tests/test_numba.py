from numba import float64, int32
from numba.experimental import jitclass

spec = [
    ("x", int32),  # a simple scalar field
    ("y", float64),  # an array field
]


# Use @jitclass to create the compiled class
@jitclass(spec)  # jitclass decorator
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx**2 + dy**2) ** 0.5


# Create instances of the compiled class
point1 = Point(1, 2.0)
point2 = Point(4, 6.0)

# Call the compiled method
distance = point1.distance(point2)
print(f"Distance between points: {distance}")
