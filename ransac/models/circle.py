import ransac.core as ransac
import math
import numpy as np


class Circle(ransac.Model):  # The input is a 2D point. The output is the distance from the circle
    def __init__(self, center=None, radius=None, zero_threshold=1e-6):
        super().__init__()
        self.center = center
        self.radius = radius
        self.zero_threshold = zero_threshold

    def Evaluate(self, x):  # Take an input variable x and return an output variable y
        distance_from_center = math.sqrt((x[0] - self.center[0])**2 + (x[1] - self.center[1])**2)
        return abs(distance_from_center - self.radius)  # Points located at a distance radius from the center will evaluate to 0.

    def Distance(self, y1, y2):  # Compute the distance between two output variables. Must return a float.
        return abs(y1 - y2)

    def Create(self, xy_tuples, **kwargs):  # Create a model with the given (x, y) observations
        if len(xy_tuples) < self.MinimumNumberOfDataToDefineModel():
            raise ValueError("Circle.Create(): len(xy_tuples) ({}) < {}".format(len(xy_tuples), self.MinimumNumberOfDataToDefineModel()))

        # (x - cx)^2 + (y - cy)^2 = r^2
        # (-2x) cx + (-2y) cy + (cx^2 + cy^2 - r^2) = -x^2 - y^2
        # | -2x   -2y    1 | | cx    |   | -x^2 - y^2 |
        # | ...            | | cy    | = | ...        |
        # | ...            | | gamma |   | ...        |
        # where gamma: = cx^2 + cy^2 - r^2

        A = np.zeros((len(xy_tuples), 3), dtype=float)
        b = np.zeros((len(xy_tuples), 1), dtype=float)
        for row in range(len(xy_tuples)):
            x0 = xy_tuples[row][0][0]
            x1 = xy_tuples[row][0][1]
            A[row, 0] = -2 * x0
            A[row, 1] = -2 * x1
            A[row, 2] = 1
            b[row, 0] = -x0**2 -x1**2
        z, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        self.center = (z[0, 0], z[1, 0])
        self.radius = math.sqrt(self.center[0]**2 + self.center[1]**2 - z[2, 0])


    def MinimumNumberOfDataToDefineModel(self):  # The minimum number or (x, y) observations to define the model
        return 3

