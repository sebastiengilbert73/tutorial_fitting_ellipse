import ransac.core as ransac
import math
import numpy as np


class Polynomial(ransac.Model):  # The input is a float number; the output is the polynomial evaluation
    def __init__(self, coefficients=None, zero_threshold=1e-6):
        super().__init__()
        self.degree = 0
        if coefficients is None:
            self.coefficients = [0]
        else:
            self.degree = len(coefficients) - 1
            self.coefficients = coefficients  # [c0, c1, c2, ..., cn]
        self.zero_threshold = zero_threshold

    def Evaluate(self, x):  # Take an input variable x and return an output variable y
        if self.coefficients is None:
            raise ValueError("Polynomial.Evaluate(): self.coefficients is None")
        elif len(self.coefficients) == 0:
            raise ValueError("Polynomial.Evaluate(): self.coefficients is empty")
        degree = len(self.coefficients) - 1
        sum = self.coefficients[0]
        for exponent in range(1, degree + 1):
            coef = self.coefficients[exponent]
            sum += coef * x**exponent
        return sum

    def Distance(self, y1, y2):  # Compute the distance between two output variables
        return abs(y1 - y2)

    def MinimumNumberOfDataToDefineModel(self, **kwargs):  # The minimum number or (x, y) observations to define the model
        return kwargs['degree'] + 1

    def Create(self, xy_tuples, **kwargs):  # Create a model with the given (x, y) observations
        self.degree = kwargs['degree']
        if len(xy_tuples) < self.degree + 1:
            raise ValueError("Polynomial.Create(): len(xy_tuples) ({}) < self.degree + 1 ({})".format(len(xy_tuples), self.degree + 1))
        number_of_observations = len(xy_tuples)
        A = np.zeros((number_of_observations, self.degree + 1), dtype=float)
        b = np.zeros((number_of_observations,), dtype=float)
        for observationNdx in range(number_of_observations):
            x = xy_tuples[observationNdx][0]
            y = xy_tuples[observationNdx][1]
            b[observationNdx] = y
            A[observationNdx, 0] = 1
            for col in range(1, self.degree + 1):
                A[observationNdx, col] = x**col

        # Least-square solve
        z, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        self.coefficients = z.tolist()