import numpy as np
import abc
import random


class Model(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def Evaluate(self, x):  # Take an input variable x and return an output variable y
        pass

    @abc.abstractmethod
    def Distance(self, y1, y2):  # Compute the distance between two output variables. Must return a float.
        pass

    @abc.abstractmethod
    def Create(self, xy_tuples, **kwargs):  # Create a model with the given (x, y) observations
        pass

    @abc.abstractmethod
    def MinimumNumberOfDataToDefineModel(self, **kwargs):  # The minimum number or (x, y) observations to define the model
        pass


class Modeler():
    def __init__(self, model_class, number_of_trials, acceptable_error):
        self.model_class = model_class
        self.number_of_trials = number_of_trials
        self.acceptable_error = acceptable_error

    def ConsensusModel(self, xy_tuples, **kwargs):
        candidate_model = self.model_class()
        final_model = self.model_class()
        if len(xy_tuples) < candidate_model.MinimumNumberOfDataToDefineModel(**kwargs):
            raise ValueError("core.ConsensusModel(): len(xy_tuples) ({}) < candidate_model.MinimumNumberOfDataToDefineModel() ({})".format(
                len(xy_tuples), candidate_model.MinimumNumberOfDataToDefineModel(**kwargs) ))

        highest_sum_of_distance_inverses = -1
        inliers_list = []
        for trial in range(1, self.number_of_trials + 1):
            # Randomly select the minimum number of data from the list
            candidate_xy_list = random.sample(xy_tuples, candidate_model.MinimumNumberOfDataToDefineModel(**kwargs))
            # Create a candidate model
            candidate_model.Create(candidate_xy_list, **kwargs)
            # Find the inliers
            sum_of_distance_inverses = 0
            candidate_inliers_list = []
            for xy in xy_tuples:
                model_prediction = candidate_model.Evaluate(xy[0])
                model_prediction_distance = candidate_model.Distance(model_prediction, xy[1])
                if model_prediction_distance < self.acceptable_error:
                    sum_of_distance_inverses += (1 - model_prediction_distance/self.acceptable_error)  # A number in [0, 1]. The higher, the better.
                    candidate_inliers_list.append(xy)
            # Compare with the champion
            if sum_of_distance_inverses > highest_sum_of_distance_inverses:
                highest_sum_of_distance_inverses = sum_of_distance_inverses
                inliers_list = candidate_inliers_list
        # Compute the final model with the agreeing data
        final_model.Create(inliers_list, **kwargs)
        outliers_list = [xy for xy in xy_tuples if xy not in inliers_list]
        return final_model, inliers_list, outliers_list

