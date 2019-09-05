"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import torch


class BaseRejector():
    """
    The base class for all rejectors. A rejector rejects some attributes of some of the samples passed based on the
    hardness score. Each rejector implements a specific rejection strategy.
    """

    def __init__(self):
        self.attribute_thresholds = None

    def __call__(self, hp_scores):
        """
        Call the rejector.
        :param hp_scores: The hardness scores in a torch tensor.
        :return: an array of the same type as hp_scores. A 1 means the sample is selected, a 0 means the sample is
            rejected.
        """
        print(self.attribute_thresholds.shape)
        print(hp_scores.shape)
        return hp_scores < self.attribute_thresholds

    def update_thresholds(self, labels, label_predictions, hp_scores):
        """
        Update the rejection thresholds based on the training dataset.
        :param labels: the ground truth labels
        :param label_predictions: the binary predictions for the labels.
        :param hp_scores: the hardness scores.
        """
        raise NotImplementedError

    def load_thresholds(self, attribtute_thresholds):
        """
        Set the thresholds to predefined values.
        :param attribtute_thresholds: the thresholds loaded from a past session.
        """
        self.attribute_thresholds = attribtute_thresholds

    def is_initialized(self):
        """
        Have the thresholds been initialized?
        :return: A boolean value.
        """
        return self.attribute_thresholds is not None

    def print_percentage_rejected(self, hp_scores):
        print("Rejecting the {:.2%} hardest of training examples. ".format((np.logical_not(self(hp_scores))).mean()))

