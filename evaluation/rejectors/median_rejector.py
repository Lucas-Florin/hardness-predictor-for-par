"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import torch
from .base_rejector import BaseRejector
from .quantile_rejector import QuantileRejector


class MedianRejector(QuantileRejector):
    """
    This rejector moves the threshold until more right than wrong classifications are rejected while not overstepping
    the maximum rejection quantile.
    """

    def __init__(self, max_rejection_quantile):
        super().__init__(max_rejection_quantile)

    def update_thresholds(self, labels, label_predictions, hp_scores):
        num_datapoints = labels.shape[0]
        num_attributes = labels.shape[1]
        quantiles = np.arange(0, self.max_rejection_quantile, 0.01)
        correctness_ratios = np.zeros(quantiles.size, num_attributes)
        possible_attribute_thresholds = np.zeros(quantiles.size, num_attributes)
        correct_label_predictions = labels == label_predictions

        for i in range(len(quantiles) - 1):
            num_reject_start = int(num_datapoints * quantiles[i])
            num_reject_end = int(num_datapoints * quantiles[i + 1])
            select = np.zeros(labels.shape, dtype="int8")
            sorted_idxs = hp_scores.argsort(0)
            quantile_idxs = sorted_idxs[-num_reject_end : -num_reject_start, :]
            possible_attribute_thresholds[i, :] = hp_scores[sorted_idxs[-num_reject_start, :], np.arange(num_attributes)]
            select[quantile_idxs] = 1

            correctness_ratios[i, :] = correct_label_predictions[select].mean()
        self.attribute_thresholds = possible_attribute_thresholds[(correctness_ratios < 0.5).argmax(0)]
        print("Rejecting " + str(np.logical_not(self(hp_scores)).mean()) + " % of training samples. ")






