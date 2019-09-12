"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import torch
from .base_rejector import BaseRejector
from .quantile_rejector import QuantileRejector
from ..metrics import mean_attribute_accuracies


class MeanAccuracyRejector(QuantileRejector):
    """
    This rejector tries to optimize for the mean accuracy of each attribute while not overstepping the maximum
    rejection quantile.
    """

    def __init__(self, max_rejection_quantile):
        super().__init__(max_rejection_quantile)

    def update_thresholds(self, labels, label_predictions, hp_scores):
        num_datapoints = labels.shape[0]
        num_attributes = labels.shape[1]
        quantiles = np.arange(0, self.rejection_quantile, 0.001)
        maccs = list()
        possible_attribute_thresholds = list()
        sorted_idxs = hp_scores.argsort(0)
        for i in range(len(quantiles)):
            num_reject = int(num_datapoints * quantiles[i]) + 1

            thresholds = hp_scores[sorted_idxs[-num_reject, :], np.arange(num_attributes)]
            possible_attribute_thresholds.append(thresholds)
            ignore = hp_scores > thresholds

            macc = mean_attribute_accuracies(label_predictions, labels, ignore)
            maccs.append(macc)
        maccs = np.array(maccs)
        possible_attribute_thresholds = np.array(possible_attribute_thresholds)
        self.attribute_thresholds = possible_attribute_thresholds[maccs.argmax(0), np.arange(num_attributes)]
        self.print_percentage_rejected(hp_scores)
