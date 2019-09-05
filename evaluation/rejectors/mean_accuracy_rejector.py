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
        quantiles = np.arange(0, self.max_rejection_quantile, 0.001)
        maccs = np.zeros(quantiles.size, num_attributes)
        possible_attribute_thresholds = np.zeros(quantiles.size, num_attributes)

        for i in range(len(quantiles)):
            num_reject = int(num_datapoints * quantiles[i])
            ignore = np.zeros(labels.shape, dtype="int8")
            sorted_idxs = hp_scores.argsort(0)
            hard_idxs = sorted_idxs[-num_reject:, :]
            possible_attribute_thresholds[i, :] = hp_scores[sorted_idxs[-num_reject, :], np.arange(num_attributes)]
            if num_reject > 0:
                ignore[hard_idxs] = 1
            macc = mean_attribute_accuracies(label_predictions, labels, ignore)
            maccs[i, :] = macc
        self.attribute_thresholds = possible_attribute_thresholds[maccs.argmax(0)]
        self.print_percentage_rejected(hp_scores)
