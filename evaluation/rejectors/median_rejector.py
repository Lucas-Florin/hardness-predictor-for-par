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

    def __init__(self, rejection_quantile):
        super().__init__(rejection_quantile)

    def update_thresholds(self, labels, label_predictions, hp_scores, correctness_ratio_threshold=None):

        num_datapoints = labels.shape[0]
        num_attributes = labels.shape[1]
        fraction_width = 0.01
        quantiles = np.arange(0, self.rejection_quantile, fraction_width)
        correctness_ratios = list()
        possible_attribute_thresholds = list()
        correct_label_predictions = labels == label_predictions
        if correctness_ratio_threshold is None:
            correctness_ratio_threshold = correct_label_predictions.mean(0)
        sorted_idxs = hp_scores.argsort(0)
        for i in range(len(quantiles) - 1):
            num_reject_start = int(num_datapoints * quantiles[i]) + 1
            num_reject_end = int(num_datapoints * quantiles[i + 1]) + 1
            thresholds_start = hp_scores[sorted_idxs[-num_reject_start, :], np.arange(num_attributes)]
            thresholds_end = hp_scores[sorted_idxs[-num_reject_end, :], np.arange(num_attributes)]
            possible_attribute_thresholds.append(thresholds_start)
            select = np.logical_and(thresholds_end < hp_scores, hp_scores <= thresholds_start)
            ratio = (correct_label_predictions * select).mean(0) / fraction_width
            correctness_ratios.append(ratio)
        correctness_ratios = np.array(correctness_ratios)
        possible_attribute_thresholds = np.array(possible_attribute_thresholds)
        threshold_idxs = (correctness_ratios > correctness_ratio_threshold).argmax(0)
        self.attribute_thresholds = possible_attribute_thresholds[threshold_idxs, np.arange(num_attributes)]
        self.print_percentage_rejected(hp_scores)






