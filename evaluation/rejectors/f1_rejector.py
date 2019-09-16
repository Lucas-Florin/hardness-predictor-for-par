"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import torch
from . import QuantileRejector
import evaluation.metrics as metrics


class F1Rejector(QuantileRejector):
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
        diffs = list()
        possible_attribute_thresholds = list()
        sorted_idxs = hp_scores.argsort(0)
        fn = np.logical_and(labels == 1, label_predictions == 0) # Number of false negatives.
        fp = np.logical_and(labels == 0, label_predictions == 1)  # Number of false positives.
        for i in range(len(quantiles)):
            num_reject = int(num_datapoints * quantiles[i]) + 1

            thresholds = hp_scores[sorted_idxs[-num_reject, :], np.arange(num_attributes)]
            possible_attribute_thresholds.append(thresholds)
            ignore = hp_scores > thresholds
            select = np.logical_not(ignore)
            fn_sel = np.logical_and(select, fn).sum(0)  # Number of false negatives.
            fp_sel = np.logical_and(select, fp).sum(0)  # Number of false positives.
            diff = np.absolute(fp_sel - fn_sel).flatten()  # The difference between fp and fn -> minimize

            diffs.append(diff)
        diffs = np.array(diffs)
        possible_attribute_thresholds = np.array(possible_attribute_thresholds)
        self.attribute_thresholds = possible_attribute_thresholds[diffs.argmin(0), np.arange(num_attributes)]
        self.print_percentage_rejected(hp_scores)
