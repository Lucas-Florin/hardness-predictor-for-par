"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import torch
from .base_rejector import BaseRejector


class QuantileRejector(BaseRejector):
    """
    This rejector sets the rejection threshold such that the maximum rejection quantile is reached.
    """

    def __init__(self, rejection_quantile):
        super().__init__()
        assert 0 <= rejection_quantile <= 1
        self.rejection_quantile = rejection_quantile
        self.num_datapoints = None
        self.num_attributes = None
        self.num_reject = None
        self.shape = None

    def update_thresholds(self, labels, label_predictions, hp_scores, sorted_scores=None, verbose=True):
        self.num_datapoints = labels.shape[0]
        self.num_attributes = labels.shape[1]
        self.num_reject = int(self.num_datapoints * self.rejection_quantile) + 1
        self.shape = labels.shape
        if sorted_scores is None:
            # TODO: implement on other rejectors
            sorted_scores = np.sort(hp_scores, axis=0)
        self.attribute_thresholds = sorted_scores[-self.num_reject, :]
        self.print_percentage_rejected(hp_scores, verbose)


