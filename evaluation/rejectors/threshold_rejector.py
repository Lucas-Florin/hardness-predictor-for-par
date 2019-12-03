"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import torch
from .base_rejector import BaseRejector
from .quantile_rejector import QuantileRejector


class ThresholdRejector(QuantileRejector):
    """
    This rejector rejects samples based on a given hardness score threshold while not overstepping the maximum
    rejection quantile.
    """

    def __init__(self, hp_score_threshold, max_rejection_quantile):
        super().__init__(max_rejection_quantile)
        assert 0 <= hp_score_threshold <= 1
        self.hp_score_threshold = hp_score_threshold

    def update_thresholds(self, labels, label_predictions, hp_scores, sorted_scores=None, verbose=True):
        super().update_thresholds(labels, label_predictions, hp_scores)
        ideal_thresholds = np.full((1, hp_scores.shape[1]), self.hp_score_threshold)
        # Check if ideal thresholds would reject too many samples.
        self.attribute_thresholds = np.where(
            self.attribute_thresholds > ideal_thresholds, self.attribute_thresholds, ideal_thresholds)
        self.print_percentage_rejected(hp_scores, verbose)



