"""
This rejector rejects nothing.

@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import torch
from .base_rejector import BaseRejector


class NoneRejector(BaseRejector):
    """
    This rejector does not reject any samples. It selects all samples.
    """

    def update_thresholds(self, labels, label_predictions, hp_scores):
        print("Rejecting nothing. ")
        self.attribute_thresholds = None  # redundant
        self.print_percentage_rejected(hp_scores)

    def load_thresholds(self, attribtute_thresholds):
        self.attribute_thresholds = None  # redundant
