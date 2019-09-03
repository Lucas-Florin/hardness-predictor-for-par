"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import torch
from .base_rejector import BaseRejector


class MeanAccuracyRejector(BaseRejector):
    """
    This rejector tries to optimize for the mean accuracy of each attribute while not overstepping the maximum
    rejection quantile.
    """

    def __call__(self, hp_scores):
        raise NotImplementedError

