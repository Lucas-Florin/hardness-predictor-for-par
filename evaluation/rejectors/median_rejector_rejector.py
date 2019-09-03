"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import torch
from .base_rejector import BaseRejector


class MedianRejector(BaseRejector):
    """
    This rejector moves the threshold until more right than wrong classifications are rejected while not overstepping
    the maximum rejection quantile.
    """

    def __call__(self, hp_scores):
        raise NotImplementedError

