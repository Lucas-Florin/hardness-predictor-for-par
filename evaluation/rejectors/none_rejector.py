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

    def __call__(self, hp_scores):
        return hp_scores.new_ones(hp_scores.shape)

