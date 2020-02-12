"""
@author Lucas Florin
@contact lucas.florin@iosb.fraunhofer.de
"""

import numpy as np
import torch
from .base_calibrator import BaseCalibrator


class NoneCalibrator(BaseCalibrator):
    """
    Placeholder for inaction. This calibrator makes no changes to the data.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, probs):
        """
        Calibrate.
        :param probs: An array of doubles to be calibrated.
        :return: An array of the same type as probs, but with the values calibrated.
        """
        return probs

