"""
@author Lucas Florin
@contact lucas.florin@iosb.fraunhofer.de
"""

import numpy as np
import torch


class BaseCalibrator():
    """
    The base class for all calibrators. A calibrator takes in an array of float values between 0 and 1 and returns
    another array of values between 0 and 1 (typically probabilities).
    TODO: explain better.
    """

    def __init__(self):
        pass

    def __call__(self, probs):
        """
        Calibrate.
        :param probs: An array of doubles to be calibrated.
        :return: An array of the same type as probs, but with the values calibrated.
        """
        raise NotImplementedError

