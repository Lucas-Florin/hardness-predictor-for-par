"""
@author Lucas Florin
@contact lucas.florin@iosb.fraunhofer.de
"""

import numpy as np
import torch
from .base_calibrator import BaseCalibrator


class LinearCalibrator(BaseCalibrator):
    """
    This calibrator has a piecewise linear calibration function with one non-differentiable part at x=t, t being the
    threshold. All values in [0, t] are linearly mapped to [0, 0.5] and values in (t, 1] are mapped to (0.5, 1].
    """

    def __init__(self, thresholds=None):
        super().__init__()


        self.update_thresholds(thresholds)

    def __call__(self, probs):
        """
        Calibrate.
        :param probs: An array of doubles to be calibrated.
        :return: An array of the same type as probs, but with the values calibrated.
        """
        assert self.is_initialized()
        if torch.is_tensor(probs):
            return self.calibrate_torch(probs)
        else:
            return self.calibrate_np(probs)

    def calibrate_np(self, probs):
        return self.calibrate(probs, self.thresholds_np, np.where)

    def calibrate_torch(self, probs):
        return self.calibrate(probs, self.thresholds_torch, torch.where)

    def calibrate(self, probs, thresholds, where):
        negative_multiplicator = 1 / thresholds / 2
        neg_probs = probs * negative_multiplicator
        positive_multiplicator = 1 / (1 - thresholds) / 2
        pos_probs = (probs - thresholds) * positive_multiplicator + 0.5
        predictions = probs > thresholds
        return where(predictions, pos_probs, neg_probs)

