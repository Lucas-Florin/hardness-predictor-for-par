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
        self.thresholds_torch = None
        self.thresholds_np = None

    def __call__(self, probs):
        """
        Calibrate.
        :param probs: An array of doubles to be calibrated.
        :return: An array of the same type as probs, but with the values calibrated.
        """
        raise NotImplementedError

    def is_initialized(self):
        return self.thresholds_np is not None and self.thresholds_torch is not None

    def update_thresholds(self, thresholds, device='cpu'):
        if thresholds is not None:
            if torch.is_tensor(thresholds):
                self.thresholds_torch = thresholds.clone().detach()
                self.thresholds_np = self.thresholds_torch.cpu().numpy()
            else:
                self.thresholds_torch = torch.tensor(thresholds, device=device)
                self.thresholds_np = self.thresholds_torch.cpu().numpy()
        else:
            self.thresholds_torch = None
            self.thresholds_np = None