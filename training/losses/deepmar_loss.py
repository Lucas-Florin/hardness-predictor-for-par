"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import torch
import torch.nn as nn
import torch.cuda as cuda
import numpy as np


class DeepMARLoss(nn.Module):
    """
    DeepMAR loss as defined in:
    D. Li, X. Chen, and K. Huang. Multi-attribute learning for pedestrian attribute recognition in surveillance
    scenarios. In Pattern Recognition (ACPR), 2015 3rd IAPR Asian Conference on, 2015.
    """
    def __init__(self, positive_attribute_ratios, batch_size, use_gpu=True, sigma=1):
        super(DeepMARLoss, self).__init__()
        self.num_classes = len(positive_attribute_ratios)
        self.use_gpu = use_gpu
        self.loss_function = nn.BCEWithLogitsLoss()
        self.logits_function = nn.Sigmoid()
        # Calculate attribute weight as defined in paper.
        self.positive_attribute_ratios = torch.tensor(np.exp(-1 * positive_attribute_ratios / sigma ** 2))
        if self.use_gpu:
            self.positive_attribute_ratios = self.positive_attribute_ratios.cuda()
        self.batch_size = None
        self.weights = None

    def forward(self, inputs, targets):
        """
        Calculate loss

        :param inputs: network output (logits)
        :param targets: ground truth labels
        :return: sigmoid cross-entropy loss
        """
        if self.use_gpu:
            targets = targets.cuda()
        batch_size = targets.shape[0]

        if batch_size != self.batch_size:
            # Only runs the first time through
            self._update_batch_size(batch_size, inputs)
        loss = self.loss_function(inputs, targets)
        return loss

    def logits(self, inputs):
        return self.logits_function(inputs)

    def _update_batch_size(self, batch_size, inputs):
        """
        Generate a new weights tensor when the batch size changes.
        :param batch_size:
        :return:
        """
        # TODO: Make nicer.
        self.batch_size = batch_size
        self.weights = inputs.new_empty((batch_size, self.num_classes))
        self.weights[:] = self.positive_attribute_ratios
        if self.use_gpu:
            self.weights.cuda()
        self.loss_function = nn.BCEWithLogitsLoss(weight=self.weights)

