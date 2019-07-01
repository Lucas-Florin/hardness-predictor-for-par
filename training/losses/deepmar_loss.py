"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.loss_function = F.binary_cross_entropy_with_logits
        self.logits_function = nn.Sigmoid()
        # Calculate attribute weight as defined in paper.
        self.positive_attribute_ratios = positive_attribute_ratios
        self.positive_weights = torch.tensor(np.exp((1 - positive_attribute_ratios) / sigma ** 2))
        self.negative_weights = torch.tensor(np.exp(positive_attribute_ratios / sigma ** 2))
        if self.use_gpu:
            self.positive_weights = self.positive_weights.cuda()
            self.negative_weights = self.negative_weights.cuda()
        self.batch_size = None
        self.weights = None

    def forward(self, inputs, targets, weights=None):
        """
        Calculate loss

        :param inputs: network output (logits)
        :param targets: ground truth labels
        :param weights:
        :return: sigmoid cross-entropy loss
        """
        if self.use_gpu:
            targets = targets.cuda()
        deepmar_weights = torch.where(targets == 1, self.positive_weights, self.negative_weights)
        if weights is not None:
            deepmar_weights *= weights
        if self.use_gpu:
            deepmar_weights = deepmar_weights.cuda()
        loss = self.loss_function(inputs, targets, weight=deepmar_weights)
        return loss

    def logits(self, inputs):
        return self.logits_function(inputs)



