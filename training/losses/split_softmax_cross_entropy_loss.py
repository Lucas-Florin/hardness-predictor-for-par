"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import torch.nn as nn
from .softmax_split import SoftmaxSplit


class SplitSoftmaxCrossEntropyLoss(nn.Module):
    """
    Split Softmax cross-entropy loss.

    Softmax is applied over each group of attributes separately. Sigmoid is applied to attributes that do not belong
    to a group.
    """
    def __init__(self, attribute_groupings, use_gpu=True):
        super(SplitSoftmaxCrossEntropyLoss, self).__init__()
        self.num_classes = len(attribute_groupings)
        self.attribute_groupings = attribute_groupings
        self.use_gpu = use_gpu
        self.loss_function = nn.BCELoss()
        self.logits_function = SoftmaxSplit(attribute_groupings)

    def forward(self, inputs, targets):
        """
        Calculate loss

        :param inputs: network output (logits)
        :param targets: ground truth labels
        :return: sigmoid cross-entropy loss
        """
        if self.use_gpu:
            targets = targets.cuda()
        logits = self.logits(inputs)
        loss = self.loss_function(logits, targets)
        return loss

    def logits(self, inputs):
        return self.logits_function(inputs)
