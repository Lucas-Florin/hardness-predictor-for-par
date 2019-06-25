import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmoidCrossEntropyLoss(nn.Module):
    """
    Sigmoid cross-entropy loss
    """
    def __init__(self, num_classes=None, use_gpu=True):
        super(SigmoidCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.loss_function = F.binary_cross_entropy_with_logits
        self.logits_function = nn.Sigmoid()

    def forward(self, inputs, targets, weight=None):
        """
        Calculate loss

        :param inputs: network output (logits)
        :param targets: ground truth labels
        :param weight:
        :return: sigmoid cross-entropy loss
        """
        if self.use_gpu:
            targets = targets.cuda()
        loss = self.loss_function(inputs, targets, weight=weight)
        return loss

    def logits(self, inputs):
        return self.logits_function(inputs)
