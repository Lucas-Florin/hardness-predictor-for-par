import torch
import torch.nn as nn
import torch.nn.functional as F


class HardnessPredictorLoss(nn.Module):
    """
    Hardness Predictor Loss as defined in:
    Pei Wang, Nuno Vasconcelos 2018: Towards Realistic Predictors.
    """
    def __init__(self, num_classes=None, use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.loss_function = F.binary_cross_entropy_with_logits
        self.logits_function = nn.Sigmoid()

    def forward(self, hp_net_outputs, main_net_predictions, weight=None):
        """
        Calculate loss

        :param hp_net_outputs: HP network output
        :param main_net_predictions: Main net outputs (logits)
        :param weight:
        :return: sigmoid cross-entropy loss
        """
        hp_broadcasted = hp_net_outputs.new_empty(main_net_predictions.shape)
        hp_broadcasted[:] = hp_net_outputs
        loss = self.loss_function(hp_broadcasted, main_net_predictions, weight=weight)
        return loss

    def logits(self, inputs):
        return self.logits_function(inputs)
