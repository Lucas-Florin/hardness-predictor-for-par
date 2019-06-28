import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

    def forward(self, hp_net_outputs, main_net_predictions, ground_truth_labels):
        """
        Calculate loss

        :param hp_net_outputs: HP network output
        :param main_net_predictions: Main net outputs (logits)
        :param ground_truth_labels:
        :return: sigmoid cross-entropy loss
        """
        #print("hpnet", hp_net_outputs.shape)
        #print(hp_net_outputs)
        # In case there is only one hardness score for each image, broadcast it into the shape of main_net_predictions.
        hp_broadcasted = hp_net_outputs.new_empty(main_net_predictions.shape)
        hp_broadcasted[:] = hp_net_outputs
        #print(ground_truth_labels.byte())
        # The prediction correctness is the probability that is predicted for the ground truth label.
        prediction_correctness = torch.where(ground_truth_labels.byte(), main_net_predictions, 1 - main_net_predictions)
        #print("broadcast", hp_broadcasted.shape)
        #print(hp_broadcasted)
        loss = self.loss_function(hp_broadcasted, 1 - prediction_correctness)
        return loss

    def logits(self, inputs):
        return self.logits_function(inputs)
