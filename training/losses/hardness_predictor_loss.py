import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HardnessPredictorLoss(nn.Module):
    """
    Hardness Predictor Loss as defined in:
    Pei Wang, Nuno Vasconcelos 2018: Towards Realistic Predictors.
    """
    def __init__(self, use_deepmar_weighting, positive_attribute_ratios, num_attributes, sigma=1, use_gpu=True):
        super().__init__()
        self.use_gpu = use_gpu
        self.num_attributes = num_attributes
        self.use_deepmar_weighting = use_deepmar_weighting
        self.loss_function = F.binary_cross_entropy_with_logits
        self.logits_function = nn.Sigmoid()
        # Calculate attribute weight as defined in DeepMAR paper.
        self.positive_attribute_ratios = positive_attribute_ratios
        self.positive_weights = torch.tensor(np.exp((1 - positive_attribute_ratios) / sigma ** 2))
        self.negative_weights = torch.tensor(np.exp(positive_attribute_ratios / sigma ** 2))
        if self.use_gpu:
            self.positive_weights = self.positive_weights.cuda()
            self.negative_weights = self.negative_weights.cuda()
        self.batch_size = None
        self.weights = None

    def forward(self, hp_net_outputs, main_net_predictions, ground_truth_labels, weights=None):
        """
        Calculate loss

        :param hp_net_outputs: HP network output
        :param main_net_predictions: Main net outputs (logits)
        :param ground_truth_labels:
        :return: sigmoid cross-entropy loss
        """
        deepmar_weights = torch.where(ground_truth_labels == 1, self.positive_weights, self.negative_weights)
        if weights is not None:
            deepmar_weights *= weights
        if self.use_gpu:
            deepmar_weights = deepmar_weights.cuda()
        if not self.use_deepmar_weighting:
            deepmar_weights = None
        # In case there is only one hardness score for each image, broadcast it into the shape of main_net_predictions.
        hp_broadcasted = self.broadcast(hp_net_outputs)
        # The prediction correctness is the probability that is predicted for the ground truth label.
        prediction_correctness = torch.where(ground_truth_labels.byte(), main_net_predictions, 1 - main_net_predictions)
        loss = self.loss_function(hp_broadcasted, 1 - prediction_correctness, weight=deepmar_weights)
        return loss

    def logits(self, inputs):
        return self.logits_function(inputs)

    def broadcast(self, hp_net_outputs):
        if hp_net_outputs.shape[1] == self.num_attributes:
            return hp_net_outputs
        shape = (hp_net_outputs.shape[0], self.num_attributes)
        if type(hp_net_outputs) == torch.Tensor:
            hp_broadcasted = hp_net_outputs.new_empty(shape)
        else:
            hp_broadcasted = np.empty_like(hp_net_outputs, shape=shape)
        hp_broadcasted[:] = hp_net_outputs
        return hp_broadcasted
