from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo


def resnet50_lib(num_classes, loss=None, pretrained=False, **kwargs):
    return torchvision.models.resnet50(pretrained=pretrained, num_classes=num_classes)
