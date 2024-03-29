import torch
from torch import nn
import torchvision


class ResNet50(nn.Module):

    def __init__(self, num_classes, pretrained=True, **kwargs):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Linear(2048, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.bn(x)
        return x

    def get_params_finetuning(self):
        return list(self.backbone.parameters())[:-2]

    def get_params_fresh(self):
        return list(self.backbone.fc.parameters()) + list(self.bn.parameters())




