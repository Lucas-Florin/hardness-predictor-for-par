import torch
import torch.nn as nn
import torchvision


class HP_ResNet50(nn.Module):
    """
    As defined in:
    Pei Wang, Nuno Vasconcelos 2018: Towards Realistic Predictors.
    """

    def __init__(self, num_classes=1, pretrained=True):
        """

        :param num_classes: number of dimensions of the output vector. Can be one or the number of attributes.
        """
        super(HP_ResNet50, self).__init__()

        self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        self.head = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)

        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def get_params_finetuning(self):
        return list()

    def get_params_fresh(self):
        return list(self.parameters())


class HP_ResNet50_strong(nn.Module):
    """
    As defined in:
    Pei Wang, Nuno Vasconcelos 2018: Towards Realistic Predictors.
    """

    def __init__(self, num_classes, pretrained=True):
        """

        :param num_classes: number of dimensions of the output vector. Can be one or the number of attributes.
        """
        super(HP_ResNet50_strong, self).__init__()

        self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Linear(2048, 1000)

        self.head = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.BatchNorm1d(num_classes)

        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def get_params_finetuning(self):
        return list(self.backbone.parameters())[:-2]

    def get_params_fresh(self):
        return list(self.backbone.fc.parameters()) + list(self.head.parameters())




