import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# TODO: make code DRY: join with main model setups.


class HP_VGG16(nn.Module):

    def __init__(self):
        super(HP_VGG16, self).__init__()
        trunk_net = torchvision.models.vgg16_bn(pretrained=True)
        for p in trunk_net.parameters():
            p.requires_grad = True

        self.trunk_net = trunk_net

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, input):
        x = self.trunk_net(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


class HP_RES50(nn.Module):
    """
    As defined in:
    Pei Wang, Nuno Vasconcelos 2018: Towards Realistic Predictors.
    """

    def __init__(self, num_classes=1, pretrained=False):
        """

        :param num_classes: number of dimensions of the output vector. Can be one or the number of attributes.
        """
        super(HP_RES50, self).__init__()
        trunk_net = torchvision.models.resnet50(pretrained=pretrained)
        for p in trunk_net.parameters():
            p.requires_grad = True

        self.trunk_net = trunk_net
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, input):
        x = self.trunk_net(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class HP_RES50_nh(nn.Module):
    """

    """

    def __init__(self, num_classes=1, pretrained=False):
        """

        :param num_classes: number of dimensions of the output vector. Can be one or the number of attributes.
        """
        super(HP_RES50_nh, self).__init__()
        trunk_net = torchvision.models.resnet50(pretrained=pretrained, num_classes=num_classes)
        for p in trunk_net.parameters():
            p.requires_grad = True

        self.trunk_net = trunk_net

    def forward(self, y):
        return self.trunk_net(y)

class HP_Squeezenet(nn.Module):
    """
    As defined in:
    Pei Wang, Nuno Vasconcelos 2018: Towards Realistic Predictors.
    """

    def __init__(self, num_classes=1, pretrained=False):
        """

        :param num_classes: number of dimensions of the output vector. Can be one or the number of attributes.
        """
        super(HP_Squeezenet, self).__init__()
        trunk_net = torchvision.models.squeezenet1_0(pretrained=pretrained)
        for p in trunk_net.parameters():
            p.requires_grad = True

        self.trunk_net = trunk_net
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, input):
        x = self.trunk_net(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class HP_Densenet(nn.Module):
    """
    As defined in:
    Pei Wang, Nuno Vasconcelos 2018: Towards Realistic Predictors.
    """

    def __init__(self, num_classes=1, pretrained=False):
        """

        :param num_classes: number of dimensions of the output vector. Can be one or the number of attributes.
        """
        super(HP_Densenet, self).__init__()
        trunk_net = torchvision.models.densenet121(pretrained=pretrained)
        for p in trunk_net.parameters():
            p.requires_grad = True

        self.trunk_net = trunk_net
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, input):
        x = self.trunk_net(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
