from .resnet import *
from .HP_Net import *


# TODO: Rely more on library definitions.


__model_factory = {
    # image classification models
    'resnet50': ResNet50,

    'hp_net_resnet50': HP_ResNet50,
    'hp_net_resnet50_strong': HP_ResNet50_strong,
    'hp_net_resnet50_nh_strong': ResNet50,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __model_factory[name](*args, **kwargs)
