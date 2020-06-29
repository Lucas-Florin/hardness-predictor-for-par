from .resnet import *
from .HP_Net import *


# TODO: Rely more on library definitions.


__model_factory = {
    # image classification models
    'resnet50': ResNet50,

    'hp_net_resnet50': HP_RES50,
    'hp_net_resnet50_nh': HP_RES50_nh,
    'hp_net_squeezenet1_0': HP_Squeezenet,
    'hp_net_densenet121': HP_Densenet
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __model_factory[name](*args, **kwargs)
