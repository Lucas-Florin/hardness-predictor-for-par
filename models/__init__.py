from .resnet import ResNet50
from .HP_Net import HP_ResNet50, HP_ResNet50_strong


__model_factory = {
    # image classification models
    'resnet50': ResNet50,

    # Hardness Predictor models
    'hp_net_resnet50': ResNet50,
    'hp_net_resnet50_fc_head_strong': HP_ResNet50_strong,
    'hp_net_resnet50_fc_head': HP_ResNet50,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __model_factory[name](*args, **kwargs)
