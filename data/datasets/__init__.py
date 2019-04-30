from .market1501 import Market1501
#from .dukemtmcreid import DukeMTMCreID

__imgreid_factory = {
    'market1501': Market1501,
    #'dukemtmcreid': DukeMTMCreID,
}


def init_img_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)
