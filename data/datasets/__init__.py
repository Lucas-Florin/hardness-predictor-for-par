from .market1501 import Market1501Attributes

# Map the dataset names to the corresponding dataset classes.
__imgreid_factory = {
    'market1501': Market1501Attributes,
}


def init_img_dataset(name, **kwargs):
    """
    Get and initialize (load) the dataset corresponding to the name.
    :param name: A string describing the name of the dataset.
    :param kwargs: command line arguments.
    :return: The loaded dataset.
    """
    if name not in list(__imgreid_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)
