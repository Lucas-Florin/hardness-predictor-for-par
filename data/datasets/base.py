import os
import os.path as osp
import numpy as np


class BaseDataset(object):
    """
    Base class of attribute dataset
    """

    def __init__(self, root):
        self.root = osp.expanduser(root)
    """
    def get_imagedata_info(self, data):
        #pids, cams = [], []
        #for _, pid, camid in data:
        #    pids += [pid]
        #    cams += [camid]
        #pids = set(pids)
        #cams = set(cams)
        #num_pids = len(pids)
        #num_cams = len(cams)
        #num_imgs = len(data)
        num_attributes = data[]
        #return num_imgs
    """
    def print_dataset_statistics(self, train, val, test, attributes):
        att_names = '{}: {}\n'.format(1, attributes[0])
        for i, a in enumerate(attributes[1:]):
            att_names += '  {}: {}\n'.format(i+2, a)

        print('Image Dataset statistics:')
        print('  ---------------------')
        print('  Attributes:')
        print('  {}'.format(att_names))
        print('  => Number of attributes: {}'.format(len(attributes)))
        print('  ---------------------')
        print('  subset   | # images')
        print('  ---------------------')
        print('  train    | {:8d} '.format(len(train)))
        print('  val      | {:8d} '.format(len(val)))
        print('  test     | {:8d} '.format(len(test)))
        print('  ---------------------')

    def load_from_file(self, file):
        """
        Load attributes, files and labels from csv file

        :param file: file path
        :return: files, labels, attribute names
        """
        data = np.genfromtxt(file, dtype='|U', delimiter=',', comments='!')
        attributes = data[0, 1:]
        data = data[1:]
        files = data[:, 0]
        labels = data[:, 1:]

        dataset = list()
        for f, l in zip(files, labels):
            dataset.append((f, l))
        if labels.shape[1] != len(attributes):
            raise Exception('Number of attributes and number of labels aren\'t equal: {} != {}'.format(labels.shape[1],
                                                                                                       len(attributes)))
        return dataset, attributes
