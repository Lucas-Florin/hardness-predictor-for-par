import os
import os.path as osp
import numpy as np


class BaseDataset(object):
    """
    Base class of attribute dataset
    """

    def __init__(self, root):
        self.root = osp.expanduser(root)

    @staticmethod
    def binarize_labels(labels, attributes):
        """
        Exchange each non-binary attribute in a dataset with the corresponding number of binary attributes.
        :param labels: a numpy array with the labels for all the datapoints in the dataset.
        :param attributes: a list of the names of the attributes in the order they appear in labels.
        :return: a numpy array with only binary attributes and the corresponding attribute name list.
        """

        binary_attributes = list()
        max_values = np.max(labels, axis=0)
        bin_labels = np.zeros([len(labels), 0])
        for i, v in enumerate(max_values):
            attribute_name = attributes[i]
            if v > 2:
                # binarize attribute if more than two values
                loc = np.zeros([len(labels), v], dtype=int)
                loc[(np.arange(len(labels)), labels[:, i] - 1)] = 1

                for j in range(v):
                    binary_attributes.append(attribute_name + str(j + 1))

            else:
                loc = (labels[:, i] - 1)[:, None]
                binary_attributes.append(attribute_name)
            bin_labels = np.concatenate([bin_labels, loc], axis=1)
        return bin_labels.astype(int), binary_attributes

    @staticmethod
    def print_dataset_statistics(train, val, test, attributes):
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

    @staticmethod
    def load_from_file(file):
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

    def get_positive_quota(self):
        labels = self.labels
        attributes = self.attributes
        num_datapoints = labels.shape[0]
        total_positive = labels.sum(0)
        total_positive_quota = total_positive / num_datapoints
        return total_positive_quota
