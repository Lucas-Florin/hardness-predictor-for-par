# encoding: utf-8
"""
@author: Lucas Florin
@contact: lucasflorin4@gmail.com
"""




import glob
import re

import numpy as np
import os.path as osp

from utils.matlab_helper import MatlabMatrix
from .base import BaseDataset





class Market1501Attributes(BaseDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market'

    def __init__(self, root, verbose=True, **kwargs):
        super(Market1501Attributes, self).__init__(root)

        # parse directories.
        self.dataset_dir = osp.join(self. root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.attribute_dir = osp.join(self.dataset_dir, 'attributes')
        self.attributes_path = osp.join(self.attribute_dir, 'market_attribute.mat')
        self.test_attributes_path = osp.join(self.attribute_dir, 'gallery_market.mat')

        self._check_before_run()
        # Load data from .mat file
        data = MatlabMatrix.loadmat(self.attributes_path)['market_attribute']

        train_data = data['train']
        # generate numpy array with labels.
        train_labels = np.array([train_data[k] for k in sorted(train_data.keys()) if k != 'image_index']).T
        # a list of the attribute names.
        train_attributes = sorted(list(train_data.keys()))
        # Binarize attributes
        train_labels_bin, train_attributes_bin = self.binarize_labels(train_labels, train_attributes)
        # a list with the identity indexes.
        train_idx = train_data['image_index']

        # generate dictionary with identity indexes as keys and labels as values.
        train_labels_bin_dict = {int(i): l for i, l in zip(train_idx, train_labels_bin)}


        test_data = data['test']
        test_labels = np.array([test_data[k] for k in sorted(test_data.keys()) if k != 'image_index']).T
        test_attributes = sorted(list(test_data.keys()))
        test_labels_bin, test_attributes_bin = self.binarize_labels(test_labels, test_attributes)
        test_idx = test_data['image_index']
        test_labels_bin_dict = {int(i): l for i, l in zip(test_idx, test_labels_bin)}
        assert test_attributes_bin == train_attributes_bin
        attributes_bin = test_attributes_bin
        assert 0 == len(set(test_labels_bin_dict.keys()) & set(train_labels_bin_dict.keys()))

        # join training and testing label data in one dict.
        labels = dict(train_labels_bin_dict)
        labels.update(test_labels_bin_dict)

        self.attributes = attributes_bin
        self.num_attributes = test_labels_bin.shape[1]
        assert len(self.attributes) == self.num_attributes

        train = self._process_dir(self.train_dir, labels)
        query = self._process_dir(self.query_dir, labels)
        gallery = self._process_dir(self.gallery_dir, labels)

        if verbose:
            print("=> Market1501 Attributes loaded")
            self.print_dataset_statistics(train, query, gallery, attributes_bin)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.test = gallery

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.attribute_dir):
            raise RuntimeError("'{}' is not available".format(self.attribute_dir))
        if not osp.exists(self.attributes_path):
            raise RuntimeError("'{}' is not available".format(self.attributes_path))
        if not osp.exists(self.test_attributes_path):
            raise RuntimeError("'{}' is not available".format(self.test_attributes_path))

    @staticmethod
    def _process_dir(dir_path, labels):
        """
        Generate a list of the images in a directory. Ignores background and distractor images.
        :param dir_path: Path to the directory.
        :param labels: dict with the labels for each identity.
        :return: a list of tuples. The first element in the tuple is the path to the image, the second is a the labels
                    of the identity on the image.
        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        dataset = []
        for img_path in img_paths:
            # get the identity pictured in the image.
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid <= 0:
                continue  # junk and background images are just ignored
            assert 1 <= pid <= 1501

            dataset.append((img_path, labels[pid]))

        return dataset
