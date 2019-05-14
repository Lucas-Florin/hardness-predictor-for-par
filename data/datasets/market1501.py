# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
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
        self.dataset_dir = osp.join(self. root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.attribute_dir = osp.join(self.dataset_dir, 'attributes')
        self.attributes_path = osp.join(self.attribute_dir, 'market_attribute.mat')
        self.test_attributes_path = osp.join(self.attribute_dir, 'gallery_market.mat')

        self._check_before_run()
        data = MatlabMatrix.loadmat(self.attributes_path)['market_attribute']
        train_data = data['train']
        train_labels = np.array([train_data[k] for k in sorted(train_data.keys()) if k != 'image_index']).T
        train_idx = train_data['image_index']
        train_labels = {int(i): l for i, l in zip(train_idx, self.binarize_labels(train_labels))}
        train_attributes = sorted(list(train_data.keys()))

        test_data = data['test']
        test_labels = np.array([test_data[k] for k in sorted(test_data.keys()) if k != 'image_index']).T
        test_idx = test_data['image_index']
        test_labels = {int(i): l for i, l in zip(test_idx, self.binarize_labels(test_labels))}
        test_attributes = sorted(list(test_data.keys()))
        assert test_attributes == train_attributes
        assert 0 == len(set(test_labels.keys()) & set(train_labels.keys()))
        labels = dict(train_labels)
        labels.update(test_labels)

        self.attributes = train_attributes
        self.num_attributes = list(train_labels.values())[0].size

        train = self._process_dir(self.train_dir, labels)
        query = self._process_dir(self.query_dir, labels)
        gallery = self._process_dir(self.gallery_dir, labels)

        if verbose:
            print("=> Market1501 Attributes loaded")
            self.print_dataset_statistics(train, query, gallery, train_attributes)

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
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        dataset = []
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid <= 0:
                continue  # junk images are just ignored
            assert 1 <= pid <= 1501

            dataset.append((img_path, labels[pid]))

        return dataset
