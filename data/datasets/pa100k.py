# encoding: utf-8
"""
@author: Lucas Florin
@contact: lucasflorin4@gmail.com
"""


import os.path as osp

from utils.matlab_helper import MatlabMatrix
from .base import BaseDataset
import numpy as np

class PA100K(BaseDataset):
    """
    PA-100K

    """
    # TODO: Paper reference.
    dataset_dir = 'pa100k'

    def __init__(self, root, verbose=True, **kwargs):
        super(PA100K, self).__init__(root)

        # parse directories.
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.img_dir = osp.join(self.dataset_dir, 'release_data/release_data')
        self.attributes_path = osp.join(self.dataset_dir, 'annotation.mat')
        self._check_before_run()

        # Load data from .mat file
        data = MatlabMatrix.loadmat(self.attributes_path)
        attributes = data["attributes"]  # List of all annotated attributes.

        train_labels = data["train_label"]
        train = [(osp.join(self.img_dir, filename), label)
                 for filename, label in zip(data["train_images_name"], train_labels)]

        val_labels = data["val_label"]
        val = [(osp.join(self.img_dir, filename), label)
               for filename, label in zip(data["val_images_name"], val_labels)]

        test_labels = data["test_label"]
        test = [(osp.join(self.img_dir, filename), label)
                for filename, label in zip(data["test_images_name"], test_labels)]

        self.labels = np.concatenate((train_labels, val_labels, test_labels), axis=0)
        self.train = train
        self.val = val
        self.test = test
        self.attributes = attributes
        self.num_attributes = len(attributes)
        self.attribute_grouping = list(range(self.num_attributes))

        if verbose:
            print("=> PA-100K Attributes loaded")
            self.print_dataset_statistics(train, val, test, attributes)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not osp.exists(self.attributes_path):
            raise RuntimeError("'{}' is not available".format(self.attributes_path))

