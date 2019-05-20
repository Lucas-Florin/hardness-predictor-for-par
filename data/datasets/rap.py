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


class RAPv2(BaseDataset):
    """
    RAPv2.0

    """
    # TODO: Paper reference.
    dataset_dir = 'RAP'
    split_idx = 0

    def __init__(self, root, verbose=True, **kwargs):
        super(RAPv2, self).__init__(root)

        # parse directories.
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.img_dir = osp.join(self.dataset_dir, 'RAP_dataset')
        self.attribute_dir = osp.join(self.dataset_dir, 'RAP_annotation')
        self.attributes_path = osp.join(self.attribute_dir, 'RAP_annotation.mat')
        self._check_before_run()

        # Load data from .mat file
        data = MatlabMatrix.loadmat(self.attributes_path)['RAP_annotation']
        attributes = data["attribute"]  # List of all annotated attributes.
        # List of the idxs of the attributes selected for PAR.
        selected_attributes = np.array(data["selected_attribute"])
        attributes = [attributes[i] for i in selected_attributes]
        labels = np.array(data["data"])  # The labels for each image.
        labels = labels[:, selected_attributes]  # Discard the labels not used for PAR.
        print(labels.shape)
        img_file_names = data["name"]  # Filenames for the images.
        partitions = data["partition_attribute"][self.split_idx]  # Dataset partition.
        train_idx = np.array(partitions["train_index"]) - 1
        val_idx = np.array(partitions["val_index"]) - 1
        test_idx = np.array(partitions["test_index"]) - 1

        train = [(osp.join(self.img_dir, img_file_names[idx]), label)
                 for idx, label in zip(train_idx, labels[train_idx, :])]

        val = [(osp.join(self.img_dir, img_file_names[idx]), label)
               for idx, label in zip(val_idx, labels[val_idx, :])]

        test = [(osp.join(self.img_dir, img_file_names[idx]), label)
                for idx, label in zip(test_idx, labels[test_idx, :])]

        self.train = train
        self.val = val
        self.test = test
        self.attributes = attributes
        self.num_attributes = len(attributes)

        if verbose:
            print("=> RAPv2.0 Attributes loaded")
            self.print_dataset_statistics(train, val, test, attributes)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not osp.exists(self.attribute_dir):
            raise RuntimeError("'{}' is not available".format(self.attribute_dir))
        if not osp.exists(self.attributes_path):
            raise RuntimeError("'{}' is not available".format(self.attributes_path))

