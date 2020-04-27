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
    Reference:
    [Li et al. 2019] Multi-attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios

    """
    dataset_dir = 'RAP'
    split_idx = 0
    bb_coordinate_idxs = np.array(list(range(120, 152, 4)))
    origin_coordinate_idx = 120

    def __init__(self, root, verbose=True, full_attributes=False, **kwargs):
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
        # List of the indexes of the attributes selected for PAR.
        selected_attributes = np.array(data["selected_attribute"]) - 1
        labels = np.array(data["data"])  # The labels for each image.
        img_file_names = data["name"]  # Filenames for the images.

        partitions = data["partition_attribute"][self.split_idx]  # Dataset partition.
        train_idx = np.array(partitions["train_index"]) - 1
        val_idx = np.array(partitions["val_index"]) - 1
        test_idx = np.array(partitions["test_index"]) - 1

        visibility = np.zeros((labels.shape[0], len(self.bb_coordinate_idxs)), dtype="bool")
        for i in range(len(self.bb_coordinate_idxs)):
            visibility[:, i] = np.logical_and(labels[:, self.bb_coordinate_idxs[i] + 2],
                                              labels[:, self.bb_coordinate_idxs[i] + 3])

        if not full_attributes:
            attributes = [attributes[i] for i in selected_attributes]
            labels = labels[:, selected_attributes]  # Discard the labels not used for PAR.

        attribute_visibility_region = np.zeros((len(attributes), ), dtype='int')
        attribute_visibility_region[np.char.startswith(attributes, "hs-")] = 1
        attribute_visibility_region[np.char.startswith(attributes, "ub-")] = 2
        attribute_visibility_region[np.logical_or(np.char.startswith(attributes, "lb-"),
                                                  np.char.startswith(attributes, "shoes-"))] = 3

        filenames = [osp.join(self.img_dir, file) for file in img_file_names]

        train = [(filenames[idx], label) for idx, label in zip(train_idx, labels[train_idx, :])]
        val = [(filenames[idx], label) for idx, label in zip(val_idx, labels[val_idx, :])]
        test = [(filenames[idx], label) for idx, label in zip(test_idx, labels[test_idx, :])]

        self.filenames = filenames
        self.train = train
        self.val = val
        self.test = test
        self.attributes = np.array(attributes)
        self.num_attributes = len(attributes)
        self.attribute_grouping = list(range(self.num_attributes))
        self.labels = labels
        self.visibility = visibility
        self.attribute_visibility_region = attribute_visibility_region
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

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

