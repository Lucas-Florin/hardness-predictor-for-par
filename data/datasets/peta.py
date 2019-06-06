# encoding: utf-8
"""
@author: Lucas Florin
@contact: lucasflorin4@gmail.com
"""




import glob
import re

import numpy as np
import os.path as osp
import re
import random

from utils.matlab_helper import MatlabMatrix
from .base import BaseDataset





class PETA(BaseDataset):
    """
    PETA Dataset
    Reference:
    Deng et al. 2014 Pedestrian Attribute Recognition At Far Distance.

    Dataset statistics:
    # images: 9500 (train) + 1900 (val) + 7600 (gallery)
    """
    dataset_dir = 'peta/original'
    # Attributes discarded because of low frequency, as defined in PETA Readme.
    discarded_attributes = {'accessoryFaceMask', 'lowerBodyLogo', 'accessoryShawl', 'lowerBodyThickStripes'}
    # The possible names for colors.
    color_names = ['Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
    # The size of dataset splits.
    num_train = 9500
    num_val = 1900
    num_test = 7600
    shuffle_fname = "data/datasets/peta_shuffle.csv"

    def __init__(self, root, verbose=True, **kwargs):
        super(PETA, self).__init__(root)

        # parse directories.
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # Used to find the directories with the different datasets.
        self.archive_dir_pattern = (self.dataset_dir + "/*/archive/")
        # Used to find the pedestrian id in an image filename.
        self.img_fname_pattern = re.compile(r"[\d]+")

        self._check_before_run()
        attributes = set()
        datasets = dict()  # Holds references to each of the different datasets.
        num_ids = 0  # Counts the number of different pedestrian ids in all the datasets.
        for dir_name in sorted(list(glob.glob(self.archive_dir_pattern))):
            data = dict()
            label_file = open(osp.join(dir_name, "Label.txt"), "r")  # This file has the labels for each id.
            while True:
                line = label_file.readline()
                if line == "":
                    break  # When reaching end of label file
                line = line.split()  # split the line along whitespaces.
                # The first element of the line is the pedestrian ID. Sometimes instead it is the filename and the ID
                # has to be extracted from it.
                ped_id = int(line[0].split(".")[0])
                # All following elements are positive labels. Discarded attributes are ignored.
                labels = set(line[1:]) - self.discarded_attributes
                # Attributes that have not been seen before are added to the set of attributes.
                attributes |= labels
                data[ped_id] = labels
            datasets[dir_name] = data
            num_ids += len(data)

        attributes = sorted(list(attributes))
        # This dict saves color attributes as a group. Binary attributes have value None
        grouped_atts = dict()
        # Regex patterns for each color.
        color_patterns = {color: re.compile(r"(.+)(" + color + r")") for color in self.color_names}
        # Extract all non-binary attributes (colored attributes) and group them.
        for att in attributes:  # For each binarized attribute
            colored_attribute_name = None
            for color in self.color_names:  # For each color.
                match = color_patterns[color].search(att)  # Check if a color is present in attribute name.
                if match is not None:  # If a color is found
                    # Extract the prefix (name of the non-binary colored attribute) and exit the loop.
                    colored_attribute_name = match.group(1)
                    break
            if colored_attribute_name is None:
                # If no match was found the attribute stays binary.
                grouped_atts[att] = None
            else:
                # If a color name was found ...
                if colored_attribute_name not in grouped_atts:
                    # If a non-binary attribute already exists the new color is added.
                    grouped_atts[colored_attribute_name] = {color}
                else:
                    # If not a new non-binary attribute is created.
                    grouped_atts[colored_attribute_name].add(color)
        attributes = list()
        # Create a list detailing the grouping of binarized attributes.
        attribute_grouping = list()
        group_counter = 0
        for att in sorted(list(grouped_atts)):
            if grouped_atts[att] is not None:
                attributes += [att + color for color in sorted(list(grouped_atts[att]))]
                attribute_grouping += [group_counter] * len(grouped_atts[att])
            else:
                attributes.append(att)
                attribute_grouping.append(group_counter)
            group_counter += 1
        assert len(attribute_grouping) == len(attributes)
        assert group_counter == len(grouped_atts)
        self.attributes = attributes
        self.attribute_grouping = attribute_grouping
        num_attributes = len(attributes)
        self.num_attributes = num_attributes
        self.num_ids = num_ids
        att_idxs = {a: i for i, a in enumerate(attributes)}
        dataset = list()  # This list saves tuples of image file names and labels.

        for dir_name in sorted(list(datasets)):  # For each dataset
            data = datasets[dir_name]
            data_bin = dict()  # Saves the binary label vectors for each datapoint.
            for i in sorted(list(data)):  # For each datapoint in the dataset
                # Generate a binary vector representation of the labels.
                labels = np.zeros((num_attributes, ), dtype="int8")  # Init binary label vector with zeros.
                for label in data[i]:
                    # For each positive label of the datapoint set the corresponding value to 1.
                    labels[att_idxs[label]] = 1
                data_bin[i] = labels

            img_files = sorted(list(glob.glob(osp.join(dir_name, "*.*"))))  # Find all files in the folder.
            for fpath in img_files:  # For each file
                fname = osp.basename(fpath)  # Separate filename from the rest of the path.
                if fname == "Label.txt":
                    continue  # Ignore the label file.
                ped_id = self.img_fname_pattern.search(fname).group()  # Extract pedestrian ID from filename.
                ped_id = int(ped_id)
                dataset.append((fpath, data_bin[ped_id]))  # Add tuple of filename and label vector to dataset.
        shuffle = np.loadtxt(self.shuffle_fname, delimiter=",").astype(np.int)
        dataset = np.array(dataset)[shuffle, :].tolist()
        self.dataset = dataset
        self.num_datapoints = len(dataset)
        labels_np = np.zeros((self.num_datapoints, num_attributes), dtype="int8")
        counter = 0
        for fpath, labels in dataset:
            labels_np[counter, :] = labels
            counter += 1
        self.labels = labels_np

        # Randomly assign dataset splits according to defined split sizes.
        self.train = dataset[:self.num_train]
        self.val = dataset[self.num_train:self.num_train + self.num_val]
        self.test = dataset[self.num_train + self.num_val:]
        if verbose:
            print("=> PETA loaded")
            self.print_dataset_statistics(self.train, self.val, self.test, self.attributes)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

