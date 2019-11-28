import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from data.data_manager import ImageDataManager
from args import argument_parser, image_dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs

from data.datasets import init_img_dataset
import tabulate as tab
import utils.plot as plot

parser = argument_parser()
args = parser.parse_args()


class DatasetAnalyzer:

    def __init__(self):
        global args
        self.args = args
        self.data_manager = ImageDataManager(not args.use_cpu, **image_dataset_kwargs(args))
        self.dataset = self.data_manager.dataset
        self.attributes = self.dataset.attributes
        self.positive_label_ratio = self.dataset.get_positive_attribute_ratio()
        self.menu()

    def menu(self):
        if self.args.plot_pos_atts:
            self.positive_label_ratio()
        if self.args.show_label_examples:
            self.label_examples()

    def label_examples(self):
        dataset = self.data_manager.split_dict["train"]
        labels = list()
        for (_, label) in self.data_manager.dataset.train:
            labels.append(label)
        labels = np.array(labels, dtype="bool")
        att_list = self.args.select_atts
        assert len(att_list) == 1
        att = att_list[0]
        att_idx = self.attributes.tolist().index(att)
        num_pos = self.args.num_save_hard
        num_neg = self.args.num_save_easy
        labels = labels[:, att_idx].flatten()
        idxs = np.arange(labels.size)
        pos_idxs = idxs[labels]
        neg_idxs = idxs[np.logical_not(labels)]
        sel_idxs = np.random.choice(pos_idxs, num_pos).tolist() + np.random.choice(neg_idxs, num_neg).tolist()
        # Display the image examples.
        plot.show_img_grid(dataset, sel_idxs, None, labels[sel_idxs],
                           save_plot=False)


    def positive_label_ratio(self):
        table = tab.tabulate(zip(self.attributes, self.positive_label_ratio), floatfmt='.2%')
        print("----------------------")
        print("Analyzing Dataset: " + args.dataset_name)
        print("Total Positive Quota: ")
        print(table)


if __name__ == '__main__':
    da = DatasetAnalyzer()