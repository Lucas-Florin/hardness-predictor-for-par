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

parser = argument_parser()
args = parser.parse_args()


def main():
    global args
    dataset = init_img_dataset(args.dataset_name, **image_dataset_kwargs(args))
    labels = dataset.labels
    attributes = dataset.attributes
    num_datapoints = labels.shape[0]
    total_positive = labels.sum(0)#.reshape((1, labels.shape[1]))
    total_positive_quota = total_positive / num_datapoints
    table = tab.tabulate(zip(attributes, total_positive_quota), floatfmt='.2%')
    print("----------------------")
    print("Analyzing Dataset: " + args.dataset_name)
    print("Total Positive Quota: ")
    print(table)


if __name__ == '__main__':
    main()