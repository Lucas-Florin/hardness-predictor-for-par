import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings
import tabulate as tab
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from data.data_manager import ImageDataManager
from data.dataset_loader import read_image
import models
from training.losses import SigmoidCrossEntropyLoss, HardnessPredictorLoss, DeepMARLoss, SplitSoftmaxCrossEntropyLoss
from utils.iotools import check_isfile, save_checkpoint
from utils.avgmeter import AverageMeter
from utils.loggers import Logger, AccLogger
from utils.torchtools import count_num_param, open_all_layers, open_specified_layers, accuracy, load_pretrained_weights
from utils.generaltools import set_random_seed
import evaluation.metrics as metrics
from training.optimizers import init_optimizer
from training.lr_schedulers import init_lr_scheduler
import utils.plot as plot
from trainer import Trainer


def main(args):
    # Decide which processor (CPU or GPU) to use.
    if not args.use_avai_gpus:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    # Start logger.
    ts = time.strftime("%Y-%m-%d_%H-%M-%S_")
    file = open(osp.join(args.save_experiment, "result_dict.pickle"), 'rb')
    result_dict = pickle.load(file)
    file.close()

    label_prediction_probs = result_dict["prediction_probs"]
    label_predictions = result_dict["predictions"]
    hp_scores = result_dict["hp_scores"]
    loaded_args = result_dict["args"]
    labels = result_dict["labels"]

    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    hard_att_labels = None
    hard_att_pred = None
    print("Looking at Hard attribute " + args.hard_att)
    att_idx = dm.attributes.index(args.hard_att)
    hard_att_labels = labels[:, att_idx]
    hard_att_pred = label_prediction_probs[:, att_idx]
    if not loaded_args.hp_net_simple:
        # If a valid attribute is given, the hardness scores for that attribute are selected, else the mean
        # over all attributes is taken.
        hp_scores = hp_scores[:, att_idx]

    if args.plot_acc_hp:
        filename = osp.join(args.save_experiment, ts + "accuracy_over_hardness.png")
        title = "Mean Accuracy over hardness for " + (args.load_weights if args.load_weights else ts)

        plot.show_accuracy_by_hardness(filename, title, args.hard_att, hard_att_labels, hard_att_pred, hp_scores)

    if args.num_save_hard + args.num_save_easy > 0:
        # This part only gets executed if the corresponding arguments are passed at the terminal.
        hp_scores = hp_scores.flatten()
        sorted_idxs = hp_scores.argsort()
        if args.show_pos_samples:
            sorted_idxs = sorted_idxs[hard_att_labels[sorted_idxs]]
        elif args.show_neg_samples:
            sorted_idxs = sorted_idxs[np.logical_not(hard_att_labels[sorted_idxs])]
        # Select easy and hard examples as specified in the terminal.
        hard_idxs = np.concatenate((sorted_idxs[:args.num_save_easy], sorted_idxs[-args.num_save_hard:]))

        filename = osp.join(args.save_experiment,  ts + "hard_images.png")
        title = "Examples by hardness for " + (args.load_weights if args.load_weights else ts)
        if hard_att_labels is not None:
            hard_att_labels = hard_att_labels[hard_idxs]
        if hard_att_pred is not None:
            hard_att_pred = hard_att_pred[hard_idxs]
        # Display the image examples.
        plot.show_img_grid(dm.split_dict[args.eval_split], hard_idxs, filename, title, args.hard_att,
                      hard_att_labels, hp_scores[hard_idxs], hard_att_pred)



if __name__ == '__main__':
    # global variables
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
