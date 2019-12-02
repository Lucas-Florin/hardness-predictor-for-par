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
import evaluation.rejectors as rejectors
from evaluation.result_manager import ResultManager


class RealisticPredictorAnalyzer:
    """
    A tool to analyze the results of the training of a realistic predictor.
    """
    # TODO: Complete conversion to class.
    # TODO: Use result manager
    # TODO: Document better
    # TODO: Implement runtime menu
    def __init__(self, args):
        self.args = args
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
        self.result_dict = result_dict

        file.close()
        self.result_manager = ResultManager(self.result_dict)
        print("Loaded result dict with keys: ")
        print(sorted(list(self.result_dict.keys())))
        """
        if "rejection_thresholds" in self.result_dict:
            self.rejector.load_thresholds(self.result_dict["rejection_thresholds"])
            if self.rejector.is_initialized():
                print("Loaded rejection thresholds. ")
            else:
                print("Loaded uninitialized (None) rejection thresholds. ")
        else:
            print("WARNING: Could not load rejection thresholds. ")
        """
        split = self.args.eval_split
        ignored_test_datapoints = None

        labels, prediction_probs, predictions, hp_scores = self.result_manager.get_outputs(split)
        _, prediction_probs_train, _, hp_scores_train = self.result_manager.get_outputs("train")
        _, prediction_probs_val, _, hp_scores_val = self.result_manager.get_outputs("val")
        if self.result_manager.check_output_dict("test"):
            _, prediction_probs_test, _, hp_scores_test = self.result_manager.get_outputs("test")
        else:
            prediction_probs_test, hp_scores_test = None, None
        loaded_args = result_dict["args"]
        f1_calibration_thresholds = result_dict["f1_thresholds"]
        attributes = result_dict["attributes"]
        positivity_ratio = result_dict["positivity_ratio"]
        #ignored_test_datapoints = result_dict["ignored_test_samples"]
        if self.args.use_confidence:
            if self.args.f1_calib:
                decision_thresholds = f1_calibration_thresholds
            else:
                decision_thresholds = None
            hp_scores = 1 - metrics.get_confidence(prediction_probs, decision_thresholds)
            hp_scores_train = 1 - metrics.get_confidence(prediction_probs_train, decision_thresholds)
            if hp_scores_test is not None:
                hp_scores_test = 1 - metrics.get_confidence(prediction_probs_test, decision_thresholds)
            hp_scores_val = 1 - metrics.get_confidence(prediction_probs_val, decision_thresholds)
            print("Using confidence scores as HP-scores. ")
        if args.f1_calib:

            predictions = prediction_probs > f1_calibration_thresholds
        else:
            predictions = prediction_probs > 0.5

        num_datapoints = labels.shape[0]
        num_attributes = labels.shape[1]

        if ignored_test_datapoints is not None:
            print("Ignoring the {:.0%} hardest of testing examples. ".format(ignored_test_datapoints.mean()))
        """
        attribute_hp_scores = hp_scores.mean(0)
        if args.reject_hard_attributes_quantile > 0:
            assert args.reject_hard_attributes_quantile <= 1
            num_reject = int(num_attributes * args.reject_hard_attributes_quantile)
            sorted_idxs = attribute_hp_scores.argsort()
            hard_idxs = sorted_idxs[-num_reject:]
            ignored_attributes = np.zeros((num_attributes,), dtype="int8")
            ignored_attributes[hard_idxs] = 1
        elif args.reject_harder_than < 1:
            ignored_attributes = attribute_hp_scores > args.reject_hard_attributes_threshold
        else:
            ignored_attributes = None
        if ignored_attributes is not None:
            print("Ignoring attributes: " + str(np.array(attributes)[ignored_attributes.astype("bool")]))
        """
        #ignored_attributes = None
        acc_atts = metrics.mean_attribute_accuracies(predictions, labels, ignore=ignored_test_datapoints)
        average_precision = metrics.hp_average_precision(labels, predictions, hp_scores)
        mean_average_precision = metrics.hp_mean_average_precision(labels, predictions, hp_scores)
        print('Results ----------')
        #if ignored_attributes is None:
        print(metrics.get_metrics_table(predictions, labels, ignore=ignored_test_datapoints))
        """
        else:
            selected_attributes = np.logical_not(ignored_attributes)
            print(metrics.get_metrics_table(
                predictions[:, selected_attributes],
                labels[:, selected_attributes],
                ignore=None if ignored_test_datapoints is None else ignored_test_datapoints[:, selected_attributes]))
        """
        print('------------------')
        print('Mean Attribute Accuracies:')
        header = ["Attribute", "Accuracy", "Positivity Ratio", "Average Precision", "Mean Average Precision"]
        table = tab.tabulate(zip(attributes, acc_atts, positivity_ratio, average_precision, mean_average_precision),
                             floatfmt='.2%', headers=header)
        print(table)
        print("Mean over all attributes of mean attribute accuracy of label prediction: {:.2%}".format(acc_atts.mean()))
        print("Mean average precision of hardness prediction over all attributes: {:.2%}".format(average_precision.mean()))
        print('------------------')
        if args.plot_acc_hp or args.plot_hp_hist or args.plot_pos_hp or args.num_save_hard + args.num_save_easy > 0:

            #att_idx = attributes.index(args.hard_att)
            selected_attributes = args.select_atts
            print("Analyzing attributes: " + str(selected_attributes))
            att_idxs = [attributes.index(att) for att in selected_attributes]

            hard_att_labels = labels[:, att_idxs]
            hard_att_pred = predictions[:, att_idxs]
            hard_att_prob = prediction_probs[:, att_idxs]
            if not loaded_args.hp_net_simple:
                # If a valid attribute is given, the hardness scores for that attribute are selected, else the mean
                # over all attributes is taken.
                hp_scores = hp_scores[:, att_idxs]
                hp_scores_train = hp_scores_train[:, att_idxs]
                hp_scores_val = hp_scores_val[:, att_idxs]
                if hp_scores_test is not None:
                    hp_scores_test = hp_scores_test[:, att_idxs]
                    #print(hp_scores_val.shape)
                    #print(hp_scores_test.shape)
                    #print(hp_scores_train.shape)
                    #print(hp_scores_test[:hp_scores_val.shape[0], :].shape)
                    #print((hp_scores_test - hp_scores_val[:hp_scores_test.shape[0], :]).mean())

        if args.plot_acc_hp:
            filename = osp.join(args.save_experiment, ts + "accuracy-over-hardness")
            #title = "Mean Accuracy over hardness"  # for " + (args.load_weights if args.load_weights else ts)

            plot.show_accuracy_over_hardness(filename, selected_attributes, hard_att_labels, hard_att_pred,
                                             hp_scores, metric=args.plot_metric, save_plot=self.args.save_plot)

        if args.plot_pos_hp:
            filename = osp.join(args.save_experiment, ts + "positivity-over-hardness")
            #title = "Positivity Rate over hardness"  # for " + (args.load_weights if args.load_weights else ts)

            plot.show_positivity_over_hardness(filename, selected_attributes, hard_att_labels, hard_att_pred,
                                               hp_scores, save_plot=self.args.save_plot)

        if args.plot_pos_atts:
            filename = osp.join(args.save_experiment, ts + "positivity-ratio")
            #title = "Positivity Rate over Attributes"  # for " + (args.load_weights if args.load_weights else ts)

            plot.plot_positivity_ratio_over_attributes(attributes, positivity_ratio, filename,
                                                       save_plot=self.args.save_plot)
        if args.plot_hp_hist:
            filename = osp.join(args.save_experiment, ts + "hardness-score-distribution")

            plot.plot_hardness_score_distribution(filename, selected_attributes,
                                                  hp_scores_train, hp_scores_val, hp_scores_test, args.plot_x_max,
                                                  save_plot=self.args.save_plot, confidnece=self.args.use_confidence)

        if args.num_save_hard + args.num_save_easy > 0 or args.show_example_imgs:
            print('Initializing image data manager')
            dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))

        if args.show_example_imgs:
            filename = osp.join(args.save_experiment, ts + "example_images.png")
            plot.show_example_imgs(dm.split_dict[split], filename, labels, attributes, save_plot=self.args.save_plot)

        if args.num_save_hard + args.num_save_easy > 0:
            assert len(self.args.select_atts) == 1
            # This part only gets executed if the corresponding arguments are passed at the terminal.
            print('Initializing image data manager')
            dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
            hp_scores = hp_scores.flatten()
            hard_att_labels = hard_att_labels.flatten()
            sorted_idxs = hp_scores.argsort()
            if args.show_pos_samples:
                sorted_idxs = sorted_idxs[hard_att_labels[sorted_idxs]]
            elif args.show_neg_samples:
                sorted_idxs = sorted_idxs[np.logical_not(hard_att_labels[sorted_idxs])]
            # Select easy and hard examples as specified in the terminal.
            if args.num_save_hard <= 0:
                hard_idxs = sorted_idxs[0:args.num_save_easy]
            else:
                hard_idxs = np.concatenate((sorted_idxs[0:args.num_save_easy], sorted_idxs[-args.num_save_hard:]))
            filename = osp.join(args.save_experiment,  ts + "hard_images.png")
            title = "Examples by hardness for " + (args.load_weights if args.load_weights else ts)
            # Display the image examples.
            plot.show_img_grid(dm.split_dict[split], hard_idxs, filename, hp_scores[hard_idxs],
                               save_plot=self.args.save_plot)



if __name__ == '__main__':
    # global variables
    parser = argument_parser()
    args = parser.parse_args()
    rpa = RealisticPredictorAnalyzer(args)
