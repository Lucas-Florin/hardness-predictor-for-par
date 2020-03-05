import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from data.data_manager import ImageDataManager
import models
from training.losses import SigmoidCrossEntropyLoss, DeepMARLoss, SplitSoftmaxCrossEntropyLoss
from utils.iotools import check_isfile, save_checkpoint
from utils.loggers import Logger, AccLogger
from utils.torchtools import count_num_param, open_all_layers, open_specified_layers, accuracy, load_pretrained_weights
import evaluation.metrics as metrics
from training.optimizers import init_optimizer
from training.lr_schedulers import init_lr_scheduler
from utils.plot import plot_epoch_losses
import tabulate as tab
from evaluation.result_manager import ResultManager

# For all Trainers:
# TODO: Console output
# TODO: Documentation
# TODO: Machine-Friendly results saving (csv)


class Trainer(object):
    """
    Trainer for a baseline.
    """
    def __init__(self, args):
        """
        Run the trainer.
        :param args: Command line args.
        """
        # TODO: only init data and model if necessary.
        self.args = args
        self.time_start = time.time()
        self.init_environment(args)

        self.init_data()
        self.result_dict = dict()
        self.result_manager = ResultManager(self.result_dict)
        self.init_model()
        self.epoch = 0
        if args.evaluate:
            print('Evaluate only')
            split = args.eval_split
            print('=> Evaluating {} on {} ...'.format(args.dataset_name, split))
            self.test()

            # Calculate testing time.
            elapsed = round(time.time() - self.time_start)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print('Testing Time: {}'.format(elapsed))
            if args.save_checkpoint:
                self.checkpoint()
            return
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Loading Time: {}'.format(elapsed))
        self.time_start = time.time()
        self.ranklogger = AccLogger()
        print('=> Start training')
        self.epoch_losses = np.zeros(shape=(args.max_epoch, len(self.criterion_list)))

        """
        # Train Fixbase epochs.
        if self.args.fixbase_epoch > 0:
            print('Train {} for {} epochs while keeping other layers frozen'.format(self.args.open_layers,
                                                                                    self.args.fixbase_epoch))
            initial_optim_state = self.optimizer.state_dict()

            for epoch in range(args.fixbase_epoch):
                self.epoch_losses[epoch] = self.train(fixbase=True)
                self.epoch += 1

            print('Done. All layers are open to train for {} epochs'.format(args.max_epoch))
            self.optimizer.load_state_dict(initial_optim_state)
        """
        # Train non-fixbase epochs.
        for epoch in range(args.start_epoch, args.max_epoch):
            print(self.scheduler.get_lr())
            loss = self.train()
            self.epoch_losses[epoch, :] = loss
            for scheduler in self.scheduler_list:
                scheduler.step()
            if (epoch + 1) > self.args.start_eval and self.args.eval_freq > 0 \
                    and (epoch + 1) % self.args.eval_freq == 0 or (epoch + 1) == self.args.max_epoch:
                self.checkpoint()
                print('=> Evaluating {} on {} ...'.format(args.dataset_name, self.args.eval_split))
                acc, _, _ = self.test()
                self.ranklogger.write(epoch + 1, acc)
                self.checkpoint()

            self.epoch += 1

        if self.args.max_epoch == 0:
            self.checkpoint()
            print('=> Evaluating {} on {} ...'.format(args.dataset_name, self.args.eval_split))
            self.test()
            self.checkpoint()

        # Calculate elapsed time.
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Training Time {}'.format(elapsed))
        self.ranklogger.show_summary()

        if args.plot_epoch_loss:
            # Plot loss over epochs.
            plot_epoch_losses(self.epoch_losses, self.args.save_experiment, self.ts)

    def checkpoint(self, ts=None):
        if ts is None:
            ts = self.ts
        filename = ts + 'checkpoint.pth.tar'
        save_checkpoint({
            'state_dicts': [model.state_dict() for model in self.model_list],
            'epoch': self.epoch + 1 if not self.args.evaluate else None,
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizer_list],
            'losses': self.epoch_losses if not self.args.evaluate else None,
            'args': self.loaded_args if self.args.evaluate else self.args,
            'ts': self.ts,
            'result_dict': self.result_dict
        }, osp.join(self.args.save_experiment, filename))
        print("Saved model checkpoint at " + filename)


    def init_environment(self, args):
        # Decide which processor (CPU or GPU) to use.
        if not args.use_avai_gpus:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

        self.use_gpu = torch.cuda.is_available()
        if args.use_cpu:
            self.use_gpu = False

        # Start logger.
        self.ts = time.strftime("%Y-%m-%d_%H-%M-%S_")

        self.log_name = self.ts + 'test' + '.log' if args.evaluate else self.ts + 'train' + '.log'
        sys.stdout = Logger(osp.join(args.save_experiment, self.log_name))

        # Print out the arguments taken from Terminal (or defaults).
        print('==========\nArgs:{}\n=========='.format(args))

        print("Timestamp: " + self.ts)
        # Warn if not using GPU.
        if self.use_gpu:
            print('Currently using GPU {}'.format(args.gpu_devices))
            cudnn.benchmark = True
        else:
            warnings.warn('Currently using CPU, however, GPU is highly recommended')

    def init_data(self):
        print('Initializing image data manager')
        self.dm = ImageDataManager(self.use_gpu, **image_dataset_kwargs(self.args))
        self.trainloader, self.testloader_dict = self.dm.return_dataloaders()
        self.attributes = self.dm.attributes
        if self.args.group_atts:
            # Each group has exactly one positive attribute.
            self.attribute_grouping = self.dm.dataset.attribute_grouping
            if not self.args.use_macc:
                self.attributes = self.dm.dataset.grouped_attribute_names
        else:
            self.attribute_grouping = None

    def init_model(self):
        raise NotImplementedError()

    def train(self, fixbase=False):
        raise NotImplementedError()

    def test(self, ignore=None):
        split = self.args.eval_split
        """
        Test the model. If predictions or ground truth are not given they are calculated based on the split defined in
        the command line arguments.
        :return:
        """
        labels, prediction_probs, predictions = self.get_label_predictions(self.args.eval_split)
        if self.args.use_macc:
            # TODO: Make this the default option.
            # Use mA for each attribute.
            acc_atts = metrics.mean_attribute_accuracies(predictions, labels, ignore)
            acc_name = 'Mean Attribute Accuracies'
        else:
            #
            acc_atts = metrics.attribute_accuracies(predictions, labels, self.attribute_grouping)

            acc_name = 'Attribute Accuracies'
        self.acc_atts = acc_atts
        positivity_ratio = self.dm.dataset.get_positive_attribute_ratio()
        self.positivity_ratio = positivity_ratio
        print('Results ----------')
        print(metrics.get_metrics_table(predictions, labels, ignore))
        print('------------------')
        print(acc_name + ':')
        if self.args.f1_calib and not self.args.group_atts:
            header = ["Attribute", "Accuracy", "Positivity Ratio", "F1-Calibration Threshold"]
            table = tab.tabulate(zip(self.attributes, acc_atts, positivity_ratio,
                                     self.f1_calibration_thresholds.flatten()),
                                 floatfmt='.2%', headers=header)
        else:
            header = ["Attribute", "Accuracy", "Positivity Ratio"]
            table = tab.tabulate(zip(self.attributes, acc_atts, positivity_ratio), floatfmt='.2%', headers=header)
        print(table)
        print("Mean over attributes: {:.2%}".format(acc_atts.mean()))
        print('------------------')
        self.result_dict.update({

            "args": self.loaded_args if self.args.evaluate else self.args,
            "attributes": self.dm.attributes,
            "f1_thresholds": self.f1_calibration_thresholds,
            "positivity_ratio": positivity_ratio

        })
        self.result_manager.update_outputs(split, prediction_probs=prediction_probs, labels=labels,
                                           predictions=predictions)
        return acc_atts.mean(), prediction_probs, predictions

    def init_f1_calibration_threshold(self):
        if self.f1_calibration_thresholds is not None:
            return
        if self.args.evaluate and "f1_thresholds" in self.result_dict:
            self.f1_calibration_thresholds = self.result_dict["f1_thresholds"]
        else:
            self.f1_calibration_thresholds = self.get_f1_calibration_threshold()

    def get_f1_calibration_threshold(self):
        """

        :param loader:
        :return:
        """
        print("Computing F1-calibration thresholds on " + self.args.f1_calib_split)
        # TODO: ignore rejected samples?

        predictions, gt, _ = self.get_full_output(split=self.args.f1_calib_split)

        return metrics.get_f1_calibration_thresholds(predictions, gt)

    def get_full_output(self, loader=None, model=None, criterion=None, split=None):
        """
        Get the output of the model for all the datapoints in the loader.
        :param loader: (Optional) The loader to be used as input. Default: the split specified in the command line args.
        :return: The predictions made by the model and the ground truth as lists.
        """
        if split is None:
            split = self.args.eval_split
        if loader is None:
            loader = self.testloader_dict[split]
        if model is None:
            model = self.model
        if criterion is None:
            criterion = self.criterion
        model.eval()
        with torch.no_grad():
            predictions, ground_truth, imgs_path_list = list(), list(), list()
            for batch_idx, (imgs, labels, img_paths) in enumerate(loader):
                if self.use_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()

                outputs = model(imgs)
                outputs = criterion.logits(outputs)
                predictions.extend(outputs.tolist())
                ground_truth.extend(labels.tolist())
                imgs_path_list.extend(img_paths)
        return np.array(predictions), np.array(ground_truth, dtype="bool"), imgs_path_list

    def get_label_predictions(self, split):
        self.init_f1_calibration_threshold()

        if self.args.evaluate and self.result_manager.check_output_dict(split):
            labels, prediction_probs, _, _ = self.result_manager.get_outputs(split)
        else:
            print("Computing label predictions. ")
            prediction_probs, labels, _ = self.get_full_output(split=split)
            self.result_manager.update_outputs(split, prediction_probs=prediction_probs, labels=labels)
        if self.args.f1_calib:
            label_predictions = prediction_probs > self.f1_calibration_thresholds
        else:
            label_predictions = prediction_probs > 0.5
        if self.args.group_atts:
            # Each group has exactly one positive attribute.
            label_predictions = metrics.group_attributes(label_predictions, self.attribute_grouping)
            print("Grouping attributes. ")
        return labels, prediction_probs, label_predictions
