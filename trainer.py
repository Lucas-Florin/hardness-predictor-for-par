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
from utils.avgmeter import AverageMeter
from utils.loggers import Logger, AccLogger
from utils.torchtools import count_num_param, open_all_layers, open_specified_layers, accuracy, load_pretrained_weights
from utils.generaltools import set_random_seed
import evaluation.metrics as metrics
from training.optimizers import init_optimizer
from training.lr_schedulers import init_lr_scheduler
from utils.plot import plot_epoch_losses
import tabulate as tab




class Trainer(object):
    """
    Trainer for a baseline.
    """
    def __init__(self, args):
        """
        Run the trainer.
        :param args: Command line args.
        """
        self.args = args
        self.init_environment(args)

        self.init_data()

        self.init_model()

        if args.evaluate:
            print('Evaluate only')
            split = args.eval_split
            print('=> Evaluating {} on {} ...'.format(args.dataset_name, split))
            self.test()

            # Calculate testing time.
            elapsed = round(time.time() - self.time_start)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print('Testing Time: {}'.format(elapsed))
            return
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Loading Time: {}'.format(elapsed))
        self.time_start = time.time()
        self.ranklogger = AccLogger()
        print('=> Start training')
        self.epoch_losses = np.zeros(shape=(args.max_epoch, len(self.criterion_list)))
        self.epoch = 0
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
            loss = self.train()

            self.epoch_losses[epoch, :] = loss
            for scheduler in self.scheduler_list:
                scheduler.step()

            if (epoch + 1) > self.args.start_eval and self.args.eval_freq > 0 \
                    and (epoch + 1) % self.args.eval_freq == 0 or (epoch + 1) == self.args.max_epoch:
                self.checkpoint()
                print('=> Evaluating {} on {} ...'.format(args.dataset_name, self.args.eval_split))
                acc, acc_atts = self.test()
                self.ranklogger.write(epoch + 1, acc)

            self.epoch += 1

        # Calculate elapsed time.
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Training Time {}'.format(elapsed))
        self.ranklogger.show_summary()

        if args.plot_epoch_loss:
            # Plot loss over epochs.
            plot_epoch_losses(self.epoch_losses, self.args.save_experiment, self.ts)

    def checkpoint(self):
        filename = self.ts + 'checkpoint.pth.tar'
        save_checkpoint({
            'state_dicts': [model.state_dict() for model in self.model_list],
            'epoch': self.epoch + 1,
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizer_list],
            'losses': self.epoch_losses,
            'args': self.args,
            'ts': self.ts
        }, osp.join(self.args.save_experiment, filename))
        print("Saved model checkpoint at " + filename)

    def init_environment(self, args):

        self.time_start = time.time()
        set_random_seed(args.seed)

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

    def init_model(self):
        raise NotImplementedError()

    def train(self, fixbase=False):
        raise NotImplementedError()

    def test(self, predictions=None, ground_truth=None):
        """
        Test the model. If predictions or ground truth are not given they are calculated based on the split defined in
        the command line arguments.
        :param predictions: (Optional) the predictions to be tested.
        :param ground_truth: (Optional) the ground truth of the predictions.
        :return:
        """

        f1_calibration_thresholds = None
        attributes = self.dm.attributes
        if predictions is None or ground_truth is None:
            standard_predictions, standard_ground_truth, _ = self.get_full_output()
            if predictions is None:
                predictions = standard_predictions
            if ground_truth is None:
                ground_truth = standard_ground_truth

        # compute test accuracies
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth, dtype="bool")
        if self.args.f1_calib:
            f1_calibration_thresholds = self.get_f1_calibration_threshold()
            predictions = predictions > f1_calibration_thresholds
        else:
            predictions = predictions > 0.5
        if self.args.group_atts:
            # Each group has exactly one positive attribute.
            attribute_grouping = self.dm.dataset.attribute_grouping
            predictions = metrics.group_attributes(predictions, attribute_grouping)
            if not self.args.use_macc:
                attributes = self.dm.dataset.grouped_attribute_names
            print("Grouping attributes. ")
        else:
            attribute_grouping = None
        if self.args.use_macc:
            # Use mA for each attribute.
            acc_atts = metrics.mean_attribute_accuracies(predictions, ground_truth)
            acc_name = 'Mean Attribute Accuracies'
        else:
            #
            acc_atts = metrics.attribute_accuracies(predictions, ground_truth, attribute_grouping)

            acc_name = 'Attribute Accuracies'

        print('Results ----------')
        print(metrics.get_metrics_table(predictions, ground_truth))
        print('------------------')
        print(acc_name + ':')
        if self.args.f1_calib and not self.args.group_atts:
            header = ["Attribute", "Accuracy", "F1-Calibration Threshold"]
            table = tab.tabulate(zip(attributes, acc_atts, f1_calibration_thresholds.flatten()), floatfmt='.2%',
                                 headers=header)
        else:
            header = ["Attribute", "Accuracy"]
            table = tab.tabulate(zip(attributes, acc_atts), floatfmt='.2%', headers=header)
        print(table)
        print("Mean over attributes: {:.2%}".format(acc_atts.mean()))
        print('------------------')

        return acc_atts.mean(), acc_atts

    def get_f1_calibration_threshold(self, loader=None):
        """

        :param loader:
        :return:
        """
        if loader is None:
            loader = self.testloader_dict["train"]
        predictions, gt, _ = self.get_full_output(loader)

        # compute test accuracies
        predictions = np.array(predictions)
        gt = np.array(gt, dtype="bool")
        return metrics.get_f1_calibration_thresholds(predictions, gt)

    def get_full_output(self, loader=None, model=None, criterion=None):
        """
        Get the output of the model for all the datapoints in the loader.
        :param loader: (Optional) The loader to be used as input. Default: the split specified in the command line args.
        :return: The predictions made by the model and the ground truth as lists.
        """
        if loader is None:
            loader = self.testloader_dict[self.args.eval_split]
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
        return predictions, ground_truth, imgs_path_list


