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


# TODO: Modularize.


class BaselineTrainer(object):
    """
    Trainer for a baseline.
    """
    def __init__(self, args):
        """
        Run the trainer.
        :param args: Command line args.
        """
        self.args = args
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

        print('Initializing image data manager')
        self.dm = ImageDataManager(self.use_gpu, **image_dataset_kwargs(args))
        self.trainloader, self.testloader_dict = self.dm.return_dataloaders()

        print('Initializing model: {}'.format(args.model))
        self.model = models.init_model(name=self.args.model, num_classes=self.dm.num_attributes, loss={'xent'},
                                       pretrained=not self.args.no_pretrained, use_gpu=self.use_gpu)
        print('Model size: {:.3f} M'.format(count_num_param(self.model)))

        # Load pretrained weights if specified in args.
        load_file = osp.join(args.save_experiment, args.load_weights)
        if args.load_weights:
            if check_isfile(load_file):
                load_pretrained_weights(self.model, load_file)
            else:
                print("WARNING: Could not load pretraining weights")

        # Load model onto GPU if GPU is used.
        self.model = nn.DataParallel(self.model).cuda() if self.use_gpu else self.model

        # Select Loss function.
        if args.loss_func == "deepmar":
            pos_ratio = self.dm.dataset.get_positive_attribute_ratio()
            self.criterion = DeepMARLoss(pos_ratio, args.train_batch_size, use_gpu=self.use_gpu,
                                         sigma=args.loss_func_param)
        elif args.loss_func == "scel":
            self.criterion = SigmoidCrossEntropyLoss(num_classes=self.dm.num_attributes, use_gpu=self.use_gpu)
        elif args.loss_func == "sscel":
            attribute_grouping = self.dm.dataset.attribute_grouping
            self.criterion = SplitSoftmaxCrossEntropyLoss(attribute_grouping, use_gpu=self.use_gpu)
        else:
            self.criterion = None

        self.optimizer = init_optimizer(self.model, **optimizer_kwargs(args))
        self.scheduler = init_lr_scheduler(self.optimizer, **lr_scheduler_kwargs(args))

        # if args.resume and check_isfile(args.resume):
        #    args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

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
        self.epoch_losses = np.zeros(shape=(args.max_epoch, ))
        self.epoch = 0

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

        # Train non-fixbase epochs.
        for epoch in range(args.start_epoch, args.max_epoch):
            loss = self.train()

            self.epoch_losses[epoch] = loss
            self.scheduler.step()

            if (epoch + 1) > self.args.start_eval and self.args.eval_freq > 0 \
                    and (epoch + 1) % self.args.eval_freq == 0 or (epoch + 1) == self.args.max_epoch:

                print('=> Evaluating {} on {} ...'.format(args.dataset_name, self.args.eval_split))
                acc, acc_atts = self.test()
                self.ranklogger.write(epoch + 1, acc)
                filename = self.ts + 'checkpoint.pth.tar'
                save_checkpoint({
                    'state_dict': self.model.state_dict(),
                    'acc': acc,
                    'acc_atts': acc_atts,
                    'epoch': epoch + 1,
                    'model': self.args.model,
                    'optimizer': self.optimizer.state_dict(),
                    'losses': self.epoch_losses,
                    'args': self.args
                }, osp.join(self.args.save_experiment, filename))
                print("Saved model checkpoint at " + filename)
            self.epoch += 1

        # Calculate elapsed time.
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Training Time {}'.format(elapsed))
        self.ranklogger.show_summary()

        if args.plot_epoch_loss:
            # Plot loss over epochs.
            plot_epoch_losses(self.epoch_losses, self.args.save_experiment, self.ts)

    def train(self, fixbase=False):
        """
        Train the model for an epoch.
        :param fixbase: Is this a fixbase epoch?
        :return: Time of execution end.
        """
        losses = AverageMeter()
        accs = AverageMeter()
        accs_atts = AverageMeter()

        self.model.train()

        if fixbase or self.args.always_fixbase:
            open_specified_layers(self.model, self.args.open_layers)
        else:
            open_all_layers(self.model)

        for batch_idx, (imgs, labels, _) in enumerate(self.trainloader):

            if self.use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), labels.size(0))
            acc, acc_atts = accuracy(self.criterion.logits(outputs), labels)
            accs.update(acc)
            accs_atts.update(acc_atts)

            if (batch_idx + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t' +
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' +
                      'Acc {acc.val:.2%} ({acc.avg:.2%})'.format(
                          self.epoch + 1, batch_idx + 1, len(self.trainloader),
                          loss=losses,
                          acc=accs
                      ))
        return losses.avg

    def test(self, loader=None):
        """

        :param loader:
        :return:
        """
        if loader is None:
            loader = self.testloader_dict[self.args.eval_split]
        f1_calibration_thresholds = self.get_threshold()
        attributes = self.dm.attributes
        self.model.eval()
        with torch.no_grad():
            predictions, gt = list(), list()
            for batch_idx, (imgs, labels, _) in enumerate(loader):
                if self.use_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()

                outputs = self.model(imgs)
                outputs = self.criterion.logits(outputs)

                predictions.extend(outputs.tolist())
                gt.extend(labels.tolist())

        # compute test accuracies
        predictions = np.array(predictions)
        gt = np.array(gt, dtype="bool")
        if args.f1_calib:
            predictions = predictions > f1_calibration_thresholds
        else:
            predictions = predictions > 0.5
        if args.group_atts:
            # Each group has exactly one positive attribute.
            attribute_grouping = self.dm.dataset.attribute_grouping
            predictions = metrics.group_attributes(predictions, attribute_grouping)
            if not args.use_macc:
                attributes = self.dm.dataset.grouped_attribute_names
            print("Grouping attributes. ")
        else:
            attribute_grouping = None
        if args.use_macc:
            # Use mA for each attribute.
            acc_atts = metrics.mean_attribute_accuracies(predictions, gt)
            acc_name = 'Mean Attribute Accuracies'
        else:
            #
            acc_atts = metrics.attribute_accuracies(predictions, gt, attribute_grouping)

            acc_name = 'Attribute Accuracies'

        print('Results ----------')
        print(metrics.get_metrics_table(predictions, gt))
        print('------------------')
        print(acc_name + ':')
        if args.f1_calib:
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

    def get_threshold(self, loader=None):
        """

        :param loader:
        :return:
        """
        if loader is None:
            loader = self.testloader_dict["train"]
        self.model.eval()
        with torch.no_grad():
            predictions, gt = list(), list()
            for batch_idx, (imgs, labels, _) in enumerate(loader):
                if self.use_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()

                outputs = self.model(imgs)
                outputs = self.criterion.logits(outputs)

                predictions.extend(outputs.tolist())
                gt.extend(labels.tolist())

        # compute test accuracies
        predictions = np.array(predictions)
        gt = np.array(gt, dtype="bool")
        return metrics.get_f1_calibration_thresholds(predictions, gt)


if __name__ == '__main__':
    # global variables
    parser = argument_parser()
    args = parser.parse_args()
    trainer = BaselineTrainer(args)
