import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings
import tabulate as tab

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
import evaluation.metrics as metrics
from training.optimizers import init_optimizer
from training.lr_schedulers import init_lr_scheduler
from utils.plot import plot_epoch_losses
from trainer import Trainer

# TODO: Document
# TODO: Console output
# TODO: Join with RealisticPredictorTrainer


class BaselineTrainer(Trainer):
    """
    Trainer for a baseline.
    """
    def __init__(self, args):
        """
        Run the trainer.
        :param args: Command line args.
        """
        super().__init__(args)


    def init_model(self):
        print('Initializing model: {}'.format(args.model))
        self.model = models.init_model(name=self.args.model, num_classes=self.dm.num_attributes, loss={'xent'},
                                       pretrained=self.args.pretrained, use_gpu=self.use_gpu)
        print('Model size: {:.3f} M'.format(count_num_param(self.model)))

        # Load pretrained weights if specified in args.
        self.loaded_args = self.args
        load_file = osp.join(args.save_experiment, args.load_weights)
        if args.load_weights:
            # TODO: implement result dict
            if check_isfile(load_file):
                cp = load_pretrained_weights([self.model], load_file)
                if "args" in cp:
                    self.loaded_args = cp["args"]
                else:
                    print("WARNING: Could not load args. ")
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

        self.f1_calibration_thresholds = None

        self.optimizer = init_optimizer(self.model, **optimizer_kwargs(args))
        self.scheduler = init_lr_scheduler(self.optimizer, **lr_scheduler_kwargs(args))

        # Set default for max_epoch if it was not passed as an argument in the console.
        if self.args.max_epoch < 0:
            self.args.max_epoch = 60

        self.model_list = [self.model]
        self.optimizer_list = [self.optimizer]
        self.scheduler_list = [self.scheduler]
        self.criterion_list = [self.criterion]

        # if args.resume and check_isfile(args.resume):
        #    args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

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
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.avg:.4f}'.format(
                    self.epoch + 1, batch_idx + 1, len(self.trainloader),
                    loss=losses
                ))
        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss.avg:.4f}'.format(
            self.epoch + 1, batch_idx + 1, len(self.trainloader),
            loss=losses
        ))
        return losses.avg

    def test(self, ignore=None):
        return_data = super().test(ignore)
        self.save_result_dict()
        return return_data




if __name__ == '__main__':
    # global variables
    parser = argument_parser()
    args = parser.parse_args()
    trainer = BaselineTrainer(args)
