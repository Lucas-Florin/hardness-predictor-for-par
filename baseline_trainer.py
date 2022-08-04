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
        print('Initializing model: {}'.format(self.args.model))
        self.model = models.init_model(name=self.args.model, num_classes=self.dm.num_attributes,
                                       pretrained=not self.args.no_pretrained,
                                       image_size=(self.args.height, self.args.width))
        if self.args.verbose: print('Model size: {:.3f} M'.format(count_num_param(self.model)))

        # Load pretrained weights if specified in args.
        self.loaded_args = self.args
        load_file = osp.join(self.args.load_weights)
        discarded_layers = None
        if self.args.load_weights:
            # TODO: implement result dict
            if check_isfile(load_file):
                checkpoint, discarded_layers = load_pretrained_weights([self.model], load_file, return_discarded_layers=True, 
                                                                       verbose=self.args.verbose, device=self.device)
                if "args" in checkpoint:
                    self.loaded_args = checkpoint["args"]
                else:
                    if self.args.verbose: print("WARNING: Could not load args. ")
            else:
                print("WARNING: Could not load pretrained weights")

        # Load model onto GPU if GPU is used.
        self.model = self.model.to(self.device)

        # Select Loss function.
        if self.args.loss_func == "deepmar":
            pos_ratio = self.dm.dataset.get_positive_attribute_ratio()
            self.criterion = DeepMARLoss(pos_ratio, self.args.train_batch_size, device=self.device,
                                         sigma=self.args.loss_func_param)
        elif self.args.loss_func == "scel":
            self.criterion = SigmoidCrossEntropyLoss(num_classes=self.dm.num_attributes)
        elif self.args.loss_func == "sscel":
            attribute_grouping = self.dm.dataset.attribute_grouping
            self.criterion = SplitSoftmaxCrossEntropyLoss(attribute_grouping)
        else:
            self.criterion = None

        self.f1_calibration_thresholds = None
        
        self.optimizer = init_optimizer(self.model, 
                                        unmatched_parameters=discarded_layers if self.args.unmatched_params_are_fresh else None,
                                        **optimizer_kwargs(self.args))
        self.scheduler = init_lr_scheduler(
            self.optimizer, steps_per_epoch=len(self.trainloader), **lr_scheduler_kwargs(self.args))

        self.model = nn.DataParallel(self.model, device_ids=[int(i) for i in self.gpu_devices]) if self.use_gpu and len(self.gpu_devices) > 1 else self.model

        # Set default for max_epoch if it was not passed as an argument in the console.
        if self.args.max_epoch < 0:
            self.args.max_epoch = 30

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
        losses_list = list()

        self.model.train()

        for batch_idx, (imgs, labels, _) in enumerate(self.trainloader):

            imgs, labels = imgs.to(self.device), labels.to(self.device)

            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad_norm)
            self.optimizer.step()
            if self.args.lr_scheduler == '1cycle':
                self.scheduler.step()

            losses.update(loss.detach().item(), labels.size(0))
            losses_list.append(loss.detach().item())

            if (batch_idx + 1) % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss: {3:.4f}\t'
                      'LR: {lr}'.format(
                    self.epoch + 1, batch_idx + 1, len(self.trainloader),
                    np.mean(losses_list), lr=self.scheduler.get_last_lr()
                ))
                losses_list = list()
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
