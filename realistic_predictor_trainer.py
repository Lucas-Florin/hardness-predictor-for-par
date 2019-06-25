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
from training.losses import SigmoidCrossEntropyLoss, HardnessPredictorLoss
from utils.iotools import check_isfile, save_checkpoint
from utils.avgmeter import AverageMeter
from utils.loggers import Logger, AccLogger
from utils.torchtools import count_num_param, open_all_layers, open_specified_layers, accuracy, load_pretrained_weights
from utils.generaltools import set_random_seed
import evaluation.metrics as metrics
from training.optimizers import init_optimizer
from training.lr_schedulers import init_lr_scheduler
from utils.plot import plot_epoch_losses
from trainer import Trainer


class RealisticPredictorTrainer(Trainer):
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
        print('Initializing main model: {}'.format(args.model))
        self.model_main = models.init_model(name=self.args.model, num_classes=self.dm.num_attributes, loss={'xent'},
                                            pretrained=not self.args.no_pretrained, use_gpu=self.use_gpu)
        print('Model size: {:.3f} M'.format(count_num_param(self.model_main)))

        print('Initializing HP Net. ')
        # Determine the size of the output vector for the HP-Net.
        num_hp_net_outputs = 1 if self.args.hp_net_simple else self.dm.num_attributes
        # Init the HP-Net
        self.model_hp = models.init_model(name="hp_net", num_classes=num_hp_net_outputs)
        print('Model size: {:.3f} M'.format(count_num_param(self.model_hp)))

        # Load pretrained weights if specified in args.
        load_file = osp.join(args.save_experiment, args.load_weights)
        if args.load_weights:
            if check_isfile(load_file):
                load_pretrained_weights(self.model_main, load_file)
            else:
                print("WARNING: Could not load pretraining weights")

        # Load model onto GPU if GPU is used.
        self.model_main = nn.DataParallel(self.model_main).cuda() if self.use_gpu else self.model_main
        self.model = self.model_main
        self.model_hp = nn.DataParallel(self.model_hp).cuda() if self.use_gpu else self.model_hp


        # Select Loss function.
        self.criterion_main = SigmoidCrossEntropyLoss(use_gpu=self.use_gpu)
        self.criterion = self.criterion_main
        self.criterion_hp = HardnessPredictorLoss()

        # TODO: SGD or Adam? (Paper uses both)
        self.optimizer_main = init_optimizer(self.model_main, **optimizer_kwargs(args))
        self.scheduler_main = init_lr_scheduler(self.optimizer_main, **lr_scheduler_kwargs(args))

        # TODO: make like original paper?
        op_args = optimizer_kwargs(args)
        op_args['lr'] *= op_args['base_lr_mult']
        self.optimizer_hp = init_optimizer(self.model_hp, **op_args)
        self.scheduler_hp = init_lr_scheduler(self.optimizer_hp, **lr_scheduler_kwargs(args))

        self.model_list = [self.model_main, self.model_hp]
        self.optimizer_list = [self.optimizer_main, self.optimizer_hp]
        self.scheduler_list = [self.scheduler_main, self.scheduler_hp]
        self.criterion_list = [self.criterion_main, self.criterion_hp]

        # if args.resume and check_isfile(args.resume):
        #    args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

    def train(self, fixbase=False):
        """
        Train the model for an epoch.
        :param fixbase: Is this a fixbase epoch?
        :return: Time of execution end.
        """
        losses_main = AverageMeter()
        losses_hp = AverageMeter()

        self.model_main.train()
        self.model_hp.train()

        # TODO: Adapt fixbase.
        if fixbase or self.args.always_fixbase:
            open_specified_layers(self.model_main, self.args.open_layers)
            open_specified_layers(self.model_hp, self.args.open_layers)
        else:
            open_all_layers(self.model_main)
            open_all_layers(self.model_hp)
        for batch_idx, (imgs, labels, _) in enumerate(self.trainloader):

            if self.use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            # Run the batch through both nets.
            label_predicitons = self.model_main(imgs)
            hardness_predictions = self.model_hp(imgs)

            # Make a detached version of the hp scores for computing the main loss.
            hardness_predictions_logits = self.criterion_hp.logits(hardness_predictions.detach())
            # Compute main loss, gradient and optimize main net.
            loss_main = self.criterion_main(label_predicitons, labels, hardness_predictions_logits)
            self.optimizer_main.zero_grad()
            loss_main.backward()
            self.optimizer_main.step()

            losses_main.update(loss_main.item(), labels.size(0))

            # Compute HP loss, gradient and optimize HP net.
            label_predicitons_logits = self.criterion_main.logits(label_predicitons.detach())
            loss_hp = self.criterion_hp(hardness_predictions, label_predicitons_logits)

            self.optimizer_hp.zero_grad()
            loss_hp.backward()
            self.optimizer_hp.step()

            losses_hp.update(loss_hp.item(), labels.size(0))

            # Print progress.
            if (batch_idx + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          self.epoch + 1, batch_idx + 1, len(self.trainloader),
                          loss=losses_main
                      ))
        return losses_main.avg, losses_hp.avg


if __name__ == '__main__':
    # global variables
    parser = argument_parser()
    args = parser.parse_args()
    trainer = RealisticPredictorTrainer(args)
