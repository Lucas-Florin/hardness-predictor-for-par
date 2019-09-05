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
from utils.plot import plot_epoch_losses, show_img_grid
from trainer import Trainer
import evaluation.rejectors as rejectors


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

        if self.args.rejector == "none":
            self.rejector = rejectors.NoneRejector()
        elif self.args.rejector == 'macc':
            self.rejector = rejectors.MeanAccuracyRejector(self.args.max_rejection_quantile)
        elif self.args.rejector == "median":
            self.rejector = rejectors.MedianRejector(self.args.max_rejection_quantile)
        elif self.args.rejector == "threshold":
            self.rejector = rejectors.ThresholdRejector(self.args.rejection_threshold, self.args.max_rejection_quantile)
        elif self.args.rejector == "quantile":
            self.rejector = rejectors.QuantileRejector(self.args.max_rejection_quantile)
        else:
            self.rejector = None

        # Load pretrained weights if specified in args.
        load_file = osp.join(args.save_experiment, args.load_weights)
        self.loaded_args = self.args
        if args.load_weights:
            if check_isfile(load_file):
                cp = load_pretrained_weights([self.model_main, self.model_hp], load_file)
                if "args" in cp:
                    self.loaded_args = cp["args"]
                else:
                    print("WARNING: Could not load args. ")
                if "rejection_thresholds" in cp:
                    self.rejector.load_thresholds(cp["rejection_thresholds"])
                else:
                    print("WARNING: Could not load rejection thresholds. ")
            else:
                print("WARNING: Could not load pretraining weights")

        # Load model onto GPU if GPU is used.
        self.model_main = nn.DataParallel(self.model_main).cuda() if self.use_gpu else self.model_main
        self.model = self.model_main
        self.model_hp = nn.DataParallel(self.model_hp).cuda() if self.use_gpu else self.model_hp

        self.pos_ratio = self.dm.dataset.get_positive_attribute_ratio()
        # Select Loss function.
        # Select Loss function.
        if args.loss_func == "deepmar":

            self.criterion = DeepMARLoss(self.pos_ratio, args.train_batch_size, use_gpu=self.use_gpu,
                                         sigma=args.loss_func_param)
        elif args.loss_func == "scel":
            self.criterion = SigmoidCrossEntropyLoss(num_classes=self.dm.num_attributes, use_gpu=self.use_gpu)
        else:
            self.criterion = None

        self.criterion_main = self.criterion
        self.criterion_hp = HardnessPredictorLoss(self.args.use_deepmar_for_hp, self.pos_ratio, use_gpu=self.use_gpu,
                                                  sigma=self.args.hp_loss_param)



        self.optimizer_main = init_optimizer(self.model_main, **optimizer_kwargs(args))
        self.scheduler_main = init_lr_scheduler(self.optimizer_main, **lr_scheduler_kwargs(args))

        op_args = optimizer_kwargs(args)
        op_args['lr'] *= op_args['base_lr_mult']
        self.optimizer_hp = init_optimizer(self.model_hp, **op_args)
        sc_args = lr_scheduler_kwargs(args)
        sc_args["stepsize"] = [i + self.args.hp_epoch_offset for i in sc_args["stepsize"]]
        self.scheduler_hp = init_lr_scheduler(self.optimizer_hp, **sc_args)

        if not self.args.evaluate:
            self.init_epochs()

        self.model_list = [self.model_main, self.model_hp]
        self.optimizer_list = [self.optimizer_main, self.optimizer_hp]
        self.scheduler_list = [self.scheduler_main, self.scheduler_hp]
        self.criterion_list = [self.criterion_main, self.criterion_hp]

        # if args.resume and check_isfile(args.resume):
        #    args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

    def update_rejector_thresholds(self):

        print("Computing hardness scores for training data. ")
        hardness_predictions, _, _ = self.get_full_output(self.trainloader, self.model_hp, self.criterion_hp)
        print("Computing label predictions for training data. ")
        labels, _, label_predictions = self.get_label_predictions(
            self.trainloader, self.model_main, self.criterion_main)
        print("Updating rejection thresholds based on training data. ")
        self.rejector.update_thresholds(labels, label_predictions, hardness_predictions)

    def init_epochs(self):
        # Initialize the epoch thresholds.
        if self.args.max_epoch < 0 and (self.args.main_net_train_epochs < 0 or self.args.hp_net_train_epochs < 0):
            raise ValueError("Neither max-epochs or not-train-epochs is defined. ")
        if self.args.main_net_train_epochs < 0:
            self.args.main_net_train_epochs = (self.args.max_epoch - self.args.hp_epoch_offset
                                               - self.args.main_net_finetuning_epochs)
        if self.args.hp_net_train_epochs < 0:
            self.args.hp_net_train_epochs = (self.args.max_epoch - self.args.hp_epoch_offset
                                             - self.args.main_net_finetuning_epochs)
        if self.args.max_epoch < 0:
            self.args.max_epoch = (max(self.args.main_net_train_epochs, self.args.hp_net_train_epochs
                                       + self.args.hp_epoch_offset) + self.args.main_net_finetuning_epochs)

    def train(self, fixbase=False):
        """
        Train the model for an epoch.
        :param fixbase: Is this a fixbase epoch?
        :return: Time of execution end.
        """
        losses_main = AverageMeter()
        losses_hp = AverageMeter()
        train_main = not self.args.train_hp_only and self.epoch < self.args.main_net_train_epochs
        train_main_finetuning = (not self.args.train_hp_only and self.epoch >= self.args.max_epoch
                                 - self.args.main_net_finetuning_epochs)
        rejection_epoch = (not self.args.train_hp_only and self.epoch == self.args.max_epoch
                           - self.args.main_net_finetuning_epochs)
        train_hp = (self.args.hp_epoch_offset <= self.epoch < self.args.hp_net_train_epochs
                    + self.args.hp_epoch_offset)

        if rejection_epoch:
            self.update_rejector_thresholds()
        if train_main:
            self.model_main.train()
            losses = losses_main
        else:
            self.model_main.eval()
            losses = losses_hp

        if train_hp:
            self.model_hp.train()
        else:
            self.model_hp.eval()

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
            label_prediciton_probs = self.model_main(imgs)
            hardness_predictions = self.model_hp(imgs)
            if train_main or train_main_finetuning:
                if self.args.no_hp_feedback or not train_hp:
                    main_net_weights = label_prediciton_probs.new_ones(label_prediciton_probs.shape)
                else:
                    # Make a detached version of the hp scores for computing the main loss.
                    main_net_weights = self.criterion_hp.logits(hardness_predictions.detach())
                if train_main_finetuning:
                    main_net_weights = main_net_weights * self.rejector(hardness_predictions.detach())
                # Compute main loss, gradient and optimize main net.
                loss_main = self.criterion_main(label_prediciton_probs, labels, main_net_weights)
                self.optimizer_main.zero_grad()
                loss_main.backward()
                self.optimizer_main.step()

                losses_main.update(loss_main.item(), labels.size(0))

            if train_hp:
                # Compute HP loss, gradient and optimize HP net.
                label_predicitons_logits = self.criterion_main.logits(label_prediciton_probs.detach())
                loss_hp = self.criterion_hp(hardness_predictions, label_predicitons_logits, labels)

                self.optimizer_hp.zero_grad()
                loss_hp.backward()
                self.optimizer_hp.step()

                losses_hp.update(loss_hp.item(), labels.size(0))

            # Print progress.
            if (batch_idx + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          self.epoch + 1, batch_idx + 1, len(self.trainloader),
                          loss=losses
                      ))
        return losses_main.avg, losses_hp.avg

    def test(self, predictions=None, ground_truth=None):
        if not self.rejector.is_initialized():
            self.update_rejector_thresholds()

        # Compute Hardness scores.
        print("Computing hardness scores. ")
        hp_scores, labels, images = self.get_full_output(model=self.model_hp, criterion=self.criterion_hp)
        #hp_scores = np.array(hp_scores)
        labels = np.array(labels, dtype="bool")
        ignore = np.logical_not(self.rejector(hp_scores))
        print("Ignoring the {:.2%} hardest of testing examples. ".format(ignore.mean()))
        # Run the standard accuracy testing.
        mean_acc = super().test(ignore)
        label_prediction_probs = self.result_dict["prediction_probs"]
        label_predictions = self.result_dict["predictions"]

        self.result_dict.update({
            "hp_scores": hp_scores,
            "rejection_thresholds": self.rejector.attribute_thresholds,
            "ignored_test_samples": ignore
        })
        pickle_path = osp.join(self.args.save_experiment, "result_dict.pickle")
        pickle_file = open(pickle_path, "wb")
        pickle.dump(self.result_dict, pickle_file)
        pickle_file.close()
        print("Saved Results at " + pickle_path)

        print("HP-Net Hardness Scores: ")
        print(tab.tabulate([
            ["Mean", np.mean(hp_scores)],
            ["Variance", np.var(hp_scores)]
        ]))
        if not self.args.hp_net_simple:
            # Display the hardness scores for every attribute.
            print("-" * 30)
            header = ["Attribute", "Hardness Score Mean", "Variance", "Average Precision", "Mean Average Precision"]
            mean = hp_scores.mean(0)
            var = hp_scores.var(0)
            average_precision = metrics.hp_average_precision(labels, label_predictions, hp_scores)
            mean_average_precision = metrics.hp_mean_average_precision(labels, label_predictions, hp_scores)
            data = list(zip(self.dm.attributes, mean, var, average_precision, mean_average_precision))
            table = tab.tabulate(data, floatfmt='.4f', headers=header)
            print(table)
            print("Mean average precision of hardness prediction over attributes: {:.2%}".format(average_precision.mean()))
            csv_path = osp.join(self.args.save_experiment, "result_table.csv")
            np.savetxt(csv_path, np.transpose(data), fmt="%s", delimiter=",")
            print("Saved Table at " + csv_path)

        hard_att_labels = None
        hard_att_pred = None
        if self.args.num_save_hard + self.args.num_save_easy > 0:
            # This part only gets executed if the corresponding arguments are passed at the terminal.
            if self.args.hard_att in self.dm.attributes:
                # If a valid attribute is given the labels for that attribute are selected.
                print("Looking at Hard attribute " + self.args.hard_att)
                att_idx = self.dm.attributes.index(self.args.hard_att)
                hard_att_labels = labels[:, att_idx]
                hard_att_pred = label_prediction_probs[:, att_idx]
            if not self.loaded_args.hp_net_simple:
                # If a valid attribute is given, the hardness scores for that attribute are selected, else the mean
                # over all attributes is taken.
                if self.args.hard_att in self.dm.attributes:
                    hp_scores = hp_scores[:, att_idx]
                else:
                    hp_scores = hp_scores.mean(1)
            hp_scores = hp_scores.flatten()
            sorted_idxs = hp_scores.argsort()
            # Select easy and hard examples as specified in the terminal.
            hard_idxs = np.concatenate((sorted_idxs[:self.args.num_save_easy],
                                        sorted_idxs[-self.args.num_save_hard:]))
            filename = osp.join(self.args.save_experiment,  self.ts + "hard_images.png")
            title = "Examples by hardness for " + (self.args.load_weights if self.args.load_weights else self.ts)
            if hard_att_labels is not None:
                hard_att_labels = hard_att_labels[hard_idxs]
            if hard_att_pred is not None:
                hard_att_pred = hard_att_pred[hard_idxs]
            # Display the image examples.
            show_img_grid(self.dm.split_dict[self.args.eval_split], hard_idxs, filename, title, self.args.hard_att,
                          hard_att_labels, hp_scores[hard_idxs], hard_att_pred)

        return mean_acc, label_prediction_probs, label_predictions  # Return the values from the super-function.



if __name__ == '__main__':
    # global variables
    parser = argument_parser()
    args = parser.parse_args()
    trainer = RealisticPredictorTrainer(args)
