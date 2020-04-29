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
from utils.torchtools import count_num_param, open_all_layers, open_specified_layers, accuracy, \
    load_pretrained_weights, freeze_all_layers
import evaluation.metrics as metrics
from training.optimizers import init_optimizer
from training.lr_schedulers import init_lr_scheduler
from utils.plot import plot_epoch_losses, show_img_grid
from trainer import Trainer
import evaluation.rejectors as rejectors
from evaluation.result_manager import ResultManager
from training.calibrators import NoneCalibrator, LinearCalibrator


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
                                            pretrained=self.args.pretrained, use_gpu=self.use_gpu)
        print('Model size: {:.3f} M'.format(count_num_param(self.model_main)))

        print('Initializing HP model: {}'.format(args.hp_model))
        # Determine the size of the output vector for the HP-Net.
        num_hp_net_outputs = 1 if self.args.hp_net_simple else self.dm.num_attributes
        # Init the HP-Net
        self.model_hp = models.init_model(name="hp_net_" + self.args.hp_model, num_classes=num_hp_net_outputs,
                                          pretrained=self.args.pretrained)
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
        elif self.args.rejector == 'f1':
            self.rejector = rejectors.F1Rejector(self.args.max_rejection_quantile)
        else:
            raise ValueError("Unsupported rejection strategy: '{}'".format(self.args.rejector))
        print("Using rejection strategy '{}'".format(self.args.rejector))

        if self.args.hp_calib == 'none':
            self.hp_calibrator = NoneCalibrator()
        elif self.args.hp_calib == 'linear':
            self.hp_calibrator = LinearCalibrator()
        else:
            raise ValueError("Unsupported calibrator: '{}'".format(self.args.hp_calib))
        print("Using calibrator for HP-Loss '{}'".format(self.args.hp_calib))


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

                if "result_dict" in cp and cp["result_dict"] is not None and self.args.evaluate:
                    self.result_dict = cp["result_dict"]
                    self.result_manager = ResultManager(self.result_dict)
                    print("Loaded result dict with keys: ")
                    print(sorted(list(self.result_dict.keys())))
                    if "rejection_thresholds" in self.result_dict:
                        self.rejector.load_thresholds(self.result_dict["rejection_thresholds"])
                        if self.rejector.is_initialized():
                            print("Loaded rejection thresholds. ")
                        else:
                            print("Loaded uninitialized (None) rejection thresholds. ")
                    else:
                        print("WARNING: Could not load rejection thresholds. ")
            else:
                print("WARNING: Could not load pretraining weights")
        self.new_eval_split = self.args.eval_split != self.loaded_args.eval_split
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
        self.criterion_hp = HardnessPredictorLoss(self.args.use_deepmar_for_hp, self.pos_ratio, self.dm.num_attributes,
                                                  use_gpu=self.use_gpu, sigma=self.args.hp_loss_param,
                                                  use_visibility=self.args.use_bbs,
                                                  visibility_weight=self.args.hp_visibility_weight)
        self.f1_calibration_thresholds = None


        self.optimizer_main = init_optimizer(self.model_main, **optimizer_kwargs(args))
        self.scheduler_main = init_lr_scheduler(self.optimizer_main, **lr_scheduler_kwargs(args))

        self.optimizer = self.optimizer_main
        self.scheduler = self.scheduler_main

        op_args = optimizer_kwargs(args)
        sc_args = lr_scheduler_kwargs(args)
        op_args['lr'] *= self.args.hp_net_lr_multiplier
        self.optimizer_hp = init_optimizer(self.model_hp, **op_args)
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
        split = self.args.rejector_thresholds_split
        self.init_f1_calibration_threshold()
        if self.result_manager.check_output_dict(split):
            labels, prediction_probs, predictions, _ = self.result_manager.get_outputs(split)
        else:
            print("Computing label predictions for training data. ")
            labels, prediction_probs, predictions = self.get_label_predictions(split)
            self.result_manager.update_outputs(split, prediction_probs=prediction_probs, labels=labels,
                                               predictions=predictions)
        if self.args.use_confidence:
            if self.args.f1_calib:
                decision_thresholds = self.f1_calibration_thresholds
                assert decision_thresholds is not None
            else:
                decision_thresholds = None
            hp_scores = 1 - metrics.get_confidence(prediction_probs, decision_thresholds)
            print("Using confidence scores as HP-scores. ")
        elif self.result_manager.check_output_dict(split):
            _, _, _, hp_scores = self.result_manager.get_outputs(split)
        else:
            print("Computing hardness scores for training data. ")
            hp_scores, _, _ = self.get_full_output(model=self.model_hp, criterion=self.criterion_hp, split=split)
            hp_scores = self.criterion_hp.broadcast(hp_scores)
            self.result_manager.update_outputs(split, hp_scores=hp_scores)
        print("Updating rejection thresholds based on training data. ")
        self.rejector.update_thresholds(labels, predictions, hp_scores)

    def update_hp_calibrator_thresholds(self, thresholds=None):
        if self.args.hp_calib == "none":
            return
        if self.args.hp_calib_thr == "f1":
            if self.hp_calibrator.is_initialized():
                return
            thresholds = self.get_baseline_f1_calibration_thresholds()
        elif self.args.hp_calib_thr == "mean":
            thresholds = 0.5 if thresholds is None else thresholds
        else:
            raise ValueError("Unsupported HP-Loss calibration threshold: '{}'".format(self.args.hp_calib_thr))
        self.hp_calibrator.update_thresholds(thresholds)

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
        print("Training schedule: ")
        print(tab.tabulate([
            ["Main-Net train epochs", self.args.main_net_train_epochs],
            ["HP-Net epoch offset", self.args.hp_epoch_offset],
            ["HP-Net train epochs", self.args.hp_net_train_epochs],
            ["Main-Net finetuning epochs", self.args.main_net_finetuning_epochs],
            ["Total epochs", self.args.max_epoch]
        ]))

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
        num_batch = len(self.trainloader)

        if rejection_epoch:
            self.update_rejector_thresholds()
        if self.args.hp_epoch_offset == self.epoch:
            self.update_hp_calibrator_thresholds()

        if train_main or train_main_finetuning:
            self.model_main.train()
            losses = losses_main
        else:
            self.model_main.eval()
            losses = losses_hp

        if train_hp:
            self.model_hp.train()
        else:
            self.model_hp.eval()

        positive_logits_sum = torch.zeros(self.dm.num_attributes)
        negative_logits_sum = torch.zeros(self.dm.num_attributes)
        positive_num = torch.zeros(self.dm.num_attributes)
        negative_num = torch.zeros(self.dm.num_attributes)
        if self.use_gpu:
            positive_logits_sum = positive_logits_sum.cuda()
            negative_logits_sum = negative_logits_sum.cuda()
            positive_num = positive_num.cuda()
            negative_num = negative_num.cuda()

        for batch_idx, (imgs, labels, _) in enumerate(self.trainloader):
            self.optimizer_main.zero_grad()
            self.optimizer_hp.zero_grad()
            if self.use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()
            if self.args.use_bbs:
                visibility_labels = labels[:, self.dm.num_attributes:]
                labels = labels[:, :self.dm.num_attributes]
                assert labels.shape == visibility_labels.shape
            else:
                visibility_labels = None
            # Run the batch through both nets.
            label_prediciton_probs = self.model_main(imgs)
            label_predicitons_logits = self.criterion_main.logits(label_prediciton_probs.detach())

            labels_bool = labels > 0.5  # TODO: make nicer
            positive_logits_sum += label_predicitons_logits[labels_bool].sum(0)
            negative_logits_sum += label_predicitons_logits[~labels_bool].sum(0)
            positive_num = labels_bool.sum(0)
            negative_num = (~labels_bool).sum(0)

            if not self.args.use_confidence:
                hardness_predictions = self.model_hp(imgs)
            if train_main or train_main_finetuning:
                if not self.args.use_confidence:
                    hardness_predictions_logits = self.criterion_hp.logits(hardness_predictions.detach())
                    hardness_predictions_logits = self.criterion_hp.broadcast(hardness_predictions_logits)
                elif train_main_finetuning:
                    if self.args.f1_calib:
                        decision_thresholds = self.f1_calibration_thresholds
                    else:
                        decision_thresholds = None
                    hardness_predictions_logits = 1 - metrics.get_confidence(label_predicitons_logits,
                                                                             decision_thresholds)
                if self.args.no_hp_feedback or not train_hp:
                    main_net_weights = label_prediciton_probs.new_ones(label_prediciton_probs.shape)
                else:
                    # Make a detached version of the hp scores for computing the main loss.
                    main_net_weights = hardness_predictions_logits
                if train_main_finetuning:
                    select = self.rejector(hardness_predictions_logits)
                    main_net_weights = main_net_weights * select
                # Compute main loss, gradient and optimize main net.
                loss_main = self.criterion_main(label_prediciton_probs, labels, main_net_weights)

                loss_main.backward()
                self.optimizer_main.step()

                losses_main.update(loss_main.item(), labels.size(0))

            if train_hp and not self.args.use_confidence:
                # Compute HP loss, gradient and optimize HP net.

                loss_hp = self.criterion_hp(hardness_predictions, self.hp_calibrator(label_predicitons_logits), labels, visibility_labels)


                loss_hp.backward()
                self.optimizer_hp.step()

                losses_hp.update(loss_hp.item(), labels.size(0))
            # Print progress.
            if (batch_idx + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t' 
                      'Main loss {loss.avg:.4f}\t'
                      'HP-Net loss {hp_loss.avg:.4f}'.format(
                          self.epoch + 1, batch_idx + 1, num_batch,
                          loss=losses_main,
                          hp_loss=losses_hp
                      ))
        print('Epoch: [{0}][{1}/{2}]\t'
              'Main loss {loss.avg:.4f}\t'
              'HP-Net loss {hp_loss.avg:.4f}'.format(
                self.epoch + 1, batch_idx + 1, num_batch,
                loss=losses_main,
                hp_loss=losses_hp
              ))
        positive_logits_sum /= positive_num
        negative_logits_sum /= negative_num
        self.update_hp_calibrator_thresholds((positive_logits_sum + negative_logits_sum) / 2)

        return losses_main.avg, losses_hp.avg

    def test(self, predictions=None, ground_truth=None):
        split = self.args.eval_split
        if not self.rejector.is_initialized() or self.args.no_cache:
            self.update_rejector_thresholds()

        # Get Hardness scores.

        if self.args.use_confidence:
            labels, prediction_probs, predictions = self.get_label_predictions(split)
            if self.args.f1_calib:
                decision_thresholds = self.f1_calibration_thresholds
            else:
                decision_thresholds = None
            hp_scores = 1 - metrics.get_confidence(prediction_probs, decision_thresholds)
            self.result_manager.update_outputs(split, hp_scores=hp_scores)
            print("Using confidence scores as HP-scores. ")
        elif self.args.evaluate and self.result_manager.check_output_dict(split) and not self.args.no_cache:
            _, _, _, hp_scores = self.result_manager.get_outputs(split)
        else:
            print("Computing hardness scores for testing data. ")
            hp_scores, _, _ = self.get_full_output(model=self.model_hp, criterion=self.criterion_hp)
            hp_scores = self.criterion_hp.broadcast(hp_scores)
            self.result_manager.update_outputs(split, hp_scores=hp_scores)

        ignore = np.logical_not(self.rejector(hp_scores))
        print("Rejecting the {:.2%} hardest of testing examples. ".format(ignore.mean()))
        # Run the standard accuracy testing.
        super().test(ignore)
        labels, prediction_probs, predictions, _ = self.result_manager.get_outputs(split)


        print("HP-Net Hardness Scores: ")
        print(tab.tabulate([
            ["Mean", np.mean(hp_scores)],
            ["Variance", np.var(hp_scores)]
        ]))

        # Display the hardness scores for every attribute.
        print("-" * 30)
        header = ["Attribute", "Positivity Ratio", "Accuracy", "Hardness Score Mean", "Average Precision", "cAP", "Rejection Threshold",
                  "Rejection Quantile"]
        mean = hp_scores.mean(0)
        var = np.sqrt(hp_scores.var(0))
        average_precision = metrics.hp_average_precision(labels, predictions, hp_scores)
        baseline_average_precision = self.get_baseline_average_precision()
        if baseline_average_precision is None:
            baseline_average_precision = 0
        comparative_average_precision = (average_precision > baseline_average_precision).astype("int8")
        # mean_average_precision = metrics.hp_mean_average_precision(labels, label_predictions, hp_scores)

        rejection_quantiles = ignore.mean(0).flatten()
        rejection_thresholds = self.rejector.attribute_thresholds
        if rejection_thresholds is None:
            rejection_thresholds = np.ones_like(rejection_quantiles)
        else:
            rejection_thresholds = rejection_thresholds.flatten()
        data = list(zip(self.dm.attributes, self.positivity_ratio, self.acc_atts, mean, average_precision,
                        comparative_average_precision, rejection_thresholds, rejection_quantiles))
        data += [["Total", self.positivity_ratio.mean(), self.acc_atts.mean(), mean.mean(),
                 average_precision.mean(), comparative_average_precision.mean(), rejection_thresholds.mean(),
                  rejection_quantiles.mean()]]
        table = tab.tabulate(data, floatfmt='.4f', headers=header)
        print(table)
        print("Mean average precision of hardness prediction over attributes: {:.2%}".format(average_precision.mean()))
        print("Comparative mean average precision: {:.2%}".format(comparative_average_precision.mean()))
        csv_path = osp.join(self.args.save_experiment, "result_table.csv")
        np.savetxt(csv_path, np.transpose(data), fmt="%s", delimiter="\t")
        print("Saved Table at " + csv_path)

        self.result_dict.update({
            "rejection_thresholds": self.rejector.attribute_thresholds,
            "calibration_thresholds": self.hp_calibrator.thresholds_np,
            "ignored_test_samples": ignore,
            "average_precision": average_precision
        })
        self.save_result_dict()

        hard_att_labels = None
        hard_att_pred = None
        if self.args.num_save_hard + self.args.num_save_easy > 0:
            # This part only gets executed if the corresponding arguments are passed at the terminal.
            if self.args.hard_att in self.dm.attributes:
                # If a valid attribute is given the labels for that attribute are selected.
                print("Looking at Hard attribute " + self.args.hard_att)
                att_idx = self.dm.attributes.index(self.args.hard_att)
                hard_att_labels = labels[:, att_idx]
                hard_att_pred = prediction_probs[:, att_idx]
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

        return comparative_average_precision.mean()

    def get_baseline_average_precision(self):
        return self.get_baseline_data(self.args.ap_baseline, "average_precision", "baseline average precision")

    def get_baseline_f1_calibration_thresholds(self):
        return self.get_baseline_data(self.args.f1_baseline, "f1_thresholds", "baseline F1 calibration thresholds")

    def get_baseline_data(self, filename, key, name):
        load_file = osp.join(self.args.save_experiment, filename)
        if filename and check_isfile(load_file):
            checkpoint = torch.load(load_file)

            if "result_dict" in checkpoint and checkpoint["result_dict"] is not None:
                result_dict = checkpoint["result_dict"]
                if key in result_dict and result_dict[key] is not None:
                    print("Loaded {} from file: {}".format(name, filename))
                    return result_dict[key]

        print("WARNING: Could not load {}. ".format(name))
        return None

    def clear_output_cache(self):
        super().clear_output_cache()
        self.rejector.reset()


if __name__ == '__main__':
    # global variables
    parser = argument_parser()
    args = parser.parse_args()
    trainer = RealisticPredictorTrainer(args)
