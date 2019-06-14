import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from data.data_manager import ImageDataManager
from data.datasets import get_img_dataset
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

# global variables
parser = argument_parser()
args = parser.parse_args()

# TODO: Modularize.
# TODO: Remove unnecessary outputs and corresponding computations.

def main():
    global args # The arguments from the Terminal.
    time_start = time.time()
    set_random_seed(args.seed)

    # Decide which processor (CPU or GPU) to use.
    if not args.use_avai_gpus:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    # Start logger.
    ts = time.strftime("%Y-%m-%d_%H-%M-%S_")

    log_name = ts + 'test' + '.log' if args.evaluate else ts + 'train' + '.log'
    sys.stdout = Logger(osp.join(args.save_experiment, log_name))

    # Print out the arguments taken from Terminal (or defaults).
    print('==========\nArgs:{}\n=========='.format(args))

    print("Timestamp: " + ts)
    # Warn if not using GPU.
    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print('Initializing model: {}'.format(args.model))
    model = models.init_model(name=args.model, num_classes=dm.num_attributes, loss={'xent'},
                              pretrained=not args.no_pretrained, use_gpu=use_gpu)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    # Load pretrained weights if specified in args.
    load_file = osp.join(args.save_experiment, args.load_weights)
    if args.load_weights:
        if check_isfile(load_file):
            load_pretrained_weights(model, load_file)
        else:
            print("WARNING: Could not load pretraining weights")

    # Load model onto GPU if GPU is used.
    model = nn.DataParallel(model).cuda() if use_gpu else model

    # Select Loss function.
    if args.loss_func == "deepmar":
        pos_ratio = dm.dataset.get_positive_attribute_ratio()
        criterion = DeepMARLoss(pos_ratio, args.train_batch_size, use_gpu=use_gpu, sigma=args.loss_func_param)
    elif args.loss_func == "scel":
        criterion = SigmoidCrossEntropyLoss(num_classes=dm.num_attributes, use_gpu=use_gpu)
    elif args.loss_func == "sscel":
        attribute_grouping = dm.dataset.attribute_grouping
        criterion = SplitSoftmaxCrossEntropyLoss(attribute_grouping, use_gpu=use_gpu)
    else:
        criterion = None

    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    #if args.resume and check_isfile(args.resume):
    #    args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

    if args.evaluate:
        print('Evaluate only')
        split = args.eval_split
        print('=> Evaluating {} on {} ...'.format(args.dataset_name, split))
        f1_calibration_thresholds = get_threshold(model, trainloader, criterion.logits, use_gpu)
        testloader = testloader_dict[split]
        acc, acc_atts = test(model, testloader, criterion.logits, dm.attributes, use_gpu, dm.dataset,
                             f1_calibration_thresholds)

        # Calculate testing time.
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Testing Time: {}'.format(elapsed))
        return
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Loading Time: {}'.format(elapsed))
    time_start = time.time()
    ranklogger = AccLogger()
    print('=> Start training')
    epoch_losses = np.zeros(shape=(args.max_epoch, ))

    # Train Fixbase epochs.
    if args.fixbase_epoch > 0:
        print('Train {} for {} epochs while keeping other layers frozen'.format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            epoch_losses[epoch] = train(epoch, model, criterion, optimizer, trainloader, dm.attributes, use_gpu,
                                        fixbase=True)


        print('Done. All layers are open to train for {} epochs'.format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)

    # Train non-fixbase epochs.
    for epoch in range(args.start_epoch, args.max_epoch):
        loss = train(epoch, model, criterion, optimizer, trainloader, dm.attributes, use_gpu)

        epoch_losses[epoch] = loss
        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch:
            split = args.eval_split
            f1_calibration_thresholds = get_threshold(model, trainloader, criterion.logits, use_gpu)

            print('=> Evaluating {} on {} ...'.format(args.dataset_name, split))
            testloader = testloader_dict[split]
            acc, acc_atts = test(model, testloader, criterion.logits, dm.attributes, use_gpu, dm.dataset,
                                 f1_calibration_thresholds)
            ranklogger.write(epoch + 1, acc)
            filename = ts + 'checkpoint' + '.pth.tar'
            save_checkpoint({
                'state_dict': model.state_dict(),
                'acc': acc,
                'acc_atts': acc_atts,
                'epoch': epoch + 1,
                'model': args.model,
                'optimizer': optimizer.state_dict(),
                'losses': epoch_losses,
                'args': args
            }, osp.join(args.save_experiment, filename))
            print("Saved model checkpoint at " + filename)

    # Calculate elapsed time.
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Training Time {}'.format(elapsed))
    ranklogger.show_summary()

    if args.plot_epoch_loss:
        # Plot loss over epochs.
        plot_epoch_losses(epoch_losses, args.save_experiment, ts)


def train(epoch, model, criterion, optimizer, trainloader, attributes, use_gpu, fixbase=False):
    """
    Train the model for an epoch.
    :param epoch: Number of current epoch (zero indexed).
    :param model: the model to be trained.
    :param criterion:
    :param optimizer:
    :param trainloader:
    :param attributes:
    :param use_gpu:
    :param fixbase: Is this a fixbase epoch?
    :return: Time of execution end.
    """
    losses = AverageMeter()
    accs = AverageMeter()
    accs_atts = AverageMeter()

    model.train()

    if fixbase or args.always_fixbase:
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)

    for batch_idx, (imgs, labels, _) in enumerate(trainloader):

        if use_gpu:
            imgs, labels = imgs.cuda(), labels.cuda()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        losses.update(loss.item(), labels.size(0))
        acc, acc_atts = accuracy(criterion.logits(outputs), labels)
        accs.update(acc)
        accs_atts.update(acc_atts)

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.2%} ({acc.avg:.2%})'.format(
                epoch + 1, batch_idx + 1, len(trainloader),
                loss=losses,
                acc=accs
            ))
    return losses.avg


def test(model, testloader, logits, attributes, use_gpu, dataset, f1_calibration_thresholds):
    """

    :param model:
    :param testloader:
    :param logits:
    :param attributes:
    :param use_gpu:
    :return:
    """
    global args
    model.eval()
    with torch.no_grad():
        predictions, gt = list(), list()
        for batch_idx, (imgs, labels, _) in enumerate(testloader):
            if use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            end = time.time()
            outputs = model(imgs)
            outputs = logits(outputs)

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
        attribute_grouping = dataset.attribute_grouping
        predictions = metrics.group_attributes(predictions, attribute_grouping)
        if not args.use_macc:
            attributes = dataset.grouped_attribute_names
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


def get_threshold(model, loader, logits, use_gpu):
    """

    :param model:
    :param loader:
    :param logits:
    :param use_gpu:
    :return:
    """
    global args
    model.eval()
    with torch.no_grad():
        predictions, gt = list(), list()
        for batch_idx, (imgs, labels, _) in enumerate(loader):
            if use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            outputs = model(imgs)
            outputs = logits(outputs)

            predictions.extend(outputs.tolist())
            gt.extend(labels.tolist())

    # compute test accuracies
    predictions = np.array(predictions)
    gt = np.array(gt, dtype="bool")
    return metrics.get_f1_calibration_thresholds(predictions, gt)


if __name__ == '__main__':
    main()
