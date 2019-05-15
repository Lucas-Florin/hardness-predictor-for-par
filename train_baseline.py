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
from training.losses import SigmoidCrossEntropyLoss
from utils.iotools import check_isfile, save_checkpoint
from utils.avgmeter import AverageMeter
from utils.loggers import Logger, AccLogger
from utils.torchtools import count_num_param, open_all_layers, open_specified_layers, accuracy, load_pretrained_weights
from utils.generaltools import set_random_seed
from evaluation.metrics import *
from training.optimizers import init_optimizer
from training.lr_schedulers import init_lr_scheduler

# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args # The arguments from the Terminal.

    set_random_seed(args.seed)

    # Decide which processor (CPU or GPU) to use.
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    # Start logger.
    ts = time.strftime("_%Y-%m-%d_%H-%M-%S")
    log_name = 'test' + ts + '.log' if args.evaluate else 'train' + ts + '.log'
    sys.stdout = Logger(osp.join(args.save_experiment, log_name))

    # Print out the arguments taken from Terminal (or defaults).
    print('==========\nArgs:{}\n=========='.format(args))

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
    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    # ?
    model = nn.DataParallel(model).cuda() if use_gpu else model

    # ?
    criterion = SigmoidCrossEntropyLoss(num_classes=dm.num_attributes, use_gpu=use_gpu)
    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    #if args.resume and check_isfile(args.resume):
    #    args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

    # TODO: Test.
    if args.evaluate:
        print('Evaluate only')
        split = args.eval_split
        print('=> Evaluating {} on {} ...'.format(args.dataset_name, split))
        testloader = testloader_dict[split]
        acc, acc_atts = test(model, testloader, criterion.logits, dm.attributes, use_gpu)

        return
    time_start = time.time()
    ranklogger = AccLogger()
    print('=> Start training')
    # TODO: Plot loss over epochs.

    # Train Fixbase epochs.
    if args.fixbase_epoch > 0:
        print('Train {} for {} epochs while keeping other layers frozen'.format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            train(epoch, model, criterion, optimizer, trainloader, dm.attributes, use_gpu, fixbase=True)

        print('Done. All layers are open to train for {} epochs'.format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)

    # Train non-fixbase epochs.
    for epoch in range(args.start_epoch, args.max_epoch):
        train(epoch, model, criterion, optimizer, trainloader, dm.attributes, use_gpu)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch:
            split = args.eval_split
            print('=> Evaluating {} on {} ...'.format(args.dataset_name, split))
            testloader = testloader_dict[split]
            acc, acc_atts = test(model, testloader, criterion.logits, dm.attributes, use_gpu)
            ranklogger.write(epoch + 1, acc)
            ts = time.strftime("_%Y-%m-%d_%H-%M-%S")
            filename = 'checkpoint' + ts + '.pth.tar'
            save_checkpoint({
                'state_dict': model.state_dict(),
                'acc': acc,
                'acc_atts': acc_atts,
                'epoch': epoch + 1,
                'model': args.model,
                'optimizer': optimizer.state_dict(),
            }, osp.join(args.save_experiment, filename))
            print("Saved model checkpoint at " + filename)

    # Calculate elapsed time.
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))
    ranklogger.show_summary()


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
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    if fixbase or args.always_fixbase:
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)

    end = time.time()
    for batch_idx, (imgs, labels, _) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, labels = imgs.cuda(), labels.cuda()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), labels.size(0))
        acc, acc_atts = accuracy(criterion.logits(outputs), labels)
        accs.update(acc)
        accs_atts.update(acc_atts)

        if (batch_idx + 1) % args.print_freq == 0:
            best_idx = torch.argmax(accs_atts.avg)
            worst_idx = torch.argmin(accs_atts.avg)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.2%} ({acc.avg:.2%})\t'
                  'Best {best_att}: {best_acc:.2%}, '
                  'worst {worst_att}: {worst_acc:.2%}'.format(
                epoch + 1, batch_idx + 1, len(trainloader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accs,
                best_att=attributes[best_idx], best_acc=accs_atts.avg[best_idx],
                worst_att=attributes[worst_idx], worst_acc=accs_atts.avg[worst_idx]
            ))

        end = time.time()


def test(model, testloader, logits, attributes, use_gpu):
    """

    :param model:
    :param testloader:
    :param logits:
    :param attributes:
    :param use_gpu:
    :return:
    """
    batch_time = AverageMeter()
    model.eval()
    print(len(testloader))
    with torch.no_grad():
        predictions, gt = list(), list()
        for batch_idx, (imgs, labels, _) in enumerate(testloader):
            print(batch_idx)
            if use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            end = time.time()
            outputs = model(imgs)
            outputs = logits(outputs)
            batch_time.update(time.time() - end)

            predictions.extend((outputs > 0.5).tolist())
            gt.extend(labels.tolist())

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.test_batch_size))

    # compute test accuracies
    correct = np.array(predictions) == np.array(gt)
    acc = np.mean(correct)
    acc_atts = np.mean(correct, axis=0)

    print('Results ----------')
    print('Accuracy: {:.2%}'.format(acc))
    print('Attribute accuracies:')
    for i in range(len(attributes)):
        print('{}: {:.2%}'.format(attributes[i], acc_atts[i]))
    print('------------------')

    return acc, acc_atts


if __name__ == '__main__':
    main()
