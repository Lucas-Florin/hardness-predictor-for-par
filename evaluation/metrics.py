"""
Metrics for evaluating models as defined in:
Pedestrian Attribute Recognition: A Survey.
Xiao Wang, Shaofei Zheng, Rui Yang, Bin Luo, Jin Tang
https://arxiv.org/abs/1901.07474v1
"""

import numpy as np
import torch


def _get_conf_mat(output, target):
    with torch.no_grad():
        prediction = output > 0.5
        tp = np.logical_and(target == 1, prediction == 1).sum(1)  # Number of true positives.
        fn = np.logical_and(target == 1, prediction == 0).sum(1)  # Number of false negatives.
        fp = np.logical_and(target == 0, prediction == 1).sum(1)  # Number of false positives.
        tn = np.logical_and(target == 0, prediction == 0).sum(1)  # Number of true negatives.

    return tp, fn, fp, tn


def mean_accuracy(output, target):
    return attribute_accuracies(output, target).mean()


def attribute_accuracies(output, target):
    with torch.no_grad():
        prediction = output > 0.5
        p = (target == 1).sum(0)  # Number of positive samples.
        tp = np.logical_and(target == 1, prediction == 1).sum(0)  # Number of true positives.
        n = (target == 0).sum(0)  # Number of negative samples.
        tn = np.logical_and(target == 0, prediction == 0).sum(0)  # Number of true negatives.

        return (tp.astype("float64") / p + tn.astype("float64") / n) / 2


def accuracy(output, target):
    with torch.no_grad():
        prediction = output > 0.5

        return (np.logical_and(target, prediction).sum(1) / np.logical_or(target, prediction).sum(1)).mean()


def precision(output, target):
    with torch.no_grad():
        prediction = output > 0.5

        return (np.logical_and(target, prediction).sum(1) / prediction.sum(1)).mean()


def recall(output, target):
    with torch.no_grad():
        prediction = output > 0.5

        return (np.logical_and(target, prediction).sum(1) / target.sum(1)).mean()


def f1measure(output, target):
    pre = precision(output, target)
    rec = recall(output, target)
    return 2 * (pre * rec) / (pre + rec)


def get_metrics(output, target):
    return (
        mean_accuracy(output, target),
        accuracy(output, target),
        precision(output, target),
        recall(output, target),
        f1measure(output, target)
    )

