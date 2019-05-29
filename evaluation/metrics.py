"""
Metrics for evaluating models as defined in:
Pedestrian Attribute Recognition: A Survey.
Xiao Wang, Shaofei Zheng, Rui Yang, Bin Luo, Jin Tang
https://arxiv.org/abs/1901.07474v1

@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import torch
import tabulate as tab


def _get_conf_mat(output, target):
    with torch.no_grad():
        prediction = output > 0.5
        tp = np.logical_and(target == 1, prediction == 1).sum(1)  # Number of true positives.
        fn = np.logical_and(target == 1, prediction == 0).sum(1)  # Number of false negatives.
        fp = np.logical_and(target == 0, prediction == 1).sum(1)  # Number of false positives.
        tn = np.logical_and(target == 0, prediction == 0).sum(1)  # Number of true negatives.

    return tp, fn, fp, tn


def mean_accuracy(output, target):
    return mean_attribute_accuracies(output, target).mean()


def mean_attribute_accuracies(output, target):
    """
    Get the average between the positive and negative accuracy for each attribute. As proposed in RAPv2.0 Paper
    :param output:
    :param target:
    :return:
    """
    with torch.no_grad():
        prediction = output > 0.5
        p = (target == 1).sum(0)  # Number of positive samples.
        p[p == 0] = 1  # Prevent division by zero
        tp = np.logical_and(target == 1, prediction == 1).sum(0)  # Number of true positives.
        n = (target == 0).sum(0)  # Number of negative samples.
        n[n == 0] = 1  # Prevent division by zero
        tn = np.logical_and(target == 0, prediction == 0).sum(0)  # Number of true negatives.

        m_acc = (tp.astype("float64") / p + tn.astype("float64") / n) / 2

        return m_acc


def attribute_accuracies(output, target, attribute_groupings=None):
    """
    Get the raw accuracy for each attribute as defined in Market1501 Attribute Paper.
    If attribute groupings are given, the average for each group is calculated.
    :param output:
    :param target:
    :param attribute_groupings:
    :return:
    """
    with torch.no_grad():
        prediction = output > 0.5
        attribute_accuracies = (prediction == target).mean(0)
        if attribute_groupings is None:
            return attribute_accuracies
        attribute_groupings = np.array(attribute_groupings, dtype=np.int)
        num_groups = attribute_groupings.max() + 1
        grouped_accuracies = np.zeros((num_groups, ))

        for group in range(num_groups):
            idxs = attribute_groupings == group
            grouped_accuracies[group] = attribute_accuracies[idxs].mean()
        return grouped_accuracies




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


def get_metrics_table(output, target):
    metrics = get_metrics(output, target)
    metric_names = [
        "Mean Accuracy",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
    table = tab.tabulate(zip(metric_names, metrics), floatfmt='.2%')
    return table


def group_attributes(output, attribute_grouping):
    """
    Group binary attributes into non-binary ones. As proposed in Market1501-Attribute Paper.
    :param output:
    :param attribute_grouping:
    :return:
    """
    with torch.no_grad():
        attg = np.array(attribute_grouping, dtype='int16')
        num_grouped_atts = attg.max() + 1
        grouped_output = np.zeros(output.shape, dtype='bool')
        num_datapoints = output.shape[0]
        for i in range(num_grouped_atts):
            idxs = np.argwhere(attg == i).flatten()
            starting_idx = idxs[0]
            if len(idxs) == 1:
                grouped_output[:, idxs] = output[:, idxs] > 0.5
            else:
                att_max = output[:, idxs].argmax(1) + starting_idx
                grouped_output[range(num_datapoints), att_max] = True
    return grouped_output





