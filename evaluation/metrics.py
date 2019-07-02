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


def mean_accuracy(output, target, ignore=None):
    return mean_attribute_accuracies(output, target, ignore).mean()


def mean_attribute_accuracies(output, target, ignore=None):
    """
    Get the average between the positive and negative accuracy for each attribute. As proposed in RAPv2.0 Paper
    :param output:
    :param target:
    :return:
    """
    with torch.no_grad():
        if ignore is None:
            # If ignore argument is not passed, nothing is ignored.
            ignore = np.zeros(target.shape, "int8")
        use = np.logical_not(ignore)
        prediction = output > 0.5
        p = (np.logical_and(use, target == 1)).sum(0)  # Number of positive samples.
        p[p == 0] = 1  # Prevent division by zero
        tp = np.logical_and(use, np.logical_and(target == 1, prediction == 1)).sum(0)  # Number of true positives.
        n = np.logical_and(use, (target == 0)).sum(0)  # Number of negative samples.
        n[n == 0] = 1  # Prevent division by zero
        tn = np.logical_and(use, np.logical_and(target == 0, prediction == 0)).sum(0)  # Number of true negatives.

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
            if idxs.sum() == 1:
                grouped_accuracies[group] = attribute_accuracies[idxs].mean()
            else:
                # For attribute groups it is checked whether the positive attribute in the group is the correct one.
                grouped_accuracies[group] = np.mean(output[:, idxs].argmax(axis=1) == target[:, idxs].argmax(axis=1))
        return grouped_accuracies


def accuracy(output, target, ignore=None):
    if ignore is None:
        # If ignore argument is not passed, nothing is ignored.
        ignore = np.zeros(target.shape, "int8")
    use = np.logical_not(ignore)
    with torch.no_grad():
        prediction = output > 0.5
        total_positives = np.logical_and(use, np.logical_or(target, prediction)).sum(1)
        total_positives[total_positives == 0] = 1   # Prevent division by zero

        return (np.logical_and(use, np.logical_and(target, prediction)).sum(1) / total_positives).mean()


def precision(output, target, ignore=None):
    if ignore is None:
        # If ignore argument is not passed, nothing is ignored.
        ignore = np.zeros(target.shape, "int8")
    use = np.logical_not(ignore)
    with torch.no_grad():
        prediction = output > 0.5
        pred_positives = np.logical_and(use, prediction).sum(1)
        pred_positives[pred_positives == 0] = 1  # Prevent division by zero

        return (np.logical_and(use, np.logical_and(target, prediction)).sum(1) / pred_positives).mean()


def recall(output, target, ignore=None):
    if ignore is None:
        # If ignore argument is not passed, nothing is ignored.
        ignore = np.zeros(target.shape, "int8")
    use = np.logical_not(ignore)
    with torch.no_grad():
        prediction = output > 0.5
        target_positives = np.logical_and(use, target).sum(1)
        target_positives[target_positives == 0] = 1   # Prevent division by zero

        return (np.logical_and(use, np.logical_and(target, prediction)).sum(1) / target_positives).mean()


def f1measure(output, target, ignore=None):
    pre = precision(output, target, ignore)
    rec = recall(output, target, ignore)
    return 2 * (pre * rec) / (pre + rec)


def get_metrics(output, target, ignore=None):
    return (
        mean_accuracy(output, target, ignore),
        accuracy(output, target, ignore),
        precision(output, target, ignore),
        recall(output, target, ignore),
        f1measure(output, target, ignore)
    )


def get_metrics_table(output, target, ignore=None):
    metrics = get_metrics(output, target, ignore)
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


def get_f1_calibration_thresholds(output, target, ignore=None, resolution=100):
    """
    Calculate the thresholds for F1 Calibration as defined in:
    [Bekele et al. 2019] The Deeper, the Better: Analysis of Person Attributes Recognition
    https://arxiv.org/abs/1901.03756
    :param output:
    :param target:
    :param resolution:
    :return:
    """
    if ignore is None:
        # If ignore argument is not passed, nothing is ignored.
        ignore = np.zeros(target.shape, "int8")
    use = np.logical_not(ignore)
    num_attributes = output.shape[1]
    num_datapoints = output.shape[0]
    thresholds = np.zeros((1, num_attributes))
    best_diff = np.empty((num_attributes, ))
    best_diff.fill(num_datapoints)
    for i in range(1, resolution):
        # For each attribute, find the threshold for which the number of false positives and the number of false
        # negatives is the same. For this threshold precision and recall are the same.
        t = i / resolution
        predictions = output > t
        fn = np.logical_and(use, np.logical_and(target == 1, predictions == 0)).sum(0)  # Number of false negatives.
        fp = np.logical_and(use, np.logical_and(target == 0, predictions == 1)).sum(0)  # Number of false positives.
        diff = np.absolute(fp - fn).flatten()  # The difference between fp and fn -> minimize
        # For each attribute, if the difference is lower than the previous best (lowest), the threshold is overwritten.
        thresholds[:, diff < best_diff] = t
        # For each attribute, the best (lowest) difference is updated.
        best_diff = np.where(diff < best_diff, diff, best_diff)
    return thresholds

