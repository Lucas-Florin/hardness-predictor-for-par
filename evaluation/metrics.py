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
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score



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


def average_precision(target, score):
    """
    Get the average precision for one query as defined in:
    Kishida, K. (2005). Property of average precision and its generalization: An examination of evaluation indicator
    for information retrieval experiments (p. 19p). Tokyo, Japan: National Institute of Informatics.
    :param target: ground truth relevance
    :param score: the ranking score
    :return: the average precision for that query
    """
    # Make sure the arrays are in the correct format.
    """
    if score.ndim == 1:
        score = score.reshape((1, score.size))
    if target.ndim == 1:
        target = target.reshape((1, target.size))
    """
    #assert (target.shape == score.shape).all()
    # Calculate the average precision using the function from scikit-learn
    return average_precision_score(target, score)


def hp_average_precision(labels, predictions, hp_scores):
    """
    Get the average precision metric for a hardness predictor for each attribute.
    Each attribute is treated as a separate query. Falsely predicted labels are ground truth relevant. The hardness
    score is the predicted relevancy score.
    :param labels: The ground truth attribute labels.
    :param predictions: The predicted attribute labels.
    :param hp_scores: The hardness scores given by the hardness predictor.
    :return: An array with the average precision for each attribute.
    """
    # Transpose the arrays so that each attribute can be analyzed separately.
    labels = labels.transpose()
    predictions = predictions.transpose()
    hp_scores = hp_scores.transpose()
    # Compute the average precision for each attribute.
    attribute_average_precisions = [average_precision(l != p, s) for l, p, s in zip(labels, predictions, hp_scores)]
    return np.array(attribute_average_precisions)


def hp_mean_average_precision(labels, predictions, hp_scores):
    """
    Get the mean average precision metric for a hardness predictor for each attribute.
    Each attribute is treated as a separate query. For each attribute, the average precision for the positive and for
    the negative samples is calculated separately and then averaged. Falsely predicted labels are ground truth
    relevant. The hardness score is the predicted relevancy score.
    :param labels: The ground truth attribute labels.
    :param predictions: The predicted attribute labels.
    :param hp_scores: The hardness scores given by the hardness predictor.
    :return: An array with the average precision for each attribute.
    """
    # Transpose the arrays so that each attribute can be analyzed separately.
    labels = labels.transpose()
    predictions = predictions.transpose()
    hp_scores = hp_scores.transpose()
    attribute_average_precisions = []
    # Compute the mean average precision score for each attribute.
    for l, p, s in zip(labels, predictions, hp_scores):
        # Compute the score for the positive and the negative samples separately
        ap_cor = average_precision(l != p, s)
        ap_err = average_precision(l == p, 1 - s)
        if np.isnan(ap_cor):
            ap_cor = 0
        if np.isnan(ap_err):
            ap_err = 0
        attribute_average_precisions.append((ap_cor + ap_err) / 2)
    return np.array(attribute_average_precisions)


def get_confidence(prediction_probs, decision_thresholds=None):
    if decision_thresholds is None:
        decision_thresholds = 0.5
    elif type(prediction_probs) == torch.Tensor:
        decision_thresholds = prediction_probs.new_tensor(decision_thresholds)
    negative_multiplicator = 1 / decision_thresholds
    positive_multiplicator = 1 / (1 - decision_thresholds)
    predictions = prediction_probs > decision_thresholds
    not_predictions = 1 - predictions
    if type(prediction_probs) == torch.Tensor:
        confidence = (prediction_probs - decision_thresholds).abs()
        predictions = prediction_probs.new_tensor(predictions)
        not_predictions = prediction_probs.new_tensor(not_predictions)
    else:
        confidence = np.abs(prediction_probs - decision_thresholds)
    confidence = ((predictions * confidence * positive_multiplicator) +
                  (not_predictions * confidence * negative_multiplicator))
    assert confidence.max() <= 1
    assert confidence.min() >= 0
    return confidence


def column_indexing(array, idxs, axis=0):
    assert axis == 0
    num_attributes = array.shape[1]
    assert num_attributes == idxs.shape[1]
    result = np.empty_like(array, shape=idxs.shape)
    for att in range(num_attributes):
        result[:, att] = array[:, att][idxs[:, att]]
    return result

