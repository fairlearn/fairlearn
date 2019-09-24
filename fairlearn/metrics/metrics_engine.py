# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from sklearn.metrics import recall_score

from . import DisparityResult, MetricResult

_MESSAGE_NON_BINARY = "Array {0} contains values other than 0 and 1"


def true_positive_rate(y_actual, y_predict, group_data, sample_weight=None):
    return metric_by_groups(recall_score, y_actual, y_predict, group_data, sample_weight)


def selection_rate(y_actual, y_predict, group_data, sample_weight=None):
    if sample_weight is not None:
        raise NotImplementedError("selection_rate and sample_weight")

    def selection_metric(y_a, y_p, sample_weight):
        return np.mean(y_p)

    return metric_by_groups(selection_metric, y_actual, y_predict, group_data, sample_weight)


def metric_by_groups(metric_function, y_actual, y_predict, group_data, sample_weight=None):
    """ Applies a metric to each subgroup of a set of data

    :param metric_function
    :type Function with signature (y_actual, y_predict, sample_weight=None) which returns a scalar

    :param y_actual
    :type Array of actual results (must be 0 or 1)

    :param y_predict
    :type Array of predicted results (must be 0 or 1)

    :param group_data
    :type Array indicating the group to which each result belongs

    :param sample_weight
    :type Array of weights to apply to each result
    """
    _check_binary(y_actual, "y_actual")
    _check_binary(y_predict, "y_predict")
    result = MetricResult()

    groups = np.unique(group_data)

    result.metric = metric_function(y_actual, y_predict, sample_weight=sample_weight)

    # The slicing we use requires Numpy arrays
    y_a = np.array(y_actual)
    y_p = np.array(y_predict)
    s_w = None
    if sample_weight is not None:
        s_w = np.array(sample_weight)
    for group in groups:
        group_indices = [i for i, elem in enumerate(group_data) if elem == group]
        group_actual = y_a[group_indices]
        group_predict = y_p[group_indices]
        group_weight = None
        if s_w is not None:
            group_weight = s_w[group_indices]
        result.group_metric[group] = metric_function(group_actual,
                                                     group_predict,
                                                     sample_weight=group_weight)

    return result


def make_group_metric(metric_function):
    def wrapper(y_actual, y_predict, group_data, sample_weight=None):
        return metric_by_groups(metric_function,
                                y_actual,
                                y_predict,
                                group_data,
                                sample_weight)

    return wrapper


def compute_disparity(group_metric_function,
                      y_actual, y_predict, group_data,
                      comparison,
                      sample_weight=None):
    metrics = group_metric_function(y_actual, y_predict, group_data, sample_weight)

    result = DisparityResult()
    result.group_metric = metrics.group_metric

    metric_values = np.array(list(result.group_metric.values()))

    if comparison == 'ratio':
        result.disparity = 1 - np.min(metric_values) / np.max(metric_values)
    elif comparison == 'diff':
        result.disparity = np.max(metric_values) - np.min(metric_values)
    else:
        raise ValueError("comparison")

    return result


def make_disparity_metric(metric_function, comparison):

    def wrapper(y_actual, y_predict, group_data, sample_weight=None):
        return compute_disparity(metric_function,
                                 y_actual,
                                 y_predict,
                                 group_data,
                                 comparison,
                                 sample_weight)

    return wrapper


def _check_binary(arr, arr_name):
    unique_values = np.unique(arr)
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(_MESSAGE_NON_BINARY.format(arr_name))
