# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from . import MetricResult

_MESSAGE_NON_BINARY = "Array {0} contains values other than 0 and 1"


def metric_by_groups(metric_function, y_true, y_pred, group_data, sample_weight=None):
    """ Applies a metric to each subgroup of a set of data

    :param metric_function
    :type Function with signature (y_true, y_pred, sample_weight=None) which returns a scalar

    :param y_true
    :type Array of actual results (must be 0 or 1)

    :param y_pred
    :type Array of predicted results (must be 0 or 1)

    :param group_data
    :type Array indicating the group to which each result belongs

    :param sample_weight
    :type Array of weights to apply to each result
    """
    _check_binary(y_true, "y_true")
    _check_binary(y_pred, "y_pred")
    result = MetricResult()

    groups = np.unique(group_data)

    result.metric = metric_function(y_true, y_pred, sample_weight=sample_weight)

    # The slicing we use requires Numpy arrays
    y_a = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    g_d = np.asarray(group_data)
    s_w = None
    if sample_weight is not None:
        s_w = np.asarray(sample_weight)

    for group in groups:
        group_indices = (group == g_d)
        group_actual = y_a[group_indices]
        group_predict = y_p[group_indices]
        group_weight = None
        if s_w is not None:
            group_weight = s_w[group_indices]
        result.group_metric[group] = metric_function(group_actual,
                                                     group_predict,
                                                     sample_weight=group_weight)

    result.min_metric = min(result.group_metric.values())
    result.max_metric = max(result.group_metric.values())

    result.min_metric_groups = set([k for k, v in result.group_metric.items() if v == result.min_metric])  # noqa:E501
    result.max_metric_groups = set([k for k, v in result.group_metric.items() if v == result.max_metric])  # noqa:E501

    result.metric_range = result.max_metric - result.min_metric
    result.metric_range_ratio = result.max_metric / result.min_metric

    return result


def make_group_metric(metric_function):
    def wrapper(y_true, y_pred, group_data, sample_weight=None):
        return metric_by_groups(metric_function,
                                y_true,
                                y_pred,
                                group_data,
                                sample_weight)

    return wrapper


def _check_binary(arr, arr_name):
    unique_values = np.unique(arr)
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(_MESSAGE_NON_BINARY.format(arr_name))
