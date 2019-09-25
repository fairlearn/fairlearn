# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from . import GroupMetricResult

_MESSAGE_SIZE_MISMATCH = "Array {0} is not the same size as {1}"


def metric_by_group(metric_function, y_true, y_pred, group_data, sample_weight=None):
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
    _check_array_sizes(y_true, y_pred, 'y_true', 'y_pred')
    _check_array_sizes(y_true, group_data, 'y_true', 'group_data')
    if sample_weight is not None:
        _check_array_sizes(y_true, sample_weight, 'y_true', 'sample_weight')

    result = GroupMetricResult()

    # Make everything a numpy array
    # This allows for fast slicing of the groups
    y_a = np.squeeze(np.asarray(y_true))
    y_p = np.squeeze(np.asarray(y_pred))
    g_d = np.squeeze(np.asarray(group_data))
    s_w = None
    if sample_weight is not None:
        s_w = np.squeeze(np.asarray(sample_weight))

    # Evaluate the overall metric with the numpy arrays
    # This ensures consistency in how metric_function is called
    if s_w is not None:
        result.overall = metric_function(y_a, y_p, sample_weight=s_w)
    else:
        result.overall = metric_function(y_a, y_p)

    groups = np.unique(group_data)
    for group in groups:
        group_indices = (group == g_d)
        group_actual = y_a[group_indices]
        group_predict = y_p[group_indices]
        group_weight = None
        if s_w is not None:
            group_weight = s_w[group_indices]
            result.by_group[group] = metric_function(group_actual,
                                                     group_predict,
                                                     sample_weight=group_weight)
        else:
            result.by_group[group] = metric_function(group_actual, group_predict)

    result.min_over_groups = min(result.by_group.values())
    result.max_over_groups = max(result.by_group.values())

    result.argmin_groups = set([k for k, v in result.by_group.items() if v == result.min_over_groups])  # noqa:E501
    result.argmax_groups = set([k for k, v in result.by_group.items() if v == result.max_over_groups])  # noqa:E501

    result.range_over_groups = result.max_over_groups - result.min_over_groups
    result.range_ratio_over_groups = result.min_over_groups / result.max_over_groups

    return result


def make_group_metric(metric_function):
    def wrapper(y_true, y_pred, group_data, sample_weight=None):
        return metric_by_group(metric_function,
                               y_true,
                               y_pred,
                               group_data,
                               sample_weight)

    return wrapper


def _check_array_sizes(a, b, a_name, b_name):
    if len(a) != len(b):
        raise ValueError(_MESSAGE_SIZE_MISMATCH.format(b_name, a_name))
