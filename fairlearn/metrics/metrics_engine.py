# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from . import GroupMetricResult

_MESSAGE_SIZE_MISMATCH = "Array {0} is not the same size as {1}"


def metric_by_group(metric_function, y_true, y_pred, group_membership, sample_weight=None):
    """ Applies a metric to each subgroup of a set of data

    Parameters
    ----------
    metric_function : function
        Function with signature (y_true, y_pred, sample_weight=None) which returns a scalar

    y_true : array_like
        The ground-truth values

    y_pred : array_like
        The predicted values

    group_membership : array_like
        Indicates the group to which each input value belongs

    sample_weight : array_like, optional
        The weights to apply to each input value

    Returns
    -------
    out : GroupMetricResult
        Object containing the result of applying metric_function to the entire dataset
        and to each group identified in group_membership.
        If the metric_function returns a scalar, then additional fields are populated
    """
    _check_array_sizes(y_true, y_pred, 'y_true', 'y_pred')
    _check_array_sizes(y_true, group_membership, 'y_true', 'group_membership')
    if sample_weight is not None:
        _check_array_sizes(y_true, sample_weight, 'y_true', 'sample_weight')

    result = GroupMetricResult()

    # Make everything a numpy array
    # This allows for fast slicing of the groups
    y_a = np.squeeze(np.asarray(y_true))
    y_p = np.squeeze(np.asarray(y_pred))
    g_d = np.squeeze(np.asarray(group_membership))
    s_w = None
    if sample_weight is not None:
        s_w = np.squeeze(np.asarray(sample_weight))

    # Evaluate the overall metric with the numpy arrays
    # This ensures consistency in how metric_function is called
    if s_w is not None:
        result.overall = metric_function(y_a, y_p, sample_weight=s_w)
    else:
        result.overall = metric_function(y_a, y_p)

    groups = np.unique(group_membership)
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

    try:
        result.minimum = min(result.by_group.values())
        result.maximum = max(result.by_group.values())

        result.argmin_set = set([k for k, v in result.by_group.items() if v == result.minimum])  # noqa:E501
        result.argmax_set = set([k for k, v in result.by_group.items() if v == result.maximum])  # noqa:E501

        result.range = result.maximum - result.minimum
        if result.minimum < 0:
            result.range_ratio = np.nan
        elif result.maximum == 0:
            # We have min=max=0
            result.range_ratio = 1
        else:
            result.range_ratio = result.minimum / result.maximum
    except ValueError:
        # Nothing to do
        # Failed to compute an extra result, most likely because operation (such as min)
        # was not defined for the return type (e.g. doing confusion matrices)
        pass

    return result


def make_group_metric(metric_function):
    def wrapper(y_true, y_pred, group_membership, sample_weight=None):
        return metric_by_group(metric_function,
                               y_true,
                               y_pred,
                               group_membership,
                               sample_weight)

    # Improve the name of the returned function
    wrapper.__name__ = "group_{0}".format(metric_function.__name__)

    return wrapper


def _check_array_sizes(a, b, a_name, b_name):
    if len(a) != len(b):
        raise ValueError(_MESSAGE_SIZE_MISMATCH.format(b_name, a_name))
