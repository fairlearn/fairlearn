# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from ._group_metric_result import GroupMetricResult
from ._input_manipulations import _convert_to_ndarray_and_squeeze

_MESSAGE_SIZE_MISMATCH = "Array {0} is not the same size as {1}"


def metric_by_group(metric_function,
                    y_true, y_pred, group_membership,
                    sample_weight=None,
                    **kwargs):
    r"""Apply a metric to each subgroup of a set of data.

    :param metric_function: Function ``(y_true, y_pred, sample_weight=None, \*\*kwargs)``

    :param y_true: Array of ground-truth values

    :param y_pred: Array of predicted values

    :param group_membership: Array Indicating the group to which each input value belongs

    :param sample_weight: Optional weights to apply to each input value

    :param \*\*kwargs: Optional arguments to be passed to the `metric_function`

    :return: Object containing the result of applying ``metric_function`` to the entire dataset
        and to each group identified in ``group_membership``.
        If the ``metric_function`` returns a scalar, then additional fields are populated
    :rtype: :class:`GroupMetricResult`
    """
    _check_array_sizes(y_true, y_pred, 'y_true', 'y_pred')
    _check_array_sizes(y_true, group_membership, 'y_true', 'group_membership')
    if sample_weight is not None:
        _check_array_sizes(y_true, sample_weight, 'y_true', 'sample_weight')

    result = GroupMetricResult()

    # Make everything a numpy array
    # This allows for fast slicing of the groups
    y_a = _convert_to_ndarray_and_squeeze(y_true)
    y_p = _convert_to_ndarray_and_squeeze(y_pred)
    g_d = _convert_to_ndarray_and_squeeze(group_membership)

    s_w = None
    if sample_weight is not None:
        s_w = _convert_to_ndarray_and_squeeze(sample_weight)

    # Evaluate the overall metric with the numpy arrays
    # This ensures consistency in how metric_function is called
    if s_w is not None:
        result.overall = metric_function(y_a, y_p, sample_weight=s_w, **kwargs)
    else:
        result.overall = metric_function(y_a, y_p, **kwargs)

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
                                                     sample_weight=group_weight,
                                                     **kwargs)
        else:
            result.by_group[group] = metric_function(group_actual,
                                                     group_predict,
                                                     **kwargs)

    return result


def make_group_metric(metric_function):
    """Turn a regular metric into a grouped metric.

    :param metric_function: The function to be wrapped. This must have signature
        ``(y_true, y_pred, sample_weight, **kwargs)``
    :type metric_function: func

    :return: A wrapped version of the supplied metric_function. It will have
        signature ``(y_true, y_pred, group_membership, sample_weight, **kwargs)``
    :rtype: func
    """
    def wrapper(y_true, y_pred, group_membership, sample_weight=None, **kwargs):
        return metric_by_group(metric_function,
                               y_true,
                               y_pred,
                               group_membership,
                               sample_weight,
                               **kwargs)

    # Improve the name of the returned function
    wrapper.__name__ = "group_{0}".format(metric_function.__name__)

    return wrapper


def _check_array_sizes(a, b, a_name, b_name):
    if len(a) != len(b):
        raise ValueError(_MESSAGE_SIZE_MISMATCH.format(b_name, a_name))
