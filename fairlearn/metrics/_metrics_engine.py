# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from ._input_manipulations import _convert_to_ndarray_and_squeeze

_OVERALL = 'overall'
_BY_GROUP_FORMAT = 'group_{0}'
_MIN = 'min'
_MAX = 'max'
_RANGE = 'range'
_RANGE_RATIO = 'range_ratio'
_ARGMIN = 'argmin'
_ARGMAX = 'argmax'

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

    # Make everything a numpy array
    # This allows for fast slicing of the groups
    y_a = _convert_to_ndarray_and_squeeze(y_true)
    y_p = _convert_to_ndarray_and_squeeze(y_pred)
    g_d = _convert_to_ndarray_and_squeeze(group_membership)

    s_w = None
    if sample_weight is not None:
        s_w = _convert_to_ndarray_and_squeeze(sample_weight)

    result = dict()
    groups = np.unique(group_membership)
    for group in groups:
        group_indices = (group == g_d)
        group_actual = y_a[group_indices]
        group_predict = y_p[group_indices]
        group_weight = None
        group_key = _BY_GROUP_FORMAT.format(group)
        if s_w is not None:
            group_weight = s_w[group_indices]
            result[group_key] = metric_function(group_actual,
                                                group_predict,
                                                sample_weight=group_weight,
                                                **kwargs)
        else:
            result[group_key] = metric_function(group_actual,
                                                group_predict,
                                                **kwargs)

    # Compute all the statistics, taking care not to stomp on our dictionary
    try:
        minimum = min(result.values())
        argmin = [k for k, v in result.items() if v == minimum]
        maximum = max(result.values())
        argmax = [k for k, v in result.items() if v == maximum]
        result_range = maximum - minimum

        if minimum < 0:
            range_ratio = np.nan
        elif maximum == 0:
            # min==max==0
            range_ratio = 1
        else:
            range_ratio = minimum/maximum

        result[_MIN] = minimum
        result[_ARGMIN] = argmin
        result[_MAX] = maximum
        result[_ARGMAX] = argmax
        result[_RANGE] = result_range
        result[_RANGE_RATIO] = range_ratio
    except ValueError:
        # Nothing to do if the result type is not amenable to 'min' etc.
        pass

    # Evaluate the overall metric with the numpy arrays
    # Do this last so that the other statistics are calculated correctly
    if s_w is not None:
        result[_OVERALL] = metric_function(y_a, y_p, sample_weight=s_w, **kwargs)
    else:
        result[_OVERALL] = metric_function(y_a, y_p, **kwargs)

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
