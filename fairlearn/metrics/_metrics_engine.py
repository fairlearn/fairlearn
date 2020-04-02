# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from ._input_manipulations import _convert_to_ndarray_and_squeeze
from sklearn.utils import Bunch

_MESSAGE_SIZE_MISMATCH = "Array {0} is not the same size as {1}"

# Parameters to metrics that should be split according to sensitive features
_DEFAULT_INDEXED_PARAMS = {"sample_weight"}


def group_summary(metric_function, y_true, y_pred,
                  sensitive_features,
                  indexed_params=None,
                  **metric_params):
    r"""Apply a metric to each subgroup of a set of data.

    :param metric_function: Function ``(y_true, y_pred, \*\*metric_params)``

    :param y_true: Array of ground-truth values

    :param y_pred: Array of predicted values

    :param sensitive_features: Array indicating the group to which each input value belongs

    :param indexed_params: Names of ``metric_function`` parameters that
        should be split according to ``sensitive_features`` in addition to ``y_true``
        and ``y_pred``. Defaults to ``None`` corresponding to ``{"sample_weight"}``.

    :param \*\*metric_params: Optional arguments to be passed to the ``metric_function``

    :return: Object containing the result of applying ``metric_function`` to the entire dataset
        and to each group identified in ``sensitive_features``
    :rtype: :py:class:`sklearn.utils.Bunch` with the fields ``overall`` and ``by_group``
    """
    _check_array_sizes(y_true, y_pred, 'y_true', 'y_pred')
    _check_array_sizes(y_true, sensitive_features, 'y_true', 'sensitive_features')

    # Make everything a numpy array
    # This allows for fast slicing of the groups
    y_t = _convert_to_ndarray_and_squeeze(y_true)
    y_p = _convert_to_ndarray_and_squeeze(y_pred)
    s_f = _convert_to_ndarray_and_squeeze(sensitive_features)

    # Evaluate the overall metric with the numpy arrays
    # This ensures consistency in how metric_function is called
    result_overall = metric_function(
        y_t, y_p,
        **_check_metric_params(y_t, metric_params, indexed_params))

    groups = np.unique(s_f)
    result_by_group = {}
    for group in groups:
        group_indices = (group == s_f)
        result_by_group[group] = metric_function(
            y_t[group_indices], y_p[group_indices],
            **_check_metric_params(y_t, metric_params, indexed_params, group_indices))

    return Bunch(overall=result_overall, by_group=result_by_group)


# This loosely follows the pattern of _check_fit_params in
# sklearn/utils/validation.py
def _check_metric_params(y_true, metric_params,
                         indexed_params=None, indices=None):
    metric_params_validated = {}
    if indexed_params is None:
        indexed_params = _DEFAULT_INDEXED_PARAMS
    for param_key, param_value in metric_params.items():
        if (param_key in indexed_params and param_value is not None):
            _check_array_sizes(y_true, param_value, 'y_true', param_key)
            p_v = _convert_to_ndarray_and_squeeze(param_value)
            if indices is not None:
                p_v = p_v[indices]
            metric_params_validated[param_key] = p_v
        else:
            metric_params_validated[param_key] = param_value

    return metric_params_validated


def make_group_metric(metric_function, indexed_params=None):
    """Turn a regular metric into a grouped metric.

    :param metric_function: The function to be wrapped. This must have signature
        ``(y_true, y_pred, **metric_params)``
    :type metric_function: func

    :param indexed_params: Names of ``metric_function`` parameters that
        should be split according to ``sensitive_features`` in addition to ``y_true``
        and ``y_pred``. Defaults to ``None`` corresponding to ``{"sample_weight"}``.

    :return: A wrapped version of the supplied ``metric_function``. It will have
        signature ``(y_true, y_pred, sensitive_features, **metric_params)``.
    :rtype: func
    """
    def wrapper(y_true, y_pred, sensitive_features, **metric_params):
        return group_summary(metric_function,
                             y_true,
                             y_pred,
                             sensitive_features=sensitive_features,
                             indexed_params=indexed_params,
                             **metric_params)

    # Improve the name of the returned function
    wrapper.__name__ = "group_{0}".format(metric_function.__name__)

    return wrapper


def difference_from_summary(summary):
    """Calculate the difference between the maximum and minimum metric value across groups.

    :param summary: A group metric summary

    :return: The difference between the maximum and the minimum group-level
        metrics described in ``summary``.
    :rtype: float
    """
    return group_max_from_summary(summary) - group_min_from_summary(summary)


def ratio_from_summary(summary):
    """Calculate the ratio between the maximum and minimum metric value across groups.

    :param summary: A group metric summary

    :return: The ratio between the maximum and the minimum group-level
        metrics described in ``summary``.
    :rtype: float
    """
    group_min = group_min_from_summary(summary)
    group_max = group_max_from_summary(summary)
    if group_min < 0.0:
        return np.nan
    elif group_max == 0.0:
        return 1.0
    else:
        return group_min / group_max


def group_min_from_summary(summary):
    """Retrieve the minimum group-level metric value from group summary.

    :param summary: A group metric summary

    :return: The minimum group-level metric value across all groups in ``summary``.
    :rtype: float
    """
    return min(summary.by_group.values())


def group_max_from_summary(summary):
    """Retrieve the minimum group-level metric value from group summary.

    :param summary: A group metric summary

    :return: The maximum group-level metric value across all groups in ``summary``.
    :rtype: float
    """
    return max(summary.by_group.values())


def _check_array_sizes(a, b, a_name, b_name):
    if len(a) != len(b):
        raise ValueError(_MESSAGE_SIZE_MISMATCH.format(b_name, a_name))
