# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import sklearn.metrics as skm
from sklearn.utils import Bunch

from ._extra_metrics import (
    true_positive_rate, true_negative_rate,
    false_positive_rate, false_negative_rate,
    _root_mean_squared_error, _balanced_root_mean_squared_error,
    mean_prediction,
    selection_rate,
    _mean_overprediction,
    _mean_underprediction,
    )

from ._input_manipulations import _convert_to_ndarray_and_squeeze


_MESSAGE_SIZE_MISMATCH = "Array {0} is not the same size as {1}"

# Parameters to metrics that should be split according to sensitive features
_DEFAULT_INDEXED_PARAMS = {"sample_weight"}


def group_summary(metric_function, y_true, y_pred, *,
                  sensitive_features,
                  indexed_params=None,
                  **metric_params):
    r"""Apply a metric to each subgroup of a set of data.

    :param metric_function: Function with signature
        ``metric_function(y_true, y_pred, \*\*metric_params)``

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


def _function_name(func):
    if hasattr(func, '__name__'):
        return func.__name__
    else:
        return str(func)


class _MetricGroupSummaryCallable:
    r"""Callable that calculates the group summary of a metric.

    :param metric_function: A metric function with the signature
        ``metric_function(y_true, y_pred, **metric_params)``
    :type metric_function: func

    :param indexed_params: The names of parameters of ``metric_function`` that
        should be split according to ``sensitive_features`` in addition to ``y_true``
        and ``y_pred``. Defaults to ``None`` corresponding to ``['sample_weight']``.
    """

    def __init__(self, metric_function, indexed_params=None, name=None):
        self._metric_function = metric_function
        self._indexed_params = indexed_params
        if name is not None:
            self.__name__ = name

    def __repr__(self):
        if self._indexed_params is None:
            args_string = ""
        else:
            args_string = ", indexed_params={0}".format(self._indexed_params)
        return "make_metric_group_summary({0}{1})".format(
            _function_name(self._metric_function),
            args_string)

    def __call__(self, y_true, y_pred, *, sensitive_features, **metric_params):
        return group_summary(self._metric_function,
                             y_true, y_pred,
                             sensitive_features=sensitive_features,
                             indexed_params=self._indexed_params,
                             **metric_params)


class _DerivedMetricCallable:
    """Callable that calculates a derived metric.

    :param transformation_function: A transformation function with the signature
        ``transformation_function(summary)`` which can consume the result
        produced by ``summary_function`` (typically a ``Bunch`` with fields including
        ``overall`` and ``by_group``)
    :type transformation_function: func

    :param summary_function: A metric group summary function with the signature
        ``summary_function(y_true, y_pred, *, sensitive_features, **metric_params)``
    :type summary_function: func
    """

    def __init__(self, transformation_function, summary_function, name=None):
        self._transformation_function = transformation_function
        self._summary_function = summary_function
        if name is not None:
            self.__name__ = name

    def __repr__(self):
        return "make_derived_metric({0}, {1})".format(
                _function_name(self._transformation_function),
                _function_name(self._summary_function))

    def __call__(self, y_true, y_pred, *, sensitive_features, **metric_params):
        return self._transformation_function(self._summary_function(
                y_true, y_pred,
                sensitive_features=sensitive_features,
                **metric_params))


def make_metric_group_summary(metric_function, indexed_params=None, name=None):
    """Make a callable that calculates the group summary of a metric.

    :param metric_function: A metric function with the signature
        ``metric_function(y_true, y_pred, **metric_params)``
    :type metric_function: func

    :param indexed_params: The names of parameters of ``metric_function`` that
        should be split according to ``sensitive_features`` in addition to ``y_true``
        and ``y_pred``. Defaults to ``None`` corresponding to ``['sample_weight']``.

    :return: A callable object with the signature
        ``metric_group_summary(y_true, y_pred, *, sensitive_features, **metric_params)``
    :rtype: func
    """
    return _MetricGroupSummaryCallable(
        metric_function, indexed_params=indexed_params, name=name)


def make_derived_metric(transformation_function, summary_function, name=None):
    """Make a callable that calculates a derived metric from the group summary.

    :param transformation_function: A transformation function with the signature
        ``transformation_function(summary)``
    :type transformation_function: func

    :param summary_function: A metric group summary function with the signature
        ``summary_function(y_true, y_pred, *, sensitive_features, **metric_params)``
    :type summary_function: func

    :return: A callable object with the signature
        ``derived_metric(y_true, y_pred, *, sensitive_features, **metric_params)``
    :rtype: func
    """
    return _DerivedMetricCallable(
        transformation_function, summary_function, name=name)


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


TRANSFORMATIONS = {
    "difference": difference_from_summary,
    "ratio": ratio_from_summary,
    "group_min": group_min_from_summary,
    "group_max": group_max_from_summary,
}

# Base metrics and the variants that are implemented by the metrics engine
METRICS_SPEC = [
    # base metrics from _extra_metrics
    (true_positive_rate, ["difference", "ratio"]),
    (true_negative_rate, ["difference", "ratio"]),
    (false_positive_rate, ["difference", "ratio"]),
    (false_negative_rate, ["difference", "ratio"]),
    (selection_rate, ["difference", "ratio"]),
    (mean_prediction, []),
    (_root_mean_squared_error, []),
    (_balanced_root_mean_squared_error, []),
    (_mean_overprediction, []),
    (_mean_underprediction, []),

    # base metrics from sklearn.metrics
    (skm.confusion_matrix, []),
    (skm.accuracy_score, ["difference", "ratio", "group_min"]),
    (skm.zero_one_loss, ["difference", "ratio", "group_max"]),
    (skm.balanced_accuracy_score, ["group_min"]),
    (skm.precision_score, ["group_min"]),
    (skm.recall_score, ["group_min"]),
    (skm.roc_auc_score, ["group_min"]),
    (skm.mean_absolute_error, ["group_max"]),
    (skm.mean_squared_error, ["group_max"]),
    (skm.r2_score, ["group_min"]),
]


def _derive_metrics(metrics_spec):
    metric_group_summary_dict = {}
    derived_metric_dict = {}

    for base_metric, variants in metrics_spec:
        metric_group_summary_name = "{0}_group_summary".format(base_metric.__name__)
        metric_group_summary = make_metric_group_summary(
            base_metric,
            name=metric_group_summary_name)
        metric_group_summary_dict[metric_group_summary_name] = metric_group_summary

        for variant in variants:
            derived_metric_name = "{0}_{1}".format(base_metric.__name__, variant)
            derived_metric = make_derived_metric(
                TRANSFORMATIONS[variant],
                metric_group_summary,
                name=derived_metric_name)
            derived_metric_dict[derived_metric_name] = derived_metric

    return metric_group_summary_dict, derived_metric_dict


_metric_group_summary_dict, _derived_metric_dict = _derive_metrics(METRICS_SPEC)

globals().update(_metric_group_summary_dict)
globals().update(_derived_metric_dict)
