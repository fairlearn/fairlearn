# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Functionality for computing metrics, with a particular focus on group metrics.

For our purpose, a metric is a function with signature
``f(y_true, y_pred, ....)``
where ``y_true`` are the set of true values and ``y_pred`` are
values predicted by a machine learning algorithm. Other
arguments may be present (most often sample weights), which will
affect how the metric is calculated.

The group metrics in this module have signatures
``g(y_true, y_pred, group_membership, ...)``
where ``group_membership`` is an array of values indicating
a group to which each pair of true and predicted values belong.
The metric is evaluated for the entire set of data, and also
for each subgroup identified in ``group_membership``.
"""

import sklearn.metrics as skm

from ._extra_metrics import (
    true_positive_rate, true_negative_rate,
    false_positive_rate, false_negative_rate,
    root_mean_squared_error, balanced_root_mean_squared_error,
    mean_overprediction, mean_prediction, mean_underprediction,
    selection_rate)

from ._metrics_engine import (
    make_metric_group_summary, group_summary,
    make_derived_metric,
    group_min_from_summary, group_max_from_summary,
    difference_from_summary, ratio_from_summary)

_transformations = {
    "difference": difference_from_summary,
    "ratio": ratio_from_summary,
    "group_min": group_min_from_summary,
    "group_max": group_max_from_summary,
}

# Base metrics and the variants that are implemented 
_METRICS_SPEC = [
    # base metrics from _extra_metrics
    (true_positive_rate, ["difference", "ratio"]),
    (true_negative_rate, []),
    (false_positive_rate, ["difference", "ratio"]),
    (false_negative_rate, []),
    (root_mean_squared_error, []),
    (balanced_root_mean_squared_error, []),
    (mean_overprediction, []),
    (mean_prediction, []),
    (mean_underprediction, []),
    (selection_rate, ["difference", "ratio", "group_min", "group_max"]),
 
    # base metrics from sklearn.metrics
    (skm.accuracy_score, ["difference", "ratio", "group_min", "group_max"]),
    (skm.confusion_matrix, []),
    (skm.precision_score, []),
    (skm.recall_score, []),
    (skm.roc_auc_score, []),
    (skm.zero_one_loss, []),
    (skm.mean_absolute_error, ["difference", "ratio", "group_min", "group_max"]),
    (skm.mean_squared_error, []),
    (skm.r2_score, []),
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
                _transformations[variant],
                metric_group_summary,
                name = derived_metric_name)
            derived_metric_dict[derived_metric_name] = derived_metric
    
    return metric_group_summary_dict, derived_metric_dict

_metric_group_summary_dict, _derived_metric_dict = _derive_metrics(_METRICS_SPEC)

globals().update(_metric_group_summary_dict)
globals().update(_derived_metric_dict)

# additional derived metrics

def demographic_parity_difference(y_true, y_pred, *, sensitive_features, sample_weight=None):
    r"""Calculate the demographic parity difference."""
    return selection_rate_difference(
        y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight)

def demographic_parity_ratio(y_true, y_pred, *, sensitive_features, sample_weight=None):
    r"""Calculate the demographic parity ratio."""
    return selection_rate_ratio(
        y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight)

def equalized_odds_difference(y_true, y_pred, *, sensitive_features, sample_weight=None):
    r"""Calculate the demographic parity difference."""
    return max(
        true_positive_rate_difference(
            y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight),
        false_positive_rate_difference(
            y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight))    

def equalized_odds_ratio(y_true, y_pred, *, sensitive_features, sample_weight=None):
    r"""Calculate the demographic parity ratio."""
    return min(
        true_positive_rate_ratio(
            y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight),
        false_positive_rate_ratio(
            y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight))    

_additional_derived = [
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equalized_odds_difference",
    "equalized_odds_ratio",
]

# -------------------------------------------

_extra_metrics = [
    "true_positive_rate",
    "true_negative_rate",
    "false_positive_rate",
    "false_negative_rate",
    "balanced_root_mean_squared_error",
    "mean_prediction",
    "mean_overprediction",
    "mean_underprediction",
    "selection_rate",
]


_engine = [
    "make_metric_group_summary",
    "group_summary",
    "make_derived_metric",
    "group_min_from_summary",
    "group_max_from_summary",
    "difference_from_summary",
    "ratio_from_summary"
]


__all__ = (
    _engine +
    _extra_metrics +
    list(_metric_group_summary_dict.keys()) +
    list(_derived_metric_dict.keys()) +
    _additional_derived)
