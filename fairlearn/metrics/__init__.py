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

from ._extra_metrics import balanced_root_mean_squared_error, fallout_rate  # noqa: F401
from ._extra_metrics import root_mean_squared_error  # noqa: F401
from ._extra_metrics import mean_overprediction, mean_prediction  # noqa: F401
from ._extra_metrics import mean_underprediction, miss_rate  # noqa: F401
from ._extra_metrics import selection_rate, specificity_score  # noqa: F401

from ._metrics_engine import make_metric_group_summary, group_summary  # noqa: F401
from ._metrics_engine import group_min_from_summary, group_max_from_summary  # noqa: F401
from ._metrics_engine import difference_from_summary, ratio_from_summary  # noqa: F401

BASE_METRICS = [
    # _extra_metrics
    fallout_rate,
    miss_rate,
    specificity_score,
    root_mean_squared_error,
    balanced_root_mean_squared_error,
    mean_overprediction,
    mean_prediction,
    mean_underprediction,
    selection_rate,

    # sklearn metrics
    skm.accuracy_score,
    skm.confusion_matrix,
    skm.precision_score,
    skm.recall_score,
    skm.roc_auc_score,
    skm.zero_one_loss,
    skm.mean_squared_error,
    skm.r2_score,
    skm.max_error,
    skm.mean_absolute_error,
    skm.mean_squared_log_error,
    skm.median_absolute_error,
]

for metric in BASE_METRICS:
    metric_group_summary_name = "{0}_group_summary".format(metric.__name__)
    globals()[metric_group_summary_name] = make_metric_group_summary(metric)

# -------------------------------------------

_extra_metrics = [
    "balanced_root_mean_squared_error",
    "fallout_rate",
    "mean_prediction",
    "mean_overprediction",
    "mean_underprediction",
    "miss_rate",
    "selection_rate",
    "specificity_score"
]

_group_metrics = [
    "accuracy_score_group_summary",
    "balanced_root_mean_squared_error_group_summary",
    "confusion_matrix_group_summary",
    "fallout_rate_group_summary",
    "max_error_group_summary",
    "mean_absolute_error_group_summary",
    "mean_prediction_group_summary",
    "mean_overprediction_group_summary",
    "mean_squared_error_group_summary",
    "mean_squared_log_error_group_summary",
    "mean_underprediction_group_summary",
    "median_absolute_error_group_summary",
    "miss_rate_group_summary",
    "precision_score_group_summary",
    "r2_score_group_summary",
    "recall_score_group_summary",
    "roc_auc_score_group_summary",
    "root_mean_squared_error_group_summary",
    "selection_rate_group_summary",
    "specificity_score_group_summary",
    "zero_one_loss_group_summary"
]

_engine = [
    "make_metric_group_summary",
    "group_summary",
    "group_min_from_summary",
    "group_max_from_summary",
    "difference_from_summary",
    "ratio_from_summary"
]


__all__ = _engine + _extra_metrics + _group_metrics
