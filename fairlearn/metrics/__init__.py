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

from ._extra_metrics import balanced_root_mean_squared_error, fallout_rate  # noqa: F401
from ._extra_metrics import mean_overprediction, mean_prediction  # noqa: F401
from ._extra_metrics import mean_underprediction, miss_rate  # noqa: F401
from ._extra_metrics import selection_rate, specificity_score  # noqa: F401
from ._extra_metrics import group_fallout_rate  # noqa: F401
from ._extra_metrics import group_miss_rate  # noqa: F401
from ._extra_metrics import group_specificity_score  # noqa: F401
from ._extra_metrics import group_balanced_root_mean_squared_error  # noqa: F401
from ._extra_metrics import group_mean_overprediction  # noqa: F401
from ._extra_metrics import group_mean_prediction  # noqa: F401
from ._extra_metrics import group_mean_underprediction  # noqa: F401

from ._selection_rate import group_selection_rate  # noqa: F401

from ._skm_wrappers import group_accuracy_score, group_confusion_matrix  # noqa: F401
from ._skm_wrappers import group_precision_score, group_recall_score  # noqa: F401
from ._skm_wrappers import group_roc_auc_score, group_zero_one_loss  # noqa: F401
from ._skm_wrappers import group_max_error  # noqa: F401
from ._skm_wrappers import group_mean_absolute_error  # noqa: F401
from ._skm_wrappers import group_mean_squared_error  # noqa: F401
from ._skm_wrappers import group_mean_squared_log_error  # noqa: F401
from ._skm_wrappers import group_median_absolute_error  # noqa: F401
from ._skm_wrappers import group_root_mean_squared_error  # noqa: F401
from ._skm_wrappers import group_r2_score  # noqa: F401

from ._group_metric_result import GroupMetricResult  # noqa: F401
from ._metrics_engine import make_group_metric, metric_by_group  # noqa: F401

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
    "group_accuracy_score",
    "group_balanced_root_mean_squared_error",
    "group_confusion_matrix",
    "group_fallout_rate",
    "group_max_error",
    "group_mean_absolute_error",
    "group_mean_prediction",
    "group_mean_overprediction",
    "group_mean_squared_error",
    "group_mean_squared_log_error",
    "group_mean_underprediction",
    "group_median_absolute_error",
    "group_miss_rate",
    "group_precision_score",
    "group_r2_score",
    "group_recall_score",
    "group_roc_auc_score",
    "group_root_mean_squared_error",
    "group_selection_rate",
    "group_specificity_score",
    "group_zero_one_loss"
]

_engine = [
    "GroupMetricResult",
    "make_group_metric",
    "metric_by_group"
]


__all__ = _engine + _extra_metrics + _group_metrics
