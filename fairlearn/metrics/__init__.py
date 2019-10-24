# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This module contains the functionality for computing metrics
"""

from ._extra_metrics import balanced_root_mean_squared_error, fallout_rate  # noqa: F401
from ._extra_metrics import mean_overprediction, mean_prediction  # noqa: F401
from ._extra_metrics import mean_underprediction, miss_rate  # noqa: F401
from ._extra_metrics import selection_rate, specificity_score  # noqa: F401

from ._group_metrics import group_accuracy_score, group_balanced_root_mean_squared_error  # noqa: F401, E501
from ._group_metrics import group_confusion_matrix, group_fallout_rate  # noqa: F401
from ._group_metrics import group_max_error, group_mean_absolute_error  # noqa: F401
from ._group_metrics import group_mean_prediction, group_mean_overprediction  # noqa: F401
from ._group_metrics import group_mean_squared_error, group_mean_underprediction  # noqa: F401
from ._group_metrics import group_mean_squared_log_error, group_median_absolute_error  # noqa: F401
from ._group_metrics import group_miss_rate, group_precision_score  # noqa: F401
from ._group_metrics import group_recall_score, group_roc_auc_score  # noqa: F401
from ._group_metrics import group_selection_rate,  group_specificity_score  # noqa: F401
from ._group_metrics import group_zero_one_loss  # noqa: F401

from ._group_metric_result import GroupMetricResult  # noqa: F401
from ._metrics_engine import make_group_metric, metric_by_group  # noqa: F401

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
    "group_recall_score",
    "group_roc_auc_score",
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
