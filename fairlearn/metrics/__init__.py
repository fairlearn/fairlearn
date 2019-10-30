# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This module contains the functionality for computing metrics, with a
particular focus on group metrics.

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
from ._extra_metrics import mean_overprediction, mean_prediction  # noqa: F401
from ._extra_metrics import mean_underprediction, miss_rate  # noqa: F401
from ._extra_metrics import selection_rate, specificity_score  # noqa: F401

from ._selection_rate import group_selection_rate  # noqa: F401

from ._skm_wrappers import group_accuracy_score, group_confusion_matrix  # noqa: F401
from ._skm_wrappers import group_precision_score, group_recall_score  # noqa: F401
from ._skm_wrappers import group_roc_auc_score, group_zero_one_loss  # noqa: F401
from ._skm_wrappers import group_mean_squared_error  # noqa: F401

from ._group_metric_result import GroupMetricResult  # noqa: F401
from ._metrics_engine import make_group_metric, metric_by_group  # noqa: F401

# -------------------------------------------

# Classification metrics
group_specificity_score = make_group_metric(specificity_score)
"""A grouped metric for the :any:`specificity_score`
"""

group_miss_rate = make_group_metric(miss_rate)
"""A grouped metric for the :any:`miss_rate`
"""

group_fallout_rate = make_group_metric(fallout_rate)
"""A grouped metric for the :any:`fallout_rate`
"""

# Regression metrics
group_max_error = make_group_metric(skm.max_error)
"""A grouped wrapper around the :any:`sklearn.metrics.max_error` routine
"""

group_mean_absolute_error = make_group_metric(skm.mean_absolute_error)
"""A grouped wrapper around the :any:`sklearn.metrics.mean_absolute_error` routine
"""

group_mean_squared_log_error = make_group_metric(skm.mean_squared_log_error)
"""A grouped wrapper around the :any:`sklearn.metrics.mean_squared_log_error` routine
"""

group_median_absolute_error = make_group_metric(skm.median_absolute_error)
"""A grouped wrapper around the :any:`sklearn.metrics.median_absolute_error` routine
"""

group_balanced_root_mean_squared_error = make_group_metric(
    balanced_root_mean_squared_error)
"""A grouped wrapper around the :any:`balanced_root_mean_squared_error` routine
"""

group_mean_prediction = make_group_metric(mean_prediction)
"""A grouped wrapper around the :any:`mean_prediction` routine
"""

group_mean_overprediction = make_group_metric(mean_overprediction)
"""A grouped wrapper around the :any:`mean_overprediction` routine
"""

group_mean_underprediction = make_group_metric(mean_underprediction)
"""A grouped wapper around the :any:`mean_underprediction` routine
"""

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
