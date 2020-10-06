# Copyright (c) Microsoft Corporation and Fairlearn contributors.
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

from ._extra_metrics import (  # noqa: F401
    true_positive_rate,
    true_negative_rate,
    false_positive_rate,
    false_negative_rate,
    _balanced_root_mean_squared_error,
    mean_prediction,
    selection_rate,
    _mean_overprediction,
    _mean_underprediction,
    )


_extra_metrics = [
    "true_positive_rate",
    "true_negative_rate",
    "false_positive_rate",
    "false_negative_rate",
    "balanced_root_mean_squared_error",
    "mean_prediction",
    "selection_rate",
    "_mean_overprediction",
    "_mean_underprediction",
]


_disparities = [
]


__all__ = (
    _extra_metrics +
    _disparities)
