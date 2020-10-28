# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Functionality for computing metrics, with a particular focus on disaggregated metrics.

For our purpose, a metric is a function with signature
``f(y_true, y_pred, ....)``
where ``y_true`` are the set of true values and ``y_pred`` are
values predicted by a machine learning algorithm. Other
arguments may be present (most often sample weights), which will
affect how the metric is calculated.

This module provides the concept of a *disaggregated metric*.
This is a metric where in addition to ``y_true`` and ``y_pred``
values, the user provides information about group membership
for each sample.
For example, a user could provide a 'Gender' column, and the
disaggregated metric would contain separate results for the subgroups
'male', 'female' and 'nonbinary' indicated by that column.
The underlying metric function is evaluated for each of these three
subgroups.
This extends to multiple grouping columns, calculating the metric
for each combination of subgroups.
"""


from ._metrics_engine import _derived_metric_dict

from ._disparities import (  # noqa: F401
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio)
from ._metric_frame import MetricFrame  # noqa: F401

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


# Add the generated metrics of the form and
# `<metric>_{difference,ratio,group_min,group_max`
globals().update(_derived_metric_dict)

# ============================================
# Build list of items to be listed in the docs

_core = [
    "MetricFrame"
]

_disparities = [
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equalized_odds_difference",
    "equalized_odds_ratio"
]

_extra_metrics = [
    "true_positive_rate",
    "true_negative_rate",
    "false_positive_rate",
    "false_negative_rate",
    "mean_prediction",
    "selection_rate",
]

__all__ = _core + _disparities + _extra_metrics + list(_derived_metric_dict.keys())
