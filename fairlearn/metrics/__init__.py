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

from ._metrics_engine import (  # noqa: F401
    make_metric_group_summary, group_summary,
    make_derived_metric,
    group_min_from_summary, group_max_from_summary,
    difference_from_summary, ratio_from_summary,
    _metric_group_summary_dict, _derived_metric_dict)

from ._disparities import (  # noqa: F401
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
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


_metrics_engine = [
    "make_metric_group_summary",
    "group_summary",
    "make_derived_metric",
    "group_min_from_summary",
    "group_max_from_summary",
    "difference_from_summary",
    "ratio_from_summary"
]


# Add the generated metrics of the form `<metric>_group summary` and
# `<metric>_{difference,ratio,group_min,group_max`
globals().update(_metric_group_summary_dict)
globals().update(_derived_metric_dict)


_disparities = [
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equalized_odds_difference",
    "equalized_odds_ratio",
]


__all__ = (
    _extra_metrics +
    _metrics_engine +
    list(_metric_group_summary_dict.keys()) +
    list(_derived_metric_dict.keys()) +
    _disparities)
