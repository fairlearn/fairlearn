# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Functionality for computing metrics, with a particular focus on group metrics.

For our purpose, a metric is a function with signature
``f(y_true, y_pred, ....)``
where ``y_true`` are the set of true values and ``y_pred`` are
values predicted by a machine learning algorithm. Other
arguments may be present (most often sample weights), which will
affect how the metric is calculated.

This module provides the concept of a _grouped metric_.
This is a metric where in addition to ``y_true`` and ``y_pred``
values, the user provides information about group membership
for each sample.
For example, a user could provide a 'Gender' column, and the
grouped metric would contain separate results for the subgroups
'male', 'female' and 'nonbinary' indicated by that column.
This extends to multiple grouping columns, calculating the metric
for each combination of subgroups.
"""

import sklearn.metrics as skm

from ._derived_metrics import make_derived_metric  # noqa: F401
from ._disparities import (  # noqa: F401
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio)
from ._grouped_metric import GroupedMetric  # noqa: F401

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


# Derived metrics from SciKit-Learn
_standard_spn = ['sample_weight']

accuracy_score_group_min = make_derived_metric('group_min',
                                               skm.accuracy_score,
                                               sample_param_names=_standard_spn)
accuracy_score_group_max = make_derived_metric('group_max',
                                               skm.accuracy_score,
                                               sample_param_names=_standard_spn)
accuracy_score_difference = make_derived_metric('difference',
                                                skm.accuracy_score,
                                                sample_param_names=_standard_spn)
accuracy_score_ratio = make_derived_metric('ratio',
                                           skm.accuracy_score,
                                           sample_param_names=_standard_spn)

# ============================================
# Build list of items to be listed in the docs

_core = [
    "GroupedMetric",
    "make_derived_metric"
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

__all__ = _core + _disparities + _extra_metrics
