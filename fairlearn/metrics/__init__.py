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

import sys as _sys

from ._base_metrics import count  # noqa: F401
from ._base_metrics import false_negative_rate  # noqa: F401
from ._base_metrics import false_positive_rate  # noqa: F401
from ._base_metrics import mean_prediction  # noqa: F401
from ._base_metrics import selection_rate  # noqa: F401
from ._base_metrics import true_negative_rate  # noqa: F401
from ._base_metrics import true_positive_rate  # noqa: F401; noqa: F401
from ._fairness_metrics import demographic_parity_difference  # noqa: F401
from ._fairness_metrics import demographic_parity_ratio  # noqa: F401
from ._fairness_metrics import equalized_odds_difference  # noqa: F401
from ._fairness_metrics import equalized_odds_ratio  # noqa: F401
from ._generated_metrics import _generated_metric_dict  # noqa: F401
from ._make_derived_metric import make_derived_metric  # noqa: F401
from ._metric_frame import MetricFrame  # noqa: F401
from ._plot_model_comparison import plot_model_comparison  # noqa: F401

# Add the generated metrics of the form and
# `<metric>_{difference,ratio,group_min,group_max`
_module_obj = _sys.modules[__name__]
for _name, _func in _generated_metric_dict.items():
    setattr(_module_obj, _name, _func)

# ============================================
# Build list of items to be listed in the docs

_core = ["MetricFrame", "make_derived_metric", "plot_model_comparison"]

_fairness = [
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equalized_odds_difference",
    "equalized_odds_ratio",
]

_base_metrics = [
    "true_positive_rate",
    "true_negative_rate",
    "false_positive_rate",
    "false_negative_rate",
    "mean_prediction",
    "selection_rate",
    "count",
]

__all__ = (
    _core + _fairness + _base_metrics + list(sorted(_generated_metric_dict.keys()))
)
