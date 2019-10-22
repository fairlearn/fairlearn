# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" This module contains the functionality for computing metrics
"""

import sklearn.metrics as skm

from .group_metric_result import GroupMetricResult  # noqa: F401
from .metrics_engine import metric_by_group, make_group_metric  # noqa: F401

from .skm_wrappers import group_accuracy_score, group_confusion_matrix  # noqa: F401
from .skm_wrappers import group_precision_score, group_recall_score  # noqa: F401
from .skm_wrappers import group_roc_auc_score, group_zero_one_loss  # noqa: F401
from .skm_wrappers import group_mean_squared_error  # noqa: F401

from .balanced_root_mean_squared_error import balanced_root_mean_squared_error
from .extra_metrics import specificity_score, miss_rate, fallout_rate
from .selection_rate import selection_rate, group_selection_rate  # noqa: F401

# Classification metrics
group_specificity_score = make_group_metric(specificity_score)
"""
A grouped metric for the specificity_score
"""

group_miss_rate = make_group_metric(miss_rate)
"""
A grouped metric for the miss_rate
"""

group_fallout_rate = make_group_metric(fallout_rate)

# Regression metrics
group_max_error = make_group_metric(skm.max_error)
group_mean_absolute_error = make_group_metric(skm.mean_absolute_error)
group_mean_squared_log_error = make_group_metric(skm.mean_squared_log_error)
group_median_absolute_error = make_group_metric(skm.median_absolute_error)

group_balanced_root_mean_squared_error = make_group_metric(balanced_root_mean_squared_error)
