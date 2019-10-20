# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sklearn.metrics as skm

from .group_metric_result import GroupMetricResult  # noqa: F401
from .metrics_engine import metric_by_group, make_group_metric  # noqa: F401

from .skm_wrappers import group_accuracy_score, group_confusion_matrix  # noqa: F401
from .skm_wrappers import group_precision_score, group_recall_score  # noqa: F401
from .skm_wrappers import group_roc_auc_score, group_zero_one_loss  # noqa: F401

from .extra_metrics import specificity_score, miss_score, fallout_score

# Classification metrics
group_specificity_score = make_group_metric(specificity_score)
group_miss_score = make_group_metric(miss_score)
group_fallout_score = make_group_metric(fallout_score)

# Regression metrics
group_max_error = make_group_metric(skm.max_error)
group_mean_absolute_error = make_group_metric(skm.mean_absolute_error)
group_mean_squared_error = make_group_metric(skm.mean_squared_error)
group_mean_squared_log_error = make_group_metric(skm.mean_squared_log_error)
group_median_absolute_error = make_group_metric(skm.median_absolute_error)
