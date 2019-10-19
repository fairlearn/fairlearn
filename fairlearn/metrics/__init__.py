# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sklearn.metrics as skm

from .group_metric_result import GroupMetricResult  # noqa: F401
from .metrics_engine import metric_by_group, make_group_metric  # noqa: F401

from .extra_metrics import specificity_score

# Classification metrics
group_accuracy_score = make_group_metric(skm.accuracy_score)
group_confusion_matrix = make_group_metric(skm.confusion_matrix)
group_precision_score = make_group_metric(skm.precision_score)
group_recall_score = make_group_metric(skm.recall_score)
group_zero_one_loss = make_group_metric(skm.zero_one_loss)

# Regression metrics
group_max_error = make_group_metric(skm.max_error)
group_mean_absolute_error = make_group_metric(skm.mean_absolute_error)
group_mean_squared_error = make_group_metric(skm.mean_squared_error)
group_mean_squared_log_error = make_group_metric(skm.mean_squared_log_error)
group_median_absolute_error = make_group_metric(skm.median_absolute_error)
