# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sklearn.metrics as skm

from ._balanced_root_mean_squared_error import balanced_root_mean_squared_error
from .extra_metrics import specificity_score, miss_rate, fallout_rate
from ._selection_rate import group_selection_rate  # noqa: F401
from ._mean_predictions import mean_prediction, mean_overprediction, mean_underprediction

from ._metrics_engine import make_group_metric

from ._skm_wrappers import group_accuracy_score, group_confusion_matrix  # noqa: F401
from ._skm_wrappers import group_precision_score, group_recall_score  # noqa: F401
from ._skm_wrappers import group_roc_auc_score, group_zero_one_loss  # noqa: F401
from ._skm_wrappers import group_mean_squared_error  # noqa: F401

# Classification metrics
group_specificity_score = make_group_metric(specificity_score)
"""A grouped metric for the specificity score
"""

group_miss_rate = make_group_metric(miss_rate)
"""A grouped metric for the miss rate
"""

group_fallout_rate = make_group_metric(fallout_rate)
"""A grouped metric for the fallout rate
"""

# Regression metrics
group_max_error = make_group_metric(skm.max_error)
"""A grouped wrapper around the sklearn.metrics.max_error routine
"""

group_mean_absolute_error = make_group_metric(skm.mean_absolute_error)
"""A grouped wrapper around the sklearn.metrics.mean_absolute_error
"""

group_mean_squared_log_error = make_group_metric(skm.mean_squared_log_error)
"""A grouped wrapper around the sklearn.metrics.mean_squared_log_error routine
"""

group_median_absolute_error = make_group_metric(skm.median_absolute_error)
"""A grouped wrapper around the sklearn.metrics.median_absolute_error
"""

group_balanced_root_mean_squared_error = make_group_metric(balanced_root_mean_squared_error)
"""A grouped wrapper around the balanced_root_mean_squared_error routine
"""

group_mean_prediction = make_group_metric(mean_prediction)
"""A grouped wrapper around the mean_prediction routine
"""

group_mean_overprediction = make_group_metric(mean_overprediction)
"""A grouped wrapper around the mean_overprediction routine
"""

group_mean_underprediction = make_group_metric(mean_underprediction)
"""A grouped wapper around the mean_underprediction routine
"""
