# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

import fairlearn.metrics as metrics

# ======================================================

Y_true = [0, 0, 0, 1, 1, 1, 0, 0]
Y_pred = [1, 0, 1, 0, 1, 0, 0, 1]
groups = [3, 4, 1, 0, 0, 0, 3, 2]
weight = [1, 2, 3, 1, 2, 3, 4, 2]

# =======================================================

# Define as a dictionary so that the actual name can be seen
# when pytest builds the tests
supported_metrics_weighted = [metrics.group_accuracy_score,
                              metrics.group_precision_score,
                              metrics.group_recall_score,
                              metrics.group_zero_one_loss,
                              metrics.group_mean_squared_error]

metrics_no_sample_weights = [metrics.group_max_error,
                             metrics.group_mean_absolute_error,
                             metrics.group_mean_squared_log_error,
                             metrics.group_median_absolute_error]

supported_metrics_unweighted = supported_metrics_weighted + metrics_no_sample_weights

# =======================================================


@pytest.mark.parametrize("group_metric_func", supported_metrics_unweighted)
def test_metric_unweighted(group_metric_func):
    result = group_metric_func(Y_true, Y_pred, groups)
    # We don't really care about the numbers (sklearn is responsible)
    # We just want to make sure we got a result
    assert len(result.by_group) == 5


@pytest.mark.parametrize("group_metric_func", supported_metrics_weighted)
def test_metric_weighted(group_metric_func):
    result = group_metric_func(Y_true, Y_pred, groups, weight)
    assert len(result.by_group) == 5
