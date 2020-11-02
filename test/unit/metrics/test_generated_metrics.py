# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest

import fairlearn.metrics as metrics

# ======================================================

y_true = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
sf_binary = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

derived_metric_results = {
    metrics.true_positive_rate_difference: 0.4,
    metrics.true_positive_rate_ratio: 0.6,
    metrics.true_negative_rate_difference: 0.083333333,
    metrics.true_negative_rate_ratio: 0.88888889,
    metrics.false_positive_rate_difference: 0.083333333,
    metrics.false_positive_rate_ratio: 0.75,
    metrics.false_negative_rate_difference: 0.4,
    metrics.false_negative_rate_ratio: 0,
    metrics.selection_rate_difference: 0.11111111,
    metrics.selection_rate_ratio: 0.8,
    metrics.accuracy_score_difference: 0.11111111,
    metrics.accuracy_score_ratio: 0.85714286,
    metrics.accuracy_score_group_min: 0.66666667,
    metrics.zero_one_loss_difference: 0.11111111,
    metrics.zero_one_loss_ratio: 0.66666667,
    metrics.zero_one_loss_group_max: 0.33333333,
    metrics.balanced_accuracy_score_group_min: 0.675,
    metrics.precision_score_group_min: 0.6,
    metrics.recall_score_group_min: 0.6,
    metrics.roc_auc_score_group_min: 0.675,
    metrics.mean_absolute_error_group_max: 0.33333333,
    metrics.mean_squared_error_group_max: 0.33333333,
    metrics.r2_score_group_min: -0.35,
    metrics.f1_score_group_max: 0.75,
    metrics.log_loss_group_min: 7.67546133,
}


# =======================================================

def test_dict_sizes():
    assert len(metrics._derived_metric_dict) == len(derived_metric_results)


@pytest.mark.parametrize("func", derived_metric_results.keys())
def test_derived_metrics_smoke(func):
    result = func(y_true, y_pred, sensitive_features=sf_binary)
    assert result == pytest.approx(derived_metric_results[func])
