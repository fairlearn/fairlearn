# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import numpy as np

import fairlearn.metrics as metrics

# ======================================================

y_true = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
sf_binary = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

metric_group_summary_results = {
    metrics.true_positive_rate_group_summary: {
        "overall": 0.75, "by_group": {0: 1, 1: 0.6}},
    metrics.true_negative_rate_group_summary: {
        "overall": 0.7, "by_group": {0: 0.66666667, 1: 0.75}},
    metrics.false_positive_rate_group_summary: {
        "overall": 0.3, "by_group": {0: 0.33333333, 1: 0.25}},
    metrics.false_negative_rate_group_summary: {
        "overall": 0.25, "by_group": {0: 0, 1: 0.4}},
    metrics._root_mean_squared_error_group_summary: {
        "overall": 0.52704628, "by_group": {0: 0.47140452, 1: 0.57735027}},
    metrics._balanced_root_mean_squared_error_group_summary: {
        "overall": 0.52386128, "by_group": {0: 0.28867513, 1: 0.56622777}},
    metrics.mean_prediction_group_summary: {
        "overall": 0.5, "by_group": {0: 0.55555556, 1: 0.44444444}},
    metrics.selection_rate_group_summary: {
        "overall": 0.5, "by_group": {0: 0.55555556, 1: 0.44444444}},
    metrics._mean_overprediction_group_summary: {
        "overall": 0.16666667, "by_group": {0: 0.22222222, 1: 0.11111111}},
    metrics._mean_underprediction_group_summary: {
        "overall": 0.11111111, "by_group": {0: -0, 1: 0.22222222}},
    metrics.accuracy_score_group_summary: {
        "overall": 0.72222222, "by_group": {0: 0.77777778, 1: 0.66666667}},
    metrics.balanced_accuracy_score_group_summary: {
        "overall": 0.725, "by_group": {0: 0.83333333, 1: 0.675}},
    metrics.confusion_matrix_group_summary: {
        'overall': np.array([[7, 3], [2, 6]]),
        'by_group': {0: np.array([[4, 2], [0, 3]]), 1: np.array([[3, 1], [2, 3]])}},
    metrics.precision_score_group_summary: {
        "overall": 0.66666667, "by_group": {0: 0.6, 1: 0.75}},
    metrics.recall_score_group_summary: {
        "overall": 0.75, "by_group": {0: 1, 1: 0.6}},
    metrics.roc_auc_score_group_summary: {
        "overall": 0.725, "by_group": {0: 0.83333333, 1: 0.675}},
    metrics.zero_one_loss_group_summary: {
        "overall": 0.27777778, "by_group": {0: 0.22222222, 1: 0.33333333}},
    metrics.mean_absolute_error_group_summary: {
        "overall": 0.27777778, "by_group": {0: 0.22222222, 1: 0.33333333}},
    metrics.mean_squared_error_group_summary: {
        "overall": 0.27777778, "by_group": {0: 0.22222222, 1: 0.33333333}},
    metrics.r2_score_group_summary: {
        "overall": -0.125, "by_group": {0: 0, 1: -0.35}},
}

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
}


# =======================================================

def test_dict_sizes():
    assert len(metrics._metric_group_summary_dict) == len(metric_group_summary_results)
    assert len(metrics._derived_metric_dict) == len(derived_metric_results)


@pytest.mark.parametrize("func", metric_group_summary_results.keys())
def test_metric_group_summary_smoke(func):
    result = func(y_true, y_pred, sensitive_features=sf_binary)
    assert result.overall == pytest.approx(metric_group_summary_results[func]["overall"])
    assert len(result.by_group) == 2
    assert result.by_group[0] == pytest.approx(metric_group_summary_results[func]["by_group"][0])
    assert result.by_group[1] == pytest.approx(metric_group_summary_results[func]["by_group"][1])


@pytest.mark.parametrize("func", derived_metric_results.keys())
def test_derived_metrics_smoke(func):
    result = func(y_true, y_pred, sensitive_features=sf_binary)
    assert result == pytest.approx(derived_metric_results[func])
