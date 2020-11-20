# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest

import fairlearn.metrics as metrics

# ======================================================

y_true = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
sf_binary = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

derived_metric_results = {
    "true_positive_rate_difference": {'method': 'to_overall', 'expected': 0.25},
    "true_positive_rate_ratio": {'method': 'to_overall', 'expected': 0.75},
    "true_negative_rate_difference": {'method': 'between_groups', 'expected': 0.083333333},
    "true_negative_rate_ratio": {'method': 'between_groups', 'expected': 0.88888889},
    "false_positive_rate_difference": {'method': 'between_groups', 'expected': 0.083333333},
    "false_positive_rate_ratio": {'method': 'between_groups', 'expected': 0.75},
    "false_negative_rate_difference": {'method': 'between_groups', 'expected': 0.4},
    "false_negative_rate_ratio": {'method': 'between_groups', 'expected': 0},
    "selection_rate_difference": {'method': 'between_groups', 'expected': 0.11111111},
    "selection_rate_ratio": {'method': 'between_groups', 'expected': 0.8},
    "accuracy_score_difference": {'method': 'between_groups', 'expected': 0.11111111},
    "accuracy_score_ratio": {'method': 'between_groups', 'expected': 0.85714286},
    "accuracy_score_group_min": {'method': None, 'expected': 0.66666667},
    "zero_one_loss_difference": {'method': 'between_groups', 'expected': 0.11111111},
    "zero_one_loss_ratio": {'method': 'between_groups', 'expected': 0.66666667},
    "zero_one_loss_group_max": {'method': None, 'expected': 0.33333333},
    "balanced_accuracy_score_group_min": {'method': None, 'expected': 0.675},
    "precision_score_group_min": {'method': None, 'expected': 0.6},
    "recall_score_group_min": {'method': None, 'expected': 0.6},
    "roc_auc_score_group_min": {'method': None, 'expected': 0.675},
    "mean_absolute_error_group_max": {'method': None, 'expected': 0.33333333},
    "mean_squared_error_group_max": {'method': None, 'expected': 0.33333333},
    "r2_score_group_min": {'method': None, 'expected': -0.35},
    "f1_score_group_min": {'method': None, 'expected': 0.66666666667},
    "log_loss_group_max": {'method': None, 'expected': 11.5130143},
}


# =======================================================

def test_dict_sizes():
    assert len(metrics._generated_metric_dict) == 25


@pytest.mark.parametrize("func_name", derived_metric_results.keys())
def test_generated_metrics_smoke(func_name):
    func = getattr(metrics, func_name)
    assert callable(func)
    result = func(y_true, y_pred,
                  sensitive_features=sf_binary,
                  method=derived_metric_results[func_name]['method'])
    assert result == pytest.approx(derived_metric_results[func_name]['expected'])


@pytest.mark.parametrize("method", ['between_groups', 'to_overall'])
def test_against_demographic_parity_difference(method):
    expected = metrics.demographic_parity_difference(y_true, y_pred,
                                                     sensitive_features=sf_binary,
                                                     method=method)
    actual = metrics.selection_rate_difference(y_true, y_pred,
                                               sensitive_features=sf_binary,
                                               method=method)
    assert expected == actual


@pytest.mark.parametrize("method", ['between_groups', 'to_overall'])
def test_against_demographic_parity_ratio(method):
    expected = metrics.demographic_parity_ratio(y_true, y_pred,
                                                sensitive_features=sf_binary,
                                                method=method)
    actual = metrics.selection_rate_ratio(y_true, y_pred,
                                          sensitive_features=sf_binary,
                                          method=method)
    assert expected == actual
