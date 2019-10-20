# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import numpy as np
import sklearn.metrics as skm

import fairlearn.metrics as metrics

# ======================================================

a = "a"
b = "b"
c = "c"

Y_true = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
Y_pred = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

Y_t3 = [a, b, c, c, c, b, b, b, c, c, a, a, a, a, c, b, c, c]
Y_p3 = [b, c, c, c, b, b, b, b, b, c, a, a, c, b, a, a, b, a]

groups = [3, 4, 1, 0, 0, 0, 3, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
weight = [1, 2, 3, 1, 2, 3, 4, 2, 3, 3, 2, 1, 2, 3, 1, 2, 3, 4]

# =======================================================

# Define as a dictionary so that the actual name can be seen
# when pytest builds the tests

supported_metrics_weighted = [(skm.accuracy_score, metrics.group_accuracy_score),
                              (skm.confusion_matrix, metrics.group_confusion_matrix)]

supported_metrics_weighted_binary = [(skm.precision_score, metrics.group_precision_score)]
supported_metrics_weighted_binary = supported_metrics_weighted_binary + supported_metrics_weighted


metrics_no_sample_weights = [metrics.group_max_error,
                             metrics.group_mean_absolute_error,
                             metrics.group_mean_squared_log_error,
                             metrics.group_median_absolute_error]

supported_metrics_unweighted = metrics_no_sample_weights
# supported_metrics_weighted + metrics_no_sample_weights

# =======================================================


@pytest.mark.parametrize("group_metric_func", supported_metrics_unweighted)
def test_metric_unweighted(group_metric_func):
    result = group_metric_func(Y_true, Y_pred, groups)
    # We don't really care about the numbers (sklearn is responsible)
    # We just want to make sure we got a result
    assert len(result.by_group) == 5


@pytest.mark.parametrize("func_tuple", supported_metrics_weighted_binary)
def test_metric_weighted(func_tuple):
    metric_func = func_tuple[0]
    group_metric_func = func_tuple[1]

    result = group_metric_func(Y_true, Y_pred, groups, sample_weight=weight)
    assert len(result.by_group) == 5

    expected_overall = metric_func(Y_true, Y_pred, sample_weight=weight)
    if isinstance(expected_overall, np.ndarray):
        assert np.array_equal(expected_overall, result.overall)
    else:
        assert expected_overall == result.overall


@pytest.mark.parametrize("func_tuple", supported_metrics_weighted)
def test_metric_weighted_ternary(func_tuple):
    metric_func = func_tuple[0]
    group_metric_func = func_tuple[1]

    result = group_metric_func(Y_t3, Y_p3, groups, sample_weight=weight)
    assert len(result.by_group) == 5
    expected_overall = metric_func(Y_t3, Y_p3, sample_weight=weight)

    if isinstance(expected_overall, np.ndarray):
        assert np.array_equal(expected_overall, result.overall)
    else:
        assert expected_overall == result.overall


# ======================================================================================

def test_group_accuracy_score_unnormalized():
    result = metrics.group_accuracy_score(Y_true, Y_pred, groups, False)

    expected_overall = skm.accuracy_score(Y_true, Y_pred, False)

    assert result.overall == expected_overall


# ======================================================================================

def test_group_confusion_matrix_labels():
    labels = [0, 4]

    result = metrics.group_confusion_matrix(Y_true, Y_pred, groups, labels=labels)
    expected_overall = skm.confusion_matrix(Y_true, Y_pred, labels=labels)

    assert np.array_equal(result.overall, expected_overall)
