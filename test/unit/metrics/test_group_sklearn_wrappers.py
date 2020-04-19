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

Y_true_ternary = [a, b, c, c, c, b, b, b, c, c, a, a, a, a, a, b, c, c]
Y_pred_ternary = [b, c, c, c, b, b, b, b, b, c, a, a, c, a, a, b, c, c]

groups = [3, 4, 1, 0, 0, 0, 3, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
weight = [1, 2, 3, 1, 2, 3, 4, 2, 3, 3, 2, 1, 2, 3, 1, 2, 3, 4]
group2 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

# =======================================================

# Define as a dictionary so that the actual name can be seen
# when pytest builds the tests

supported_metrics_weighted = [
    (skm.accuracy_score, metrics.accuracy_score_group_summary),
    (skm.confusion_matrix, metrics.confusion_matrix_group_summary),
    (skm.zero_one_loss, metrics.zero_one_loss_group_summary)]

# The following only work with binary data when called with their default arguments
supported_metrics_weighted_binary = [
    (skm.precision_score, metrics.precision_score_group_summary),
    (skm.recall_score, metrics.recall_score_group_summary),
    (skm.roc_auc_score, metrics.roc_auc_score_group_summary),
    (skm.mean_squared_error, metrics.mean_squared_error_group_summary),
    (skm.r2_score, metrics.r2_score_group_summary)]
supported_metrics_weighted_binary = supported_metrics_weighted_binary + supported_metrics_weighted


metrics_no_sample_weights = [
    (skm.mean_absolute_error, metrics.mean_absolute_error_group_summary)]

supported_metrics_unweighted = metrics_no_sample_weights + supported_metrics_weighted_binary

# =======================================================


@pytest.mark.parametrize("func_tuple", supported_metrics_unweighted)
def test_metric_unweighted(func_tuple):
    metric_func = func_tuple[0]
    group_metric_func = func_tuple[1]

    result = group_metric_func(Y_true, Y_pred, sensitive_features=groups)
    # We don't really care about the numbers (sklearn is responsible)
    # We just want to make sure we got a result
    assert len(result.by_group) == 5

    expected_overall = metric_func(Y_true, Y_pred)
    if isinstance(expected_overall, np.ndarray):
        assert np.array_equal(expected_overall, result.overall)
    else:
        assert expected_overall == result.overall


@pytest.mark.parametrize("func_tuple", supported_metrics_weighted_binary)
def test_metric_weighted(func_tuple):
    metric_func = func_tuple[0]
    group_metric_func = func_tuple[1]

    result = group_metric_func(
        Y_true, Y_pred, sensitive_features=groups, sample_weight=weight)
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

    result = group_metric_func(
        Y_true_ternary, Y_pred_ternary, sensitive_features=groups, sample_weight=weight)
    assert len(result.by_group) == 5
    expected_overall = metric_func(Y_true_ternary, Y_pred_ternary, sample_weight=weight)

    if isinstance(expected_overall, np.ndarray):
        assert np.array_equal(expected_overall, result.overall)
    else:
        assert expected_overall == result.overall


# ======================================================================================

def test_group_accuracy_score_unnormalized():
    result = metrics.accuracy_score_group_summary(
        Y_true, Y_pred, sensitive_features=groups, normalize=False)

    expected_overall = skm.accuracy_score(Y_true, Y_pred, False)

    assert result.overall == expected_overall


# ======================================================================================

def test_group_confusion_matrix_labels():
    labels = [0, 4]

    result = metrics.confusion_matrix_group_summary(
        Y_true, Y_pred, sensitive_features=groups, labels=labels)
    expected_overall = skm.confusion_matrix(Y_true, Y_pred, labels=labels)

    assert np.array_equal(result.overall, expected_overall)


# ======================================================================================

def test_group_precision_score_ternary():
    result = metrics.precision_score_group_summary(
        Y_true_ternary, Y_pred_ternary, sensitive_features=group2, average=None)
    expected_overall = skm.precision_score(Y_true_ternary, Y_pred_ternary, average=None)

    assert np.array_equal(result.overall, expected_overall)


def test_group_precision_score_pos_label():
    result = metrics.precision_score_group_summary(
        Y_true, Y_pred, sensitive_features=groups, pos_label=0)
    expected_overall = skm.precision_score(Y_true, Y_pred, pos_label=0)

    assert np.array_equal(result.overall, expected_overall)

# ======================================================================================


def test_group_recall_score_ternary():
    result = metrics.recall_score_group_summary(
        Y_true_ternary, Y_pred_ternary, sensitive_features=group2, average=None)
    expected_overall = skm.recall_score(Y_true_ternary, Y_pred_ternary, average=None)

    assert np.array_equal(result.overall, expected_overall)


def test_group_recall_score_pos_label():
    result = metrics.recall_score_group_summary(
        Y_true, Y_pred, sensitive_features=groups, pos_label=0)
    expected_overall = skm.recall_score(Y_true, Y_pred, pos_label=0)

    assert np.array_equal(result.overall, expected_overall)

# ======================================================================================


def test_group_roc_auc_score_average():
    result = metrics.roc_auc_score_group_summary(
        Y_true, Y_pred, sensitive_features=groups, average='samples')
    expected_overall = skm.roc_auc_score(Y_true, Y_pred, average='samples')

    assert expected_overall == result.overall


def test_group_roc_auc_score_max_fpr():
    result = metrics.roc_auc_score_group_summary(
        Y_true, Y_pred, sensitive_features=groups, max_fpr=0.5)
    expected_overall = skm.roc_auc_score(Y_true, Y_pred, max_fpr=0.5)

    assert expected_overall == result.overall

# ======================================================================================


def test_group_zero_one_loss_unnormalized():
    result = metrics.zero_one_loss_group_summary(
        Y_true, Y_pred, sensitive_features=groups, normalize=False)

    expected_overall = skm.zero_one_loss(Y_true, Y_pred, False)

    assert result.overall == expected_overall

# =============================================================================================


def test_group_mean_squared_error_multioutput_single_ndarray():
    y_t = np.random.rand(len(groups), 2)
    y_p = np.random.rand(len(groups), 2)
    result = metrics.mean_squared_error_group_summary(
        y_t, y_p, sensitive_features=groups, multioutput='raw_values')

    expected_overall = skm.mean_squared_error(y_t, y_p, multioutput='raw_values')

    assert np.array_equal(result.overall, expected_overall)

# =============================================================================================


def test_group_r2_score_multioutput():
    y_t = np.random.rand(len(groups), 2)
    y_p = np.random.rand(len(groups), 2)
    result = metrics.r2_score_group_summary(
        y_t, y_p, sensitive_features=groups, multioutput='raw_values')

    expected_overall = skm.r2_score(y_t, y_p, multioutput='raw_values')

    assert np.array_equal(result.overall, expected_overall)
    for target_group in np.unique(groups):
        mask = np.asarray(groups) == target_group
        expected = skm.r2_score(y_t[mask], y_p[mask], multioutput='raw_values')
        assert np.array_equal(result.by_group[target_group], expected)

# =============================================================================================


def test_group_mean_squared_error_multioutput_list_ndarray():
    y_t = [np.random.rand(2) for x in groups]
    y_p = [np.random.rand(2) for x in groups]
    result = metrics.mean_squared_error_group_summary(
        y_t, y_p, sensitive_features=groups, multioutput='raw_values')

    expected_overall = skm.mean_squared_error(y_t, y_p, multioutput='raw_values')

    assert np.array_equal(result.overall, expected_overall)

    for target_group in np.unique(groups):
        y_true = []
        y_pred = []
        for i in range(len(groups)):
            if groups[i] == target_group:
                y_true.append(y_t[i])
                y_pred.append(y_p[i])
        expected = skm.mean_squared_error(y_true, y_pred, multioutput='raw_values')
        assert np.array_equal(result.by_group[target_group], expected)
