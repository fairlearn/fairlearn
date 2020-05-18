# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.metrics as metrics

# ==============================================
# True Negative Rate


def test_tnr_all_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 0, 1]

    result = metrics.true_negative_rate(y_true, y_pred)
    assert result == 1


def test_tnr_none_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [1, 1, 1, 1, 0]

    result = metrics.true_negative_rate(y_true, y_pred)
    assert result == 0


def test_tnr_some_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 1, 1]

    result = metrics.true_negative_rate(y_true, y_pred)
    assert result == 0.75


def test_tnr_some_correct_with_false_negative():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 1, 0, 0]

    result = metrics.true_negative_rate(y_true, y_pred)
    assert result == 0.75


# ========================================
# False Negative Rate

def test_fnr_all_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 0, 1]

    result = metrics.false_negative_rate(y_true, y_pred)
    assert result == 0


def test_fnr_none_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [1, 1, 1, 1, 0]

    result = metrics.false_negative_rate(y_true, y_pred)
    assert result == 1


def test_fnr_with_false_positive():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 1, 1]

    result = metrics.false_negative_rate(y_true, y_pred)
    assert result == 0


def test_fnr_some_correct():
    y_true = [1, 1, 1, 1, 0, 0]
    y_pred = [0, 0, 1, 0, 0, 1]

    result = metrics.false_negative_rate(y_true, y_pred)
    assert result == 0.75


# ============================
# False Positive Rate

def test_fpr_all_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 0, 1]

    result = metrics.false_positive_rate(y_true, y_pred)
    assert result == 0


def test_fpr_none_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [1, 1, 1, 1, 0]

    result = metrics.false_positive_rate(y_true, y_pred)
    assert result == 1


def test_fpr_some_correct():
    y_true = [0, 0, 0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 0, 0, 1, 0]

    result = metrics.false_positive_rate(y_true, y_pred)

    assert result == 0.5
