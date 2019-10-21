# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.metrics as metrics

# ==============================================
# Specificity (aka True Negative Rate)


def test_specificity_all_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 0, 1]

    result = metrics.specificity_score(y_true, y_pred)
    assert result == 1


def test_specificity_none_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [1, 1, 1, 1, 0]

    result = metrics.specificity_score(y_true, y_pred)
    assert result == 0


def test_specificity_some_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 1, 1]

    result = metrics.specificity_score(y_true, y_pred)
    assert result == 0.75


def test_specificity_some_correct_with_false_negative():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 1, 0, 0]

    result = metrics.specificity_score(y_true, y_pred)
    assert result == 0.75


# ========================================
# miss score (aka False Negative Rate)

def test_miss_all_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 0, 1]

    result = metrics.miss_rate(y_true, y_pred)
    assert result == 0


def test_miss_none_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [1, 1, 1, 1, 0]

    result = metrics.miss_rate(y_true, y_pred)
    assert result == 1


def test_miss_with_false_positive():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 1, 1]

    result = metrics.miss_rate(y_true, y_pred)
    assert result == 0


def test_miss_some_correct():
    y_true = [1, 1, 1, 1, 0, 0]
    y_pred = [0, 0, 1, 0, 0, 1]

    result = metrics.miss_rate(y_true, y_pred)
    assert result == 0.75


# ============================
# Fall-out (aka False Positive Rate)

def test_fallout_all_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 0, 1]

    result = metrics.fallout_rate(y_true, y_pred)
    assert result == 0


def test_fallout_none_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [1, 1, 1, 1, 0]

    result = metrics.fallout_rate(y_true, y_pred)
    assert result == 1


def test_fallout_some_correct():
    y_true = [0, 0, 0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 0, 0, 1, 0]

    result = metrics.fallout_rate(y_true, y_pred)

    assert result == 0.5
