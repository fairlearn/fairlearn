# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

import fairlearn.metrics as metrics
from fairlearn.metrics._extra_metrics import _get_labels_for_confusion_matrix

# =============================================


def test_get_labels_for_confusion_matrix_smoke():
    y_true = [0, 1]
    y_pred = [1, 1]

    result = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label=1)
    assert np.array_equal(result, [0, 1])
    result = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label=0)
    assert np.array_equal(result, [1, 0])

def test_get_labels_for_confusion_matrix_too_many():
    y_true=[0,1]
    y_pred = [1,2]

    with 


# ==============================================
# True Positive Rate

def test_tpr_all_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 0, 1]

    result = metrics.true_positive_rate(y_true, y_pred)
    assert result == 1

    result = metrics.true_positive_rate(y_true, y_pred, pos_label=0)
    assert result == 1


def test_tpr_none_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [1, 1, 1, 1, 0]

    result = metrics.true_positive_rate(y_true, y_pred)
    assert result == 0

    result = metrics.true_positive_rate(y_true, y_pred, pos_label=0)
    assert result == 0


def test_tpr_some_correct():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 0, 0, 0]

    result = metrics.true_positive_rate(y_true, y_pred)
    assert result == 0.25
    result = metrics.true_positive_rate(y_true, y_pred, pos_label=0)
    assert result == 0.75

# ==============================================
# True Negative Rate


def test_tnr_all_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 0, 1]

    result = metrics.true_negative_rate(y_true, y_pred)
    assert result == 1

    result = metrics.true_negative_rate(y_true, y_pred, pos_label=0)
    assert result == 1


def test_tnr_none_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [1, 1, 1, 1, 0]

    result = metrics.true_negative_rate(y_true, y_pred)
    assert result == 0

    result = metrics.true_negative_rate(y_true, y_pred, pos_label=0)
    assert result == 0


def test_tnr_some_correct():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 1, 1]

    result = metrics.true_negative_rate(y_true, y_pred)
    assert result == 0.75
    result = metrics.true_negative_rate(y_true, y_pred, pos_label=0)
    assert result == 1


def test_tnr_some_correct_other_labels():
    y_true = ['b', 'b', 'b', 'b', 'a']
    y_pred = ['b', 'b', 'b', 'a', 'a']

    result = metrics.true_negative_rate(y_true, y_pred, pos_label='b')
    assert result == 1
    result = metrics.true_negative_rate(y_true, y_pred, pos_label='a')
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
    result = metrics.false_negative_rate(y_true, y_pred, pos_label=1)
    assert result == 0.75
    result = metrics.false_negative_rate(y_true, y_pred, pos_label=0)
    assert result == 0.5


def test_fnr_some_correct_other_labels():
    y_true = ['a', 'a', 'a', 'a', 'b', 'b']
    y_pred = ['b', 'b', 'a', 'b', 'b', 'a']

    result = metrics.false_negative_rate(y_true, y_pred, pos_label='a')
    assert result == 0.75
    result = metrics.false_negative_rate(y_true, y_pred, pos_label='b')
    assert result == 0.5


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
    result = metrics.false_positive_rate(y_true, y_pred, pos_label=1)
    assert result == 0.5
    result = metrics.false_positive_rate(y_true, y_pred, pos_label=0)
    assert result == pytest.approx(0.6666667)
