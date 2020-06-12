# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
import sklearn.metrics as skm

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


def test_get_labels_for_confusion_matrix_bad_unique_count():
    with pytest.raises(ValueError) as exception:
        _get_labels_for_confusion_matrix([0, 1], [1, 2], pos_label=1)
    assert str(exception.value) == "Must have no more than two unique y values"


def test_get_labels_for_confusion_matrix_bad_pos_label():
    with pytest.raises(ValueError) as exception:
        _get_labels_for_confusion_matrix([0, 1], [1, 0], pos_label=2)
    assert str(exception.value) == "Must have pos_label in y values"


def test_get_labels_for_confusion_matrix_ambiguous():
    with pytest.raises(ValueError) as exception:
        _get_labels_for_confusion_matrix([1, 1], [1, 1], None)
    assert str(exception.value) == "Must specify pos_label for degenerate data"


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


def test_tpr_against_sklearn():
    y_true = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
    y_pred = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]

    result = metrics.true_positive_rate(y_true, y_pred)
    result_skm = skm.recall_score(y_true, y_pred)
    assert result == pytest.approx(result_skm)

    result = metrics.true_positive_rate(y_true, y_pred, pos_label=0)
    result_skm = skm.recall_score(y_true, y_pred, pos_label=0)
    assert result == pytest.approx(result_skm)


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


def test_fnr_against_sklearn():
    y_true = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]

    result = metrics.false_negative_rate(y_true, y_pred)
    result_skm = 1 - skm.recall_score(y_true, y_pred)
    assert result == pytest.approx(result_skm)

    result = metrics.false_negative_rate(y_true, y_pred, pos_label=0)
    result_skm = 1 - skm.recall_score(y_true, y_pred, pos_label=0)
    assert result == pytest.approx(result_skm)


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

# ============================
# Single value arrays


def test_all_zeros():
    zeros = np.zeros(10)

    assert metrics.true_positive_rate(zeros, zeros, pos_label=1) == 0
    assert metrics.false_positive_rate(zeros, zeros, pos_label=1) == 0
    assert metrics.true_negative_rate(zeros, zeros, pos_label=1) == 1
    assert metrics.false_negative_rate(zeros, zeros, pos_label=1) == 0

    assert metrics.true_positive_rate(zeros, zeros, pos_label=0) == 1
    assert metrics.false_positive_rate(zeros, zeros, pos_label=0) == 0
    assert metrics.true_negative_rate(zeros, zeros, pos_label=0) == 0
    assert metrics.false_negative_rate(zeros, zeros, pos_label=0) == 0


def test_all_ones():
    ones = np.ones(10)

    assert metrics.true_positive_rate(ones, ones, pos_label=1) == 1
    assert metrics.false_positive_rate(ones, ones, pos_label=1) == 0
    assert metrics.true_negative_rate(ones, ones, pos_label=1) == 0
    assert metrics.false_negative_rate(ones, ones, pos_label=1) == 0

    assert metrics.true_positive_rate(ones, ones, pos_label=0) == 0
    assert metrics.false_positive_rate(ones, ones, pos_label=0) == 0
    assert metrics.true_negative_rate(ones, ones, pos_label=0) == 1
    assert metrics.false_negative_rate(ones, ones, pos_label=0) == 0
