# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
import sklearn.metrics as skm

import fairlearn.metrics as metrics
from fairlearn.metrics._extra_metrics import _get_labels_for_confusion_matrix
from fairlearn.metrics._extra_metrics import _NO_SUCH_ELEMENT

# =============================================


class TestGetLabelsForConfusionMatrix:
    def test_smoke(self):
        r0 = _get_labels_for_confusion_matrix([0, 1], [1, 1], None)
        assert np.array_equal(r0, [0, 1])
        r1 = _get_labels_for_confusion_matrix([-1, 1], [-1, -1], None)
        assert np.array_equal(r1, [-1, 1])

    def test_smoke_numeric_pos_label(self):
        r0 = _get_labels_for_confusion_matrix([0, 2], [2, 0], 2)
        assert np.array_equal(r0, [0, 2])
        r1 = _get_labels_for_confusion_matrix([0, 2], [2, 0], 0)
        assert np.array_equal(r1, [2, 0])

    def test_smoke_alpha_pos_label(self):
        r0 = _get_labels_for_confusion_matrix(['a', 'a'], ['a', 'b'], 'b')
        assert np.array_equal(r0, ['a', 'b'])
        r1 = _get_labels_for_confusion_matrix(['a', 'a'], ['a', 'b'], 'a')
        assert np.array_equal(r1, ['b', 'a'])

    def test_single_value_numeric_no_pos_label(self):
        r0 = _get_labels_for_confusion_matrix([0, 0], [0, 0], None)
        assert np.array_equal(r0, [0, 1])
        r1 = _get_labels_for_confusion_matrix([-1, -1], [-1, -1], None)
        assert np.array_equal(r1, [-1, 1])
        r2 = _get_labels_for_confusion_matrix([1, 1], [1, 1], None)
        assert np.array_equal(r2, [_NO_SUCH_ELEMENT, 1])

    def test_single_value_numeric_pos_label(self):
        r0 = _get_labels_for_confusion_matrix([0, 0], [0, 0], 3)
        assert np.array_equal(r0, [0, 3])
        r1 = _get_labels_for_confusion_matrix([0, 0], [0, 0], 0)
        assert np.array_equal(r1, [_NO_SUCH_ELEMENT, 0])

    def test_single_value_alpha_pos_label(self):
        r0 = _get_labels_for_confusion_matrix(['a', 'a'], ['a', 'a'], 'a')
        assert np.array_equal(r0, [_NO_SUCH_ELEMENT, 'a'])
        r1 = _get_labels_for_confusion_matrix(['a', 'a'], ['a', 'a'], 0)
        assert np.array_equal(r1, ['a', 0])

    def test_too_many_values(self):
        expected_msg = "Must have no more than two unique y values"
        with pytest.raises(ValueError) as e0:
            _get_labels_for_confusion_matrix([0, 1], [1, 2], None)
        assert str(e0.value) == expected_msg
        with pytest.raises(ValueError) as e1:
            _get_labels_for_confusion_matrix(['a', 'b'], ['a', 'c'], 'a')
        assert str(e1.value) == expected_msg

    def test_need_pos_label(self):
        expected_msg = "If pos_label is not specified, values must be take from {0, 1} or {-1, 1}"
        with pytest.raises(ValueError) as e0:
            _get_labels_for_confusion_matrix([0, 2], [2, 0], None)
        assert str(e0.value) == expected_msg
        with pytest.raises(ValueError) as e1:
            _get_labels_for_confusion_matrix([-1, 0], [0, -1], None)
        assert str(e1.value) == expected_msg

    def test_pos_label_not_in_data(self):
        expected_msg = "Must have pos_label in y values"
        with pytest.raises(ValueError) as e0:
            _get_labels_for_confusion_matrix([0, 1], [1, 1], -1)
        assert str(e0.value) == expected_msg
        with pytest.raises(ValueError) as e1:
            _get_labels_for_confusion_matrix([4, 2], [2, 2], 3)
        assert str(e1.value) == expected_msg


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


def test_tpr_values_1_2():
    # Want to ensure that the 'pos_label' is taken to be 2 by default
    # Since this logic is actually in a subroutine, only test for TPR
    y_true = [1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2]
    y_pred = [1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2]

    # Use sklearn for validation
    result = metrics.true_positive_rate(y_true, y_pred)
    # Note that skm.recall_score default pos_label=1, not pos_label=1
    result_skm = skm.recall_score(y_true, y_pred, pos_label=2)
    assert result == pytest.approx(result_skm)

    # Can also test against ourselves
    result_1 = metrics.true_positive_rate(y_true, y_pred, pos_label=1)
    result_2 = metrics.true_positive_rate(y_true, y_pred, pos_label=2)
    assert result_2 == result
    assert result_1 != result

    result_skm_1 = skm.recall_score(y_true, y_pred, pos_label=1)
    assert result_1 == pytest.approx(result_skm_1)


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
