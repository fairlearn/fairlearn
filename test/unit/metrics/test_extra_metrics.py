# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
import sklearn.metrics as skm

import fairlearn.metrics as metrics
from fairlearn.metrics._extra_metrics import _get_labels_for_confusion_matrix

# =============================================


class TestGetLabelsForConfusionMatrix:
    def test_smoke(self):
        r0 = _get_labels_for_confusion_matrix([0, 1], None)
        assert np.array_equal(r0, [0, 1])
        r1 = _get_labels_for_confusion_matrix([-1, 1], None)
        assert np.array_equal(r1, [-1, 1])

    def test_smoke_numeric_pos_label(self):
        r0 = _get_labels_for_confusion_matrix([0, 2], 2)
        assert np.array_equal(r0, [0, 2])
        r1 = _get_labels_for_confusion_matrix([0, 2], 0)
        assert np.array_equal(r1, [2, 0])

    def test_smoke_alpha_pos_label(self):
        r0 = _get_labels_for_confusion_matrix(['a', 'b'], 'b')
        assert np.array_equal(r0, ['a', 'b'])
        r1 = _get_labels_for_confusion_matrix(['a', 'b'], 'a')
        assert np.array_equal(r1, ['b', 'a'])

    def test_single_value_numeric_no_pos_label(self):
        r0 = _get_labels_for_confusion_matrix([0], None)
        assert np.array_equal(r0, [0, 1])
        r1 = _get_labels_for_confusion_matrix([-1], None)
        assert np.array_equal(r1, [-1, 1])
        r2 = _get_labels_for_confusion_matrix([1], None)
        assert np.array_equal(r2, [None, 1])

    def test_single_value_numeric_pos_label(self):
        r0 = _get_labels_for_confusion_matrix([0], 3)
        assert np.array_equal(r0, [0, 3])
        r1 = _get_labels_for_confusion_matrix([0], 0)
        assert np.array_equal(r1, [None, 0])

    def test_single_value_alpha_pos_label(self):
        r0 = _get_labels_for_confusion_matrix(['a'], 'a')
        assert np.array_equal(r0, [None, 'a'])
        r1 = _get_labels_for_confusion_matrix(['a'], 0)
        assert np.array_equal(r1, ['a', 0])

    def test_too_many_values(self):
        expected_msg = "Must have no more than two unique y values"
        expected_msg_no_pos = \
            "If pos_label is not specified, values must be from {0, 1} or {-1, 1}"
        with pytest.raises(ValueError) as e0:
            _get_labels_for_confusion_matrix([0, 1, 2], None)
        # Note that while the following error message should be deterministic
        # from a user PoV, either message is helpful
        assert str(e0.value) in {expected_msg, expected_msg_no_pos}
        # In the following, only one error is possible
        with pytest.raises(ValueError) as e1:
            _get_labels_for_confusion_matrix(['a', 'b', 'c'], 'a')
        assert str(e1.value) == expected_msg

    def test_need_pos_label(self):
        expected_msg = "If pos_label is not specified, values must be from {0, 1} or {-1, 1}"
        with pytest.raises(ValueError) as e0:
            _get_labels_for_confusion_matrix([0, 2], None)
        assert str(e0.value) == expected_msg
        with pytest.raises(ValueError) as e1:
            _get_labels_for_confusion_matrix([-1, 0], None)
        assert str(e1.value) == expected_msg

    def test_pos_label_not_in_data(self):
        expected_msg = "Must have pos_label in y values"
        with pytest.raises(ValueError) as e0:
            _get_labels_for_confusion_matrix([0, 1], -1)
        assert str(e0.value) == expected_msg
        with pytest.raises(ValueError) as e1:
            _get_labels_for_confusion_matrix([4, 2], 3)
        assert str(e1.value) == expected_msg


class TestTPR:
    def test_all_correct(self):
        y_true = [0, 0, 0, 0, 1]
        y_pred = [0, 0, 0, 0, 1]

        result = metrics.true_positive_rate(y_true, y_pred)
        assert result == 1

        result = metrics.true_positive_rate(y_true, y_pred, pos_label=0)
        assert result == 1

    def test_none_correct(self):
        y_true = [0, 0, 0, 0, 1]
        y_pred = [1, 1, 1, 1, 0]

        result = metrics.true_positive_rate(y_true, y_pred)
        assert result == 0

        result = metrics.true_positive_rate(y_true, y_pred, pos_label=0)
        assert result == 0

    def test_some_correct(self):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 0, 0, 0]

        result = metrics.true_positive_rate(y_true, y_pred)
        assert result == 0.25
        result = metrics.true_positive_rate(y_true, y_pred, pos_label=0)
        assert result == 0.75

    def test_against_sklearn(self):
        y_true = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
        y_pred = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]

        result = metrics.true_positive_rate(y_true, y_pred)
        result_skm = skm.recall_score(y_true, y_pred)
        assert result == pytest.approx(result_skm)

        result = metrics.true_positive_rate(y_true, y_pred, pos_label=0)
        result_skm = skm.recall_score(y_true, y_pred, pos_label=0)
        assert result == pytest.approx(result_skm)

    def test_against_sklearn_weighted(self):
        y_true = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
        y_pred = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
        weight = [1, 2, 4, 5, 1, 2, 2, 3, 1, 2, 1, 3, 1, 5, 4, 2]

        result = metrics.true_positive_rate(y_true, y_pred, sample_weight=weight)
        result_skm = skm.recall_score(y_true, y_pred, sample_weight=weight)
        assert result == pytest.approx(result_skm)

        result = metrics.true_positive_rate(y_true, y_pred,
                                            pos_label=0, sample_weight=weight)
        result_skm = skm.recall_score(y_true, y_pred,
                                      pos_label=0, sample_weight=weight)
        assert result == pytest.approx(result_skm)

    def test_tpr_values_alpha(self):
        a = 'a'
        b = 'b'
        y_true = [a, b, a, b, b, b, b, b, a, a, a, a, b, b, a, b]
        y_pred = [a, b, b, a, a, b, a, a, b, b, a, b, a, b, a, a]

        # Use sklearn for validation
        result_a = metrics.true_positive_rate(y_true, y_pred, pos_label=a)
        result_skm_a = skm.recall_score(y_true, y_pred, pos_label=a)
        assert result_a == pytest.approx(result_skm_a)

        result_b = metrics.true_positive_rate(y_true, y_pred, pos_label=b)
        result_skm_b = skm.recall_score(y_true, y_pred, pos_label=b)
        assert result_b == pytest.approx(result_skm_b)

        # Sanity check on the in put data
        assert result_a != result_b


class TestTNR:
    def test_all_correct(self):
        y_true = [0, 0, 0, 0, 1]
        y_pred = [0, 0, 0, 0, 1]

        result = metrics.true_negative_rate(y_true, y_pred)
        assert result == 1

        result = metrics.true_negative_rate(y_true, y_pred, pos_label=0)
        assert result == 1

    def test_none_correct(self):
        y_true = [0, 0, 0, 0, 1]
        y_pred = [1, 1, 1, 1, 0]

        result = metrics.true_negative_rate(y_true, y_pred)
        assert result == 0

        result = metrics.true_negative_rate(y_true, y_pred, pos_label=0)
        assert result == 0

    def test_some_correct(self):
        y_true = [-1, -1, -1, -1, 1]
        y_pred = [-1, -1, -1, 1, 1]

        result = metrics.true_negative_rate(y_true, y_pred)
        assert result == 0.75
        result = metrics.true_negative_rate(y_true, y_pred, pos_label=-1)
        assert result == 1

    def test_some_correct_other_labels(self):
        y_true = ['b', 'b', 'b', 'b', 'a']
        y_pred = ['b', 'b', 'b', 'a', 'a']

        result = metrics.true_negative_rate(y_true, y_pred, pos_label='b')
        assert result == 1
        result = metrics.true_negative_rate(y_true, y_pred, pos_label='a')
        assert result == 0.75

    def test_tnr_some_correct_with_false_negative(self):
        y_true = [0, 0, 0, 0, 1]
        y_pred = [0, 0, 1, 0, 0]

        result = metrics.true_negative_rate(y_true, y_pred)
        assert result == 0.75

    def test_against_sklearn(self):
        y_true = [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1,
                  0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]
        y_pred = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0,
                  1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1]

        actual = metrics.true_negative_rate(y_true, y_pred)
        tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred).ravel()
        assert (tn/(tn+fp)) == actual
        actual = metrics.true_negative_rate(y_true, y_pred, pos_label=0)
        tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel()
        assert (tn/(tn+fp)) == actual

    def test_against_sklearn_weighted(self):
        y_true = [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1,
                  0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]
        y_pred = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0,
                  1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1]
        weights = [1, 2, 3, 5, 3, 2, 5, 3, 5, 1, 5, 3, 5, 2, 3,
                   2, 5, 2, 3, 1, 5, 3, 2, 1, 1, 5, 2, 3, 5, 1]

        actual = metrics.true_negative_rate(y_true, y_pred, sample_weight=weights)
        tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, sample_weight=weights).ravel()
        assert (tn/(tn+fp)) == actual
        actual = metrics.true_negative_rate(y_true, y_pred, pos_label=0, sample_weight=weights)
        tn, fp, fn, tp = skm.confusion_matrix(
            y_true, y_pred,
            labels=[1, 0],
            sample_weight=weights).ravel()
        assert (tn/(tn+fp)) == actual


class TestFNR:

    def test_all_correct(self):
        y_true = [0, 0, 0, 0, 1]
        y_pred = [0, 0, 0, 0, 1]

        result = metrics.false_negative_rate(y_true, y_pred)
        assert result == 0

    def test_none_correct(self):
        y_true = [0, 0, 0, 0, 1]
        y_pred = [1, 1, 1, 1, 0]

        result = metrics.false_negative_rate(y_true, y_pred)
        assert result == 1

    def test_with_false_positive(self):
        y_true = [0, 0, 0, 0, 1]
        y_pred = [0, 0, 0, 1, 1]

        result = metrics.false_negative_rate(y_true, y_pred)
        assert result == 0

    def test_some_correct(self):
        y_true = [1, 1, 1, 1, 0, 0]
        y_pred = [0, 0, 1, 0, 0, 1]

        result = metrics.false_negative_rate(y_true, y_pred)
        assert result == 0.75
        result = metrics.false_negative_rate(y_true, y_pred, pos_label=1)
        assert result == 0.75
        result = metrics.false_negative_rate(y_true, y_pred, pos_label=0)
        assert result == 0.5

    def test_some_correct_other_labels(self):
        y_true = ['a', 'a', 'a', 'a', 'b', 'b']
        y_pred = ['b', 'b', 'a', 'b', 'b', 'a']

        result = metrics.false_negative_rate(y_true, y_pred, pos_label='a')
        assert result == 0.75
        result = metrics.false_negative_rate(y_true, y_pred, pos_label='b')
        assert result == 0.5

    def test_against_sklearn(self):
        y_true = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]
        y_pred = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]

        result = metrics.false_negative_rate(y_true, y_pred)
        result_skm = 1 - skm.recall_score(y_true, y_pred)
        assert result == pytest.approx(result_skm)

        result = metrics.false_negative_rate(y_true, y_pred, pos_label=0)
        result_skm = 1 - skm.recall_score(y_true, y_pred, pos_label=0)
        assert result == pytest.approx(result_skm)

    def test_against_sklearn_weighted(self):
        y_true = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]
        y_pred = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
        weight = [1, 2, 3, 1, 5, 2, 3, 1, 2, 5, 3, 2, 5, 2, 1, 3]

        result = metrics.false_negative_rate(y_true, y_pred, sample_weight=weight)
        result_skm = 1 - skm.recall_score(y_true, y_pred, sample_weight=weight)
        assert result == pytest.approx(result_skm)

        result = metrics.false_negative_rate(y_true, y_pred,
                                             pos_label=0, sample_weight=weight)
        result_skm = 1 - skm.recall_score(y_true, y_pred,
                                          pos_label=0, sample_weight=weight)
        assert result == pytest.approx(result_skm)


class TestFPR:

    def test_all_correct(self):
        y_true = [0, 0, 0, 0, 1]
        y_pred = [0, 0, 0, 0, 1]

        result = metrics.false_positive_rate(y_true, y_pred)
        assert result == 0

    def test_none_correct(self):
        y_true = [0, 0, 0, 0, 1]
        y_pred = [1, 1, 1, 1, 0]

        result = metrics.false_positive_rate(y_true, y_pred)
        assert result == 1

    def test_some_correct(self):
        y_true = [-1, -1, -1, -1, 1, 1, 1]
        y_pred = [-1, 1, 1, -1, -1, 1, -1]

        result = metrics.false_positive_rate(y_true, y_pred)
        assert result == 0.5
        result = metrics.false_positive_rate(y_true, y_pred, pos_label=1)
        assert result == 0.5
        result = metrics.false_positive_rate(y_true, y_pred, pos_label=-1)
        assert result == pytest.approx(0.6666667)

    def test_against_sklearn(self):
        y_true = [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0]
        y_pred = [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]

        actual = metrics.false_positive_rate(y_true, y_pred)
        tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred).ravel()
        assert (fp/(fp+tn)) == actual
        actual = metrics.false_positive_rate(y_true, y_pred, pos_label=0)
        tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel()
        assert (fp/(fp+tn)) == actual

    def test_against_sklearn_weighted(self):
        y_true = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0]
        y_pred = [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]
        weight = [1, 2, 1, 1, 1, 3, 1, 4, 1, 2, 3, 4, 2, 3, 1, 2, 3, 1, 3, 2, 4, 2, 3, 1, 1, 5]

        actual = metrics.false_positive_rate(y_true, y_pred, sample_weight=weight)
        tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, sample_weight=weight).ravel()
        assert (fp/(fp+tn)) == actual
        actual = metrics.false_positive_rate(y_true, y_pred,
                                             pos_label=0, sample_weight=weight)
        tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred,
                                              labels=[1, 0], sample_weight=weight).ravel()
        assert (fp/(fp+tn)) == actual


class TestSingleValueArrays:
    def test_all_zeros(self):
        zeros = np.zeros(10)

        # Should behave as if pos_label=1
        assert metrics.true_positive_rate(zeros, zeros) == 0
        assert metrics.false_positive_rate(zeros, zeros) == 0
        assert metrics.true_negative_rate(zeros, zeros) == 1
        assert metrics.false_negative_rate(zeros, zeros) == 0

        assert metrics.true_positive_rate(zeros, zeros, pos_label=1) == 0
        assert metrics.false_positive_rate(zeros, zeros, pos_label=1) == 0
        assert metrics.true_negative_rate(zeros, zeros, pos_label=1) == 1
        assert metrics.false_negative_rate(zeros, zeros, pos_label=1) == 0

        assert metrics.true_positive_rate(zeros, zeros, pos_label=0) == 1
        assert metrics.false_positive_rate(zeros, zeros, pos_label=0) == 0
        assert metrics.true_negative_rate(zeros, zeros, pos_label=0) == 0
        assert metrics.false_negative_rate(zeros, zeros, pos_label=0) == 0

    def test_all_ones(self):
        ones = np.ones(10)

        # should behave as if pos_label=1
        assert metrics.true_positive_rate(ones, ones) == 1
        assert metrics.false_positive_rate(ones, ones) == 0
        assert metrics.true_negative_rate(ones, ones) == 0
        assert metrics.false_negative_rate(ones, ones) == 0

        assert metrics.true_positive_rate(ones, ones, pos_label=1) == 1
        assert metrics.false_positive_rate(ones, ones, pos_label=1) == 0
        assert metrics.true_negative_rate(ones, ones, pos_label=1) == 0
        assert metrics.false_negative_rate(ones, ones, pos_label=1) == 0

        assert metrics.true_positive_rate(ones, ones, pos_label=0) == 0
        assert metrics.false_positive_rate(ones, ones, pos_label=0) == 0
        assert metrics.true_negative_rate(ones, ones, pos_label=0) == 1
        assert metrics.false_negative_rate(ones, ones, pos_label=0) == 0

    def test_all_negative_ones(self):
        neg_ones = -1 * np.ones(10)

        # Should behave as if pos_label=1
        assert metrics.true_positive_rate(neg_ones, neg_ones) == 0
        assert metrics.false_positive_rate(neg_ones, neg_ones) == 0
        assert metrics.true_negative_rate(neg_ones, neg_ones) == 1
        assert metrics.false_negative_rate(neg_ones, neg_ones) == 0

        assert metrics.true_positive_rate(neg_ones, neg_ones, pos_label=1) == 0
        assert metrics.false_positive_rate(neg_ones, neg_ones, pos_label=1) == 0
        assert metrics.true_negative_rate(neg_ones, neg_ones, pos_label=1) == 1
        assert metrics.false_negative_rate(neg_ones, neg_ones, pos_label=1) == 0

        assert metrics.true_positive_rate(neg_ones, neg_ones, pos_label=-1) == 1
        assert metrics.false_positive_rate(neg_ones, neg_ones, pos_label=-1) == 0
        assert metrics.true_negative_rate(neg_ones, neg_ones, pos_label=-1) == 0
        assert metrics.false_negative_rate(neg_ones, neg_ones, pos_label=-1) == 0
