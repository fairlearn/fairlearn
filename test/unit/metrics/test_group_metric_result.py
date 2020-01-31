# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.metrics import group_accuracy_score
from fairlearn.metrics import group_confusion_matrix

Y_true = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
Y_pred = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
groups = [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
gr_inv = [1-g for g in groups]


class TestComparisons():
    def test_simple_equality(self):
        a = group_accuracy_score(Y_true, Y_pred, groups)
        b = group_accuracy_score(Y_true, Y_pred, groups)

        assert a == b
        assert b == a
        assert not(a != b)
        assert not(b != a)

    def test_simple_inequality(self):
        a = group_accuracy_score(Y_true, Y_pred, groups)
        b = group_accuracy_score(Y_true, Y_pred, gr_inv)

        assert not(a == b)
        assert not(b == a)
        assert a != b
        assert b != a

    def test_complex_equality(self):
        a = group_confusion_matrix(Y_true, Y_pred, groups)
        b = group_confusion_matrix(Y_true, Y_pred, groups)

        assert a == b
        assert b == a
        assert not(a != b)
        assert not(b != a)

    def test_complex_inequality(self):
        a = group_confusion_matrix(Y_true, Y_pred, groups)
        b = group_confusion_matrix(Y_true, Y_pred, gr_inv)

        assert not(a == b)
        assert not(b == a)
        assert a != b
        assert b != a

    def test_mixed_types(self):
        a = group_accuracy_score(Y_true, Y_pred, groups)
        b = group_confusion_matrix(Y_true, Y_pred, groups)

        assert not(a == b)
        assert not(b == a)
        assert a != b
        assert b != a
