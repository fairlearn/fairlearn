# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.metrics import group_accuracy_score
from fairlearn.metrics import group_confusion_matrix


class TestComparisons():
    Y_true = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    Y_pred = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    groups = [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    gr_inv = [1-g for g in groups]

    def test_simple_equality(self):
        a = group_accuracy_score(TestComparisons.Y_true,
                                 TestComparisons.Y_pred,
                                 TestComparisons.groups)
        b = group_accuracy_score(TestComparisons.Y_true,
                                 TestComparisons.Y_pred,
                                 TestComparisons.groups)

        assert a == b
        assert b == a
        assert not(a != b)
        assert not(b != a)

    def test_simple_inequality(self):
        a = group_accuracy_score(TestComparisons.Y_true,
                                 TestComparisons.Y_pred,
                                 TestComparisons.groups)
        b = group_accuracy_score(TestComparisons.Y_true,
                                 TestComparisons.Y_pred,
                                 TestComparisons.gr_inv)

        assert not(a == b)
        assert not(b == a)
        assert a != b
        assert b != a

    def test_complex_equality(self):
        a = group_confusion_matrix(TestComparisons.Y_true,
                                   TestComparisons.Y_pred,
                                   TestComparisons.groups)
        b = group_confusion_matrix(TestComparisons.Y_true,
                                   TestComparisons.Y_pred,
                                   TestComparisons.groups)

        assert a == b
        assert b == a
        assert not(a != b)
        assert not(b != a)

    def test_complex_inequality(self):
        a = group_confusion_matrix(TestComparisons.Y_true,
                                   TestComparisons.Y_pred,
                                   TestComparisons.groups)
        b = group_confusion_matrix(TestComparisons.Y_true,
                                   TestComparisons.Y_pred,
                                   TestComparisons.gr_inv)

        assert not(a == b)
        assert not(b == a)
        assert a != b
        assert b != a

    def test_mixed_types(self):
        a = group_accuracy_score(TestComparisons.Y_true,
                                 TestComparisons.Y_pred,
                                 TestComparisons.groups)
        b = group_confusion_matrix(TestComparisons.Y_true,
                                   TestComparisons.Y_pred,
                                   TestComparisons.groups)

        assert not(a == b)
        assert not(b == a)
        assert a != b
        assert b != a


class TestSerialization:
    y_true = [0, 1, 1, 1, 0, 0, 1, 1]
    y_pred = [1, 1, 0, 1, 1, 1, 1, 0]
    groups = [0, 0, 0, 0, 1, 1, 1, 1]

    def test_to_dict_smoke(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)

        expected = dict()
        expected['overall'] = 0.375
        expected['by_group'] = dict()
        expected['by_group'][0] = 0.5
        expected['by_group'][1] = 0.25

        assert a.to_dict() == expected

    def test_to_dict_overall_only(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)

        expected = dict()
        expected['overall'] = 0.375
        actual = a.to_dict(include_derived=False, include_by_group=False)
        assert expected == actual

    def test_to_dict_by_group_only(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)

        expected = dict()
        expected['by_group'] = dict()
        expected['by_group'][0] = 0.5
        expected['by_group'][1] = 0.25
        actual = a.to_dict(include_overall=False)
        assert expected == actual

    def test_to_dict_derived_only(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)

        expected = dict()
        expected['maximum'] = 0.5
        expected['minimum'] = 0.25
        expected['range'] = 0.25
        expected['range_ratio'] = 0.5
        actual = a.to_dict(include_derived=True, include_by_group=False, include_overall=False)
        assert expected == actual

    def test_to_dict_all(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)

        expected = dict()
        expected['overall'] = 0.375
        expected['by_group'] = dict()
        expected['by_group'][0] = 0.5
        expected['by_group'][1] = 0.25
        expected['maximum'] = 0.5
        expected['minimum'] = 0.25
        expected['range'] = 0.25
        expected['range_ratio'] = 0.5
        actual = a.to_dict(include_derived=True)
        assert expected == actual

    def test_to_dict_empty(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)

        expected = dict()
        actual = a.to_dict(include_overall=False, include_by_group=False, include_derived=False)
        assert expected == actual

    def test_repr(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)
        # It appears the order of the keys in the serialized dict
        # is not always the same
        expected = [r"{'overall': 0.375, 'by_group': {0: 0.5, 1: 0.25}}",
                    r"{'by_group': {0: 0.5, 1: 0.25}, 'overall': 0.375}"]
        actual = repr(a)
        assert actual in expected

    def test_to_pandas_series_smoke(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)

        actual = a.to_pandas_series()
        assert actual['overall'] == a.overall
        assert actual['0'] == a.by_group[0]
        assert actual['1'] == a.by_group[1]

    def test_to_pandas_series_overall_only(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)

        actual = a.to_pandas_series(include_overall=True,
                                    include_by_group=False,
                                    include_derived=False)
        assert len(actual) == 1
        assert actual['overall'] == a.overall

    def test_to_pandas_series_by_group_only(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)

        actual = a.to_pandas_series(include_overall=False,
                                    include_by_group=True,
                                    include_derived=False)
        assert len(actual) == 2
        assert actual['0'] == a.by_group[0]
        assert actual['1'] == a.by_group[1]

    def test_to_pandas_series_derived_only(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)

        actual = a.to_pandas_series(include_overall=False,
                                    include_by_group=False,
                                    include_derived=True)
        assert len(actual) == 4
        assert actual['maximum'] == 0.5
        assert actual['minimum'] == 0.25
        assert actual['range'] == 0.25
        assert actual['range_ratio'] == 0.5

    def test_to_pandas_series_all_with_title(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)

        title = "my_title"
        actual = a.to_pandas_series(title=title,
                                    include_derived=True)
        assert actual.name == title
        assert len(actual) == 7
        assert actual['maximum'] == 0.5
        assert actual['minimum'] == 0.25
        assert actual['range'] == 0.25
        assert actual['range_ratio'] == 0.5
        assert actual['0'] == a.by_group[0]
        assert actual['1'] == a.by_group[1]
        assert actual['overall'] == a.overall

    def test_to_pandas_series_empty(self):
        a = group_accuracy_score(TestSerialization.y_true,
                                 TestSerialization.y_pred,
                                 TestSerialization.groups)

        title = "my_title"
        actual = a.to_pandas_series(title=title,
                                    include_overall=False,
                                    include_by_group=False,
                                    include_derived=False)
        assert actual.name == title
        assert len(actual) == 0
