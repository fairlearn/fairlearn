# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

import fairlearn.metrics as metrics


class MetricMock:
    def mock_func(self, y_actual, y_predict, sample_weight=None):
        self.y_a = y_actual
        self.y_p = y_predict
        self.s_w = sample_weight

        if sample_weight is None:
            return np.sum(y_actual)
        else:
            return np.sum(np.multiply(y_actual, sample_weight))


class TestMetricByGroups:
    def test_smoke(self):
        y_a = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 1, 1, 1, 1]
        mm = MetricMock()

        result = metrics.metric_by_groups(mm.mock_func, y_a, y_p, gid)

        assert result.metric == 5
        assert len(result.group_metric) == 2
        assert result.group_metric[0] == 2
        assert result.group_metric[1] == 3

    def test_string_groups(self):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        y_a = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [a, a, a, b, b, c, c, c]
        mm = MetricMock()

        result = metrics.metric_by_groups(mm.mock_func, y_a, y_p, gid)

        assert result.metric == 5
        assert len(result.group_metric) == 3
        assert result.group_metric[a] == 1
        assert result.group_metric[b] == 1
        assert result.group_metric[c] == 3

    def test_with_weights(self):
        y_a = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 1, 1, 2, 2]
        s_w = [1, 1, 1, 1, 2, 2, 3, 3]
        mm = MetricMock()

        result = metrics.metric_by_groups(mm.mock_func, y_a, y_p, gid, sample_weight=s_w)

        assert result.metric == 10
        assert len(result.group_metric) == 3
        assert result.group_metric[0] == 2
        assert result.group_metric[1] == 2
        assert result.group_metric[2] == 6

    def test_actual_not_0_1(self):
        y_a = [0, 2, 0, 2, 0, 2, 2, 2]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 1, 1, 2, 2]
        mm = MetricMock()

        with pytest.raises(ValueError) as exCtxt:
            _ = metrics.metric_by_groups(mm.mock_func, y_a, y_p, gid)

        assert exCtxt.value.args[0] == "Array y_actual contains values other than 0 and 1"

    def test_predict_not_0_1(self):
        y_a = [0, 1, 1, 1, 1, 0, 0, 1]
        y_p = [0, 2, 0, 2, 0, 2, 2, 2]
        gid = [0, 0, 0, 0, 1, 1, 2, 2]
        mm = MetricMock()

        with pytest.raises(ValueError) as exCtxt:
            _ = metrics.metric_by_groups(mm.mock_func, y_a, y_p, gid)

        assert exCtxt.value.args[0] == "Array y_predict contains values other than 0 and 1"


'''
def test_true_positive_rate_smoke():
    y_actual = [0, 0, 1, 1, 0, 1, 1, 1]
    y_predict = [0, 1, 1, 1, 1, 0, 0, 1]
    group_ids = [0, 0, 0, 0, 1, 1, 1, 1]

    result = metrics.true_positive_rate(y_actual, y_predict, group_ids)

    assert result.metric == 0.6
    assert result.group_metric[0] == 1.0
    assert result.group_metric[1] == pytest.approx(0.33333333333)


def test_true_positive_rate_smoke_chars():
    y_actual = [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]
    y_predict = [0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
    group_ids = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    result = metrics.true_positive_rate(y_actual, y_predict, group_ids)

    assert result.metric == 0.5
    assert result.group_metric[0] == 1.0
    assert result.group_metric[1] == pytest.approx(0.333333333333)
    assert result.group_metric[2] == 0


def test_selection_rate_smoke():
    y_a = [0, 0, 1, 1, 0, 1, 1, 1]
    y_p = [0, 0, 1, 1, 0, 0, 0, 1]
    grp = [0, 0, 0, 0, 1, 1, 1, 1]

    result = metrics.selection_rate(y_a, y_p, grp)

    assert result.metric == 0.375
    assert result.group_metric[0] == 0.5
    assert result.group_metric[1] == 0.25


def test_demographic_disparity():
    y_a = [0, 0, 1, 1, 0, 1, 1, 1]
    y_p = [0, 0, 1, 1, 0, 0, 0, 1]
    grp = [0, 0, 0, 0, 1, 1, 1, 1]

    demographic_disparity = metrics.make_disparity_metric(metrics.selection_rate, 'ratio')

    result = demographic_disparity(y_a, y_p, grp)

    assert result.disparity == 0.5


def test_unequalised_opportunity():
    y_a = [0, 0, 1, 1, 0, 1, 1, 1]
    y_p = [0, 0, 1, 1, 0, 0, 0, 1]
    grp = [0, 0, 0, 0, 1, 1, 1, 1]

    unequal_opportunity = metrics.make_disparity_metric(metrics.true_positive_rate, 'ratio')

    result = unequal_opportunity(y_a, y_p, grp)

    assert result.disparity == pytest.approx(0.66666667)


def test_non_binary_y_actual():
    y_a = [0, 0, 1, 1, 0, 1, 1, 2]
    y_p = [0, 0, 1, 1, 0, 0, 0, 1]
    grp = [0, 0, 0, 0, 1, 1, 1, 1]

    with pytest.raises(ValueError) as exCtxt:
        _ = metrics.selection_rate(y_a, y_p, grp)

    assert exCtxt.value.args[0] == "Array y_actual contains values other than 0 and 1"


def test_non_binary_y_predict():
    y_a = [0, 0, 1, 1, 0, 1, 1, 1]
    y_p = [0, 0, 1, 1, 0, 0, 0, 2]
    grp = [0, 0, 0, 0, 1, 1, 1, 1]

    with pytest.raises(ValueError) as exCtxt:
        _ = metrics.selection_rate(y_a, y_p, grp)

    assert exCtxt.value.args[0] == "Array y_predict contains values other than 0 and 1"
'''
