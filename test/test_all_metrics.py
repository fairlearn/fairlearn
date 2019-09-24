# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

import fairlearn.metrics as metrics


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

    assert result.disparity == pytest.approx(0.3333333333333)
