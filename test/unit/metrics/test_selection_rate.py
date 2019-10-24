# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

import fairlearn.metrics as metrics


def test_selection_rate_unweighted():
    y_true = [0, 0, 0, 0, 0, 0, 0, 0]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 1]

    result = metrics.selection_rate(y_true, y_pred)

    assert result == 0.625


def test_selection_rate_weighted():
    y_true = [0, 0, 0, 0, 0, 0, 0, 0]
    y_pred = [0, 1, 1, 0, 0, 0, 0, 0]
    weight = [1, 2, 3, 4, 1, 2, 1, 2]

    result = metrics.selection_rate(y_true, y_pred, sample_weight=weight)

    assert result == 0.3125


def test_selection_rate_non_numeric():
    a = "a"
    b = "b"
    y_true = [a, b, a, b, a, b, a, b]
    y_pred = [a, a, a, b, b, b, a, a]

    result = metrics.selection_rate(y_true, y_pred, pos_label=b)

    assert result == 0.375


def test_group_selection_rate():
    a = "a"
    b = "b"
    q = "q"
    r = "r"

    y_true = [a, b, a, b, b, b, a, a]
    y_pred = [a, b, a, a, b, b, a, b]
    weight = [1, 2, 3, 4, 1, 2, 4, 8]
    groups = [r, r, r, r, q, q, q, q]

    result = metrics.group_selection_rate(y_true, y_pred, groups,
                                          pos_label=b,
                                          sample_weight=weight)

    assert result.overall == 0.52
    assert result.by_group[r] == 0.2
    assert result.by_group[q] == pytest.approx(0.73333333)
