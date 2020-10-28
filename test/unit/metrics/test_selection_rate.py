# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest

import fairlearn.metrics as metrics
from fairlearn.metrics._selection_rate import _EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE


def test_selection_rate_empty():
    with pytest.raises(ValueError) as exc:
        _ = metrics.selection_rate([], [])
    assert _EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE == exc.value.args[0]


def test_selection_rate_single_element():
    assert 1 == metrics.selection_rate([1], [1])
    assert 1 == metrics.selection_rate([0], [1])
    assert 0 == metrics.selection_rate([1], [0])
    assert 0 == metrics.selection_rate([0], [0])


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
