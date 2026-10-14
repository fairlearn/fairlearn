# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

import fairlearn.metrics as metrics
from fairlearn.metrics._base_metrics import _EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE


def test_selection_rate_empty():
    with pytest.raises(ValueError) as exc:
        _ = metrics.selection_rate([], [])
    assert exc.value.args[0] == _EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    (
        ([1], [1], 1),
        ([0], [1], 1),
        ([1], [0], 0),
        ([0], [0], 0),
        (1, 1, 1),
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        ([False], [False], 0),
        ([True], [True], 1),
        (False, False, 0),
        (True, True, 1),
    ),
)
def test_selection_rate_single_element(y_true, y_pred, expected):
    assert expected == metrics.selection_rate(y_true, y_pred)


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


@pytest.mark.parametrize("y_pred", [[1], [0]])
def test_selection_rate_weighted_single_element_is_scalar(y_pred):
    result = metrics.selection_rate([1], y_pred, sample_weight=[2])

    assert np.ndim(result) == 0
    assert result == y_pred[0]


def test_demographic_parity_difference_weighted_with_single_member_group():
    # Group "b" has a single sample. Its selection rate must be a scalar, or
    # the between-groups difference comes out wrong.
    y_true = [0, 1, 1, 0]
    y_pred = [1, 0, 1, 1]
    sensitive_features = ["a", "a", "a", "b"]
    weight = [1, 1, 2, 3]

    mf = metrics.MetricFrame(
        metrics=metrics.selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"sample_weight": weight},
    )
    assert mf.by_group.to_dict() == {"a": 0.75, "b": 1.0}

    result = metrics.demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features, sample_weight=weight
    )
    assert result == pytest.approx(0.25)


def test_selection_rate_non_numeric():
    a = "a"
    b = "b"
    y_true = [a, b, a, b, a, b, a, b]
    y_pred = [a, a, a, b, b, b, a, a]

    result = metrics.selection_rate(y_true, y_pred, pos_label=b)

    assert result == 0.375
