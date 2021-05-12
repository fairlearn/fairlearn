# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import math
import pytest

import fairlearn.metrics as metrics


def test_all_correct():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 1, 1, 1, 1]

    result = metrics._balanced_root_mean_squared_error(y_true, y_pred)

    assert result == 0


def test_all_incorrect():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [1, 1, 1, 1, 0, 0, 0, 0]

    result = metrics._balanced_root_mean_squared_error(y_true, y_pred)

    assert result == 1


def test_weighted():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 0, 0, 1]
    weight = [1, 2, 3, 4, 5, 6, 7, 8]

    result = metrics._balanced_root_mean_squared_error(y_true, y_pred, weight)

    expected_0 = math.sqrt(4.0/10)
    expected_1 = math.sqrt(18.0/26)

    assert (expected_0+expected_1)/2 == result


def test_probability_predict():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.5, 0.75]

    result = metrics._balanced_root_mean_squared_error(y_true, y_pred)

    expected_0 = 0.5
    expected_1 = 0.25

    assert (expected_0+expected_1)/2 == result


@pytest.mark.parametrize("y_true", [[0, 0, 0], [1, 1, 1], [0, 0, 2], [1, 1, 2], [0, 1, 2]])
def test_y_true_not_0_1(y_true):
    y_pred = [0.5, 0.5, 0.5]

    message = "Only 0 and 1 are allowed in y_true and both must be present"
    with pytest.raises(ValueError) as execInfo:
        _ = metrics._balanced_root_mean_squared_error(y_true, y_pred)
    assert message == execInfo.value.args[0]
