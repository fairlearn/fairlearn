# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import fairlearn.metrics as metrics


def test_mean_prediction_unweighted():
    y_pred = [0, 1, 2, 3, 4]
    y_true = None

    result = metrics.mean_prediction(y_true, y_pred)

    assert result == 2


def test_mean_prediction_single():
    y_pred = [42]
    y_true = None

    result = metrics.mean_prediction(y_true, y_pred)

    assert result == 42


def test_mean_prediction_weighted():
    y_pred = [0, 1, 2, 3, 4]
    y_true = None
    weight = [8, 2, 1, 2, 1]

    result = metrics.mean_prediction(y_true, y_pred, weight)

    assert result == 1


def test_mean_prediction_weighted_single():
    y_pred = [42]
    y_true = None
    weight = [2]

    result = metrics.mean_prediction(y_true, y_pred, weight)

    assert result == 42
