# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

import fairlearn.metrics as metrics


def test_true_positive_rate_smoke():
    y_actual = [0, 0, 1, 1]
    y_predict = [1, 0, 1, 1]
    group_ids = [0, 1, 0, 1]

    result = metrics.true_positive_rate(y_actual, y_predict, group_ids)

    assert result.metric == pytest.approx(0.6666666666667)
    assert result.group_metric[0] == 0.5
    assert result.group_metric[1] == 1.0
