# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.metrics as metrics


def test_true_positive_rate_smoke():
    y_actual = [0, 0, 1, 1]
    y_predict = [1, 0, 1, 0]
    group_ids = [0, 1, 0, 1]

    result = metrics.true_positive_rate(y_actual, y_predict, group_ids)

    assert result.metric == 0.5
