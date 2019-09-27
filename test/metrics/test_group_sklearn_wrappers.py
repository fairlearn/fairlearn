# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

import fairlearn.metrics as metrics

# ======================================================

Y_true = [0, 0, 0, 1, 1, 1, 0, 0]
Y_pred = [1, 0, 1, 0, 1, 0, 0, 1]
groups = [3, 4, 1, 0, 0, 0, 3, 2]
weight = [1, 2, 3, 1, 2, 3, 4, 2]

# =======================================================

supported_metrics = [metrics.group_precision_score]

# =======================================================


@pytest.mark.parametrize("group_metric_func", supported_metrics)
def test_metric_unweighted(group_metric_func):
    result = group_metric_func(Y_true, Y_pred, groups)
    assert len(result.by_group) == 5


@pytest.mark.parametrize("group_metric_func", supported_metrics)
def test_metric_weighted(group_metric_func):
    result = group_metric_func(Y_true, Y_pred, groups, weight)
    assert len(result.by_group) == 5
