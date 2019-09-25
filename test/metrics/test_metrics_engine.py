# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

import fairlearn.metrics as metrics


def mock_func(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        return np.sum(y_true)
    else:
        return np.sum(np.multiply(y_true, sample_weight))


class TestMetricByGroups:
    def test_smoke(self):
        y_a = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 1, 1, 1, 1]

        result = metrics.metric_by_groups(mock_func, y_a, y_p, gid)

        assert result.metric == 5
        assert len(result.group_metrics) == 2
        assert result.group_metrics[0] == 2
        assert result.group_metrics[1] == 3
        assert result.min_metric == 2
        assert result.min_metric_groups == {0}
        assert result.max_metric == 3
        assert result.max_metric_groups == {1}
        assert result.metric_range == 1
        assert result.metric_range_ratio == 1.5

    def test_string_groups(self):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        y_a = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [a, a, a, b, b, c, c, c]

        result = metrics.metric_by_groups(mock_func, y_a, y_p, gid)

        assert result.metric == 5
        assert len(result.group_metrics) == 3
        assert result.group_metrics[a] == 1
        assert result.group_metrics[b] == 1
        assert result.group_metrics[c] == 3
        assert result.min_metric == 1
        assert result.min_metric_groups == {a, b}
        assert result.max_metric == 3
        assert result.max_metric_groups == {c}
        assert result.metric_range == 2
        assert result.metric_range_ratio == 3

    def test_with_weights(self):
        y_a = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 1, 1, 2, 2]
        s_w = [1, 1, 1, 1, 2, 2, 3, 3]

        result = metrics.metric_by_groups(mock_func, y_a, y_p, gid, sample_weight=s_w)

        assert result.metric == 10
        assert len(result.group_metrics) == 3
        assert result.group_metrics[0] == 2
        assert result.group_metrics[1] == 2
        assert result.group_metrics[2] == 6
        assert result.min_metric == 2
        assert result.min_metric_groups == {0, 1}
        assert result.max_metric == 6
        assert result.max_metric_groups == {2}
        assert result.metric_range == 4
        assert result.metric_range_ratio == 3

    def test_actual_not_0_1(self):
        y_a = [0, 2, 0, 2, 0, 2, 2, 2]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 1, 1, 2, 2]

        with pytest.raises(ValueError) as exCtxt:
            _ = metrics.metric_by_groups(mock_func, y_a, y_p, gid)

        assert exCtxt.value.args[0] == "Array y_true contains values other than 0 and 1"

    def test_predict_not_0_1(self):
        y_a = [0, 1, 1, 1, 1, 0, 0, 1]
        y_p = [0, 2, 0, 2, 0, 2, 2, 2]
        gid = [0, 0, 0, 0, 1, 1, 2, 2]

        with pytest.raises(ValueError) as exCtxt:
            _ = metrics.metric_by_groups(mock_func, y_a, y_p, gid)

        assert exCtxt.value.args[0] == "Array y_pred contains values other than 0 and 1"


class TestMakeGroupMetric:
    def test_smoke(self):
        y_a = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 1, 1, 1, 1]

        grouped_metric_func = metrics.make_group_metric(mock_func)
        result = grouped_metric_func(y_a, y_p, gid)
        assert result.metric == 5
        assert len(result.group_metrics) == 2
        assert result.group_metrics[0] == 2
        assert result.group_metrics[1] == 3
        assert result.min_metric == 2
        assert result.max_metric == 3
        assert result.min_metric_groups == {0}
        assert result.max_metric_groups == {1}
        assert result.metric_range == 1
        assert result.metric_range_ratio == 1.5

    def test_keys_and_weights(self):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        y_a = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [a, a, a, b, b, c, c, c]
        s_w = [1, 1, 1, 5, 5, 7, 7, 7]

        grouped_metric_func = metrics.make_group_metric(mock_func)
        result = grouped_metric_func(y_a, y_p, gid, s_w)
        assert result.metric == 27
        assert len(result.group_metrics) == 3
        assert result.group_metrics[a] == 1
        assert result.group_metrics[b] == 5
        assert result.group_metrics[c] == 21
        assert result.min_metric == 1
        assert result.max_metric == 21
        assert result.min_metric_groups == {a}
        assert result.max_metric_groups == {c}
        assert result.metric_range == 20
        assert result.metric_range_ratio == 21
