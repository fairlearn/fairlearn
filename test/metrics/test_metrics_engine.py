# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

import fairlearn.metrics as metrics

# ===========================================================

# Conversions from Python lists to our supported datatypes


def identity(X):
    return X


def tondarray(X):
    return np.asarray(X)


def tondarray2d(X):
    # ndarray where second dimension is of length 1
    arr = np.asarray(X)
    arr = np.expand_dims(arr, 1)
    assert len(arr.shape) == 2
    return arr


def topandasseries(X):
    return pd.Series(X)


def topandasdf(X):
    return pd.DataFrame(X)


supported_conversions = [identity, tondarray, tondarray2d, topandasseries, topandasdf]

# ===========================================================


def mock_func(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        return np.sum(y_true)
    else:
        return np.sum(np.multiply(y_true, sample_weight))


class TestMetricByGroups:
    @pytest.mark.parametrize("transform_gid", supported_conversions)
    @pytest.mark.parametrize("transform_y_p", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_smoke(self, transform_y_a, transform_y_p, transform_gid):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([0, 0, 0, 0, 1, 1, 1, 1])

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

    @pytest.mark.parametrize("transform_gid", supported_conversions)
    @pytest.mark.parametrize("transform_y_p", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_string_groups(self, transform_y_a, transform_y_p, transform_gid):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([a, a, a, b, b, c, c, c])

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

    @pytest.mark.parametrize("transform_s_w", supported_conversions)
    @pytest.mark.parametrize("transform_gid", supported_conversions)
    @pytest.mark.parametrize("transform_y_p", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_with_weights(self, transform_y_a, transform_y_p, transform_gid, transform_s_w):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([0, 0, 0, 0, 1, 1, 2, 2])
        s_w = transform_s_w([1, 1, 1, 1, 2, 2, 3, 3])

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

    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_true_not_0_1(self, transform_y_a):
        y_a = transform_y_a([0, 2, 0, 2, 0, 2, 2, 2])
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 1, 1, 2, 2]

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.metric_by_groups(mock_func, y_a, y_p, gid)

        assert exception_context.value.args[0] == "Array y_true contains values other than 0 and 1"

    @pytest.mark.parametrize("transform_y_p", supported_conversions)
    def test_pred_not_0_1(self, transform_y_p):
        y_a = [0, 1, 1, 1, 1, 0, 0, 1]
        y_p = transform_y_p([0, 2, 0, 2, 0, 2, 2, 2])
        gid = [0, 0, 0, 0, 1, 1, 2, 2]

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.metric_by_groups(mock_func, y_a, y_p, gid)

        assert exception_context.value.args[0] == "Array y_pred contains values other than 0 and 1"

    @pytest.mark.parametrize("transform_y_p", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_true_predict_length_mismatch(self, transform_y_a, transform_y_p):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0])
        gid = [0, 0, 0, 0, 1, 1, 2, 2]
        s_w = [1, 1, 1, 1, 2, 2, 3, 3]

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.metric_by_groups(mock_func, y_a, y_p, gid, s_w)

        assert exception_context.value.args[0] == "Array y_pred is not the same size as y_true"

    @pytest.mark.parametrize("transform_gid", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_true_group_length_mismatch(self, transform_y_a, transform_gid):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = [0, 1, 1, 1, 1, 0, 0, 0]
        gid = transform_gid([0, 0, 0, 0, 1, 1, 2])
        s_w = [1, 1, 1, 1, 2, 2, 3, 3]

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.metric_by_groups(mock_func, y_a, y_p, gid, s_w)

        assert exception_context.value.args[0] == "Array group_data is not the same size as y_true"

    @pytest.mark.parametrize("transform_s_w", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_true_weight_length_mismatch(self, transform_y_a, transform_s_w):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = [0, 1, 1, 1, 1, 0, 0, 0]
        gid = [0, 0, 0, 0, 1, 1, 2, 3]
        s_w = transform_s_w([1, 1, 1, 1, 2, 2, 3])

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.metric_by_groups(mock_func, y_a, y_p, gid, s_w)

        assert exception_context.value.args[0] == "Array sample_weight is not the same size as y_true"


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

    @pytest.mark.parametrize("transform_s_w", supported_conversions)
    @pytest.mark.parametrize("transform_gid", supported_conversions)
    @pytest.mark.parametrize("transform_y_p", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_keys_and_weights(self, transform_y_a, transform_y_p, transform_gid, transform_s_w):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        z = "something_longer"
        y_a = transform_y_a([0, 1, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([a, z, a, b, b, c, c, c])
        s_w = transform_s_w([1, 1, 1, 5, 5, 7, 7, 7])

        grouped_metric_func = metrics.make_group_metric(mock_func)
        result = grouped_metric_func(y_a, y_p, gid, s_w)
        assert result.metric == 28
        assert len(result.group_metrics) == 4
        assert result.group_metrics[a] == 1
        assert result.group_metrics[b] == 5
        assert result.group_metrics[c] == 21
        assert result.group_metrics[z] == 1
        assert result.min_metric == 1
        assert result.max_metric == 21
        assert result.min_metric_groups == {a, z}
        assert result.max_metric_groups == {c}
        assert result.metric_range == 20
        assert result.metric_range_ratio == 21
